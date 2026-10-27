"""Pre-experiment parameter tuning by successive halving.

For every tunable algorithm of an experiment (``SPACES``: step sizes etc.), all candidates are run on a few
separate tuning scenarios (own seed, short signals) for ``min_frames`` frames. The score of a candidate is
the mean misalignment [dB] over the second half of the frames (diverged = inf; a candidate that diverges on
any scenario is dropped). The better half survives and is re-run for twice as many frames, until one
candidate is left, the best one is unchanged and converged (score <= ``conv_db``), or ``max_frames`` is
reached. Tuning runs separately for every grid point (all grid axes); the winners are stored in
``tuned/<experiment>.json`` under the point key and applied per job by ``alg_spec`` in ``run_experiments.py``.
The tuned table is part of the manifest of the full run.

Parameters that define the experiment (lambda, prior, eta, ...) are not tuned.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .experiments import (
    _run_alg,
    _seed_key,
    make_jobs,
    make_scenario,
    point_key,
    precompute_priors,
)
from .runner import _init_worker, git_info


def _r(v: float) -> float:
    return float(f"{v:.3g}")


_MU = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
SPACES = {
    "dabsi": dict(mu=[0.1, 0.2, 0.4, 0.6, 0.8], rho=[0.5, 1.0, 2.0]),
    "dgd": dict(alpha=[_r(v) for v in np.logspace(-7, -3, 9)]),
    "dsaawet": dict(alpha=[_r(v) for v in np.logspace(-5, 0, 11)]),
    "mcqn": dict(mu=_MU),
    "spmcqn": dict(mu=_MU),
}


def _score_task(texp: dict, job: dict, cands: list, T: int, cache_dir: str) -> list:
    """Scenario of ``job`` once, then all candidates ``(name, idx, spec)``; returns ``(idx, score_db)``."""
    scn = make_scenario(texp, job)
    T = min(T, len(range(0, scn.noisy.shape[0] - 4 * texp["L"], texp["L"])))
    out = []
    for name, i, spec in cands:
        mis, _ = _run_alg(name, spec, texp, scn, _seed_key(job), Path(cache_dir), T)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 20 * np.log10(np.asarray(mis[:, 0], dtype=float))
        tail = db[len(db) // 2 :]
        out.append((i, float(tail.mean()) if np.isfinite(tail).all() else math.inf))
    return out


def _evaluate(texp, jobs, name, spec, cands, idxs, T, pool, procs, cache_dir) -> dict:
    """Mean score over the tuning scenarios for each candidate in ``idxs``."""
    n_chunks = max(
        1, math.ceil(procs / len(jobs))
    )  # keep all processes busy with few scenarios
    chunks = [c for c in (idxs[k::n_chunks] for k in range(n_chunks)) if c]
    tasks = [
        (j, [(name, i, {**spec, **cands[i]}) for i in ch]) for j in jobs for ch in chunks
    ]
    if pool is None:
        results = [_score_task(texp, j, c, T, str(cache_dir)) for j, c in tasks]
    else:
        futs = [pool.submit(_score_task, texp, j, c, T, str(cache_dir)) for j, c in tasks]
        results = [f.result() for f in futs]
    per = {i: [] for i in idxs}
    for r in results:
        for i, s in r:
            per[i].append(s)
    return {i: float(np.mean(v)) for i, v in per.items()}


def tune(
    exp: dict,
    exp_dir: str,
    tuned_path: str,
    where: dict | None = None,
    runs: int = 3,
    min_frames: int = 250,
    max_frames: int = 4000,
    conv_db: float = -10.0,
    algs: list[str] | None = None,
    processes: int = 1,
    spaces: dict | None = None,
    force: bool = False,
    log=print,
) -> dict:
    """Tune all (or ``algs``) algorithms of ``exp``; writes/updates ``tuned_path`` after every algorithm."""
    spaces = spaces or SPACES
    L, sc, grid = exp["L"], exp["scenario"], exp["grid"]
    where = {k: (v if isinstance(v, list) else [v]) for k, v in (where or {}).items()}
    if set(where) - set(grid):
        raise ValueError(
            f"--where keys {sorted(set(where) - set(grid))} are not grid axes {sorted(grid)}"
        )

    nr = min(sc["nr_samples"], (max_frames + 6) * L)
    max_frames = min(max_frames, nr // L - 4)
    min_frames = min(min_frames, max_frames)
    texp = {
        **exp,
        "seed": exp["seed"]
        + 1_000_003,  # tuning scenarios never coincide with experiment scenarios
        "prior_seed": exp[
            "seed"
        ],  # but the prior covariance is the one of the experiment
        "runs": runs,
        "scenario": {
            **sc,
            "nr_samples": nr,
            "partitions": max(1, round(sc["partitions"] * nr / sc["nr_samples"])),
        },
    }
    points = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    points = [p for p in points if all(p[k] in v for k, v in where.items())]
    settings = dict(
        L=L,
        runs=runs,
        min_frames=min_frames,
        max_frames=max_frames,
        conv_db=conv_db,
        seed=exp["seed"],
    )
    log(
        f"tuning {exp['name']}: {len(points)} grid points, frames {min_frames}..{max_frames}"
    )

    path = Path(tuned_path)
    tuned = dict(
        experiment=exp["name"], settings=settings, git=git_info(), params={}, history={}
    )
    if path.exists():
        old = json.loads(path.read_text())
        if old["settings"] != json.loads(json.dumps(settings)) and not force:
            raise SystemExit(
                f"{path} was made with other tuning settings; use --force to redo it."
            )
        if not force:
            tuned["params"], tuned["history"] = old["params"], old["history"]

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    precompute_priors(texp, Path(exp_dir))
    cache_dir = Path(exp_dir) / "_priors"
    pool = (
        ProcessPoolExecutor(
            processes, mp_context=mp.get_context("spawn"), initializer=_init_worker
        )
        if processes > 1
        else None
    )
    try:
        for point in points:
            pk = point_key(point)
            ptexp = {**texp, "grid": {k: [v] for k, v in point.items()}}
            jobs = make_jobs(ptexp)
            tuned["params"].setdefault(pk, {})
            tuned["history"].setdefault(pk, {})
            log(f"  {pk}: {len(jobs)} scenarios")
            for name, spec in exp["algs"].items():
                if (
                    (algs and name not in algs)
                    or spec.get("tune") is False
                    or spec["kind"] not in spaces
                    or name in tuned["params"][pk]
                ):
                    continue
                space = spaces[spec["kind"]]
                cands = [dict(zip(space, v)) for v in itertools.product(*space.values())]
                alive, best_prev, T, hist = list(range(len(cands))), None, min_frames, []
                while True:
                    sc_ = _evaluate(
                        ptexp,
                        jobs,
                        name,
                        spec,
                        cands,
                        alive,
                        T,
                        pool,
                        processes,
                        cache_dir,
                    )
                    ranked = sorted(
                        (i for i in alive if math.isfinite(sc_[i])), key=lambda i: sc_[i]
                    )
                    hist.append(
                        dict(
                            frames=T, scores={json.dumps(cands[i]): sc_[i] for i in alive}
                        )
                    )
                    if not ranked:
                        log(
                            f"    {name}: all {len(alive)} candidates diverged at {T} frames; not tuned"
                        )
                        break
                    best = ranked[0]
                    log(
                        f"    {name}: {T:5d} frames, {len(alive):2d} candidates, best {cands[best]} at {sc_[best]:.1f} dB"
                    )
                    if (
                        len(ranked) == 1
                        or T >= max_frames
                        or (best == best_prev and sc_[best] <= conv_db)
                    ):
                        tuned["params"][pk][name] = cands[best]
                        break
                    alive, best_prev, T = (
                        ranked[: math.ceil(len(ranked) / 2)],
                        best,
                        min(2 * T, max_frames),
                    )
                tuned["history"][pk][name] = hist
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(tuned, indent=1))
                tmp.replace(path)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    return tuned


def apply_tuned(exp: dict, tuned_path: str) -> dict:
    """Copy of ``exp`` with the tuned parameters per grid point; ``alg_spec`` applies them per job."""
    return {**exp, "tuned": json.loads(Path(tuned_path).read_text())["params"]}
