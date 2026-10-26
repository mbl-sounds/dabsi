"""Run an experiment over a process pool (resumable).

Layout: ``<out>/<exp name>/manifest.json`` (config + code version), ``<job_id>/<alg>.npz`` per finished
(job, algorithm), ``errors.log``. A (job, alg) is done iff its file exists, so resuming = re-running.
Worker crashes (e.g. OOM kill) only cost the running algorithms; the pool is rebuilt and the open
jobs are recomputed from disk. Ctrl-C stops cleanly; finished results stay.
Sharded runs (``shard=(i, n)``) take every n-th open job, so several nodes can share one output
directory; they skip the temp-file cleanup, which would otherwise delete files of other nodes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .experiments import job_id, make_jobs, precompute_priors, run_job

MAX_POOL_RESTARTS = 3


def default_processes() -> int:
    """CPUs this process may use (Slurm/cgroup aware; ``os.cpu_count()`` reports the whole node)."""
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity else (os.cpu_count() or 1)


def git_info() -> dict:
    """Commit and whether files below ``python/`` differ from it."""
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        ).stdout.strip()

    return dict(
        commit=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain", "--", ".")),
    )


def _init_worker():
    import torch

    torch.set_num_threads(1)


def _shard_jobs(exp: dict, shard: tuple[int, int] | None = None) -> list[dict]:
    jobs = make_jobs(exp)
    if shard is None:
        return jobs
    i, n = shard
    if not 0 <= i < n:
        raise ValueError(f"invalid shard {i}/{n}")
    # job k belongs to shard k % n; run-major order keeps every shard balanced over the grid
    return [j for k, j in enumerate(jobs) if k % n == i]


def _open_jobs(exp: dict, out: Path, shard: tuple[int, int] | None = None) -> list[dict]:
    return [
        j
        for j in _shard_jobs(exp, shard)
        if any(not (out / job_id(j) / f"{a}.npz").exists() for a in exp["algs"])
    ]


def run_experiment(
    exp: dict,
    out_root: str,
    processes: int = 1,
    dry_run: bool = False,
    force: bool = False,
    shard: tuple[int, int] | None = None,
) -> int:
    """Run all unfinished jobs (or shard ``i`` of ``n``); returns the number of failed jobs."""
    out = Path(out_root) / exp["name"]
    cfg = json.loads(json.dumps(exp))  # normalise (tuples -> lists)
    manifest, git = out / "manifest.json", git_info()
    if manifest.exists():
        old = json.loads(manifest.read_text())["config"]
        same = {**old, "runs": None} == {
            **cfg,
            "runs": None,
        }  # extending the number of runs is fine
        if not same and not force:
            raise SystemExit(
                f"{manifest} holds a different config; use another --out or --force to overwrite it."
            )
        force = force or old["runs"] != cfg["runs"]
    mine = _shard_jobs(exp, shard)
    todo = _open_jobs(exp, out, shard)
    label = f" shard {shard[0]}/{shard[1]}" if shard else ""
    print(
        f"{exp['name']}{label}: {len(mine)} jobs, {len(mine) - len(todo)} done, {len(todo)} to run -> {out}",
        flush=True,
    )
    if dry_run:
        for j in todo:
            print("  ", job_id(j))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if not manifest.exists() or force:
        # unique temp name and atomic rename: shards on different nodes may write concurrently
        tmp = manifest.with_name(f"manifest.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(dict(config=cfg, git=git), indent=2))
        tmp.replace(manifest)

    if shard is None:  # a sharded run must not delete temp files of jobs on other nodes
        for stale in out.glob("*/*.tmp"):
            stale.unlink()  # leftovers of killed writes
    if not todo:
        return 0
    precompute_priors(exp, out)

    failed, t0, n_done = 0, time.time(), 0

    def report(jid, secs, err=None):
        nonlocal failed, n_done
        n_done += 1
        if err:
            failed += 1
            with open(out / "errors.log", "a") as f:
                f.write(f"{jid}\n{err}\n")
        print(
            f"[{n_done}/{len(todo)}] {jid} {'FAILED' if err else f'{secs:6.1f}s'}  (elapsed {time.time() - t0:.0f}s)",
            flush=True,
        )
        if err:
            print(err, flush=True)

    if processes <= 1:
        for j in todo:
            try:
                report(*run_job(exp, j, str(out), git))
            except Exception:
                report(job_id(j), 0, traceback.format_exc())
        return failed

    for attempt in range(MAX_POOL_RESTARTS + 1):
        if attempt:
            todo = _open_jobs(exp, out, shard)
            print(
                f"worker pool died; restart {attempt}/{MAX_POOL_RESTARTS} with {len(todo)} open jobs",
                flush=True,
            )
            if not todo:
                break
        pool = ProcessPoolExecutor(
            processes, mp_context=mp.get_context("spawn"), initializer=_init_worker
        )
        futs = {pool.submit(run_job, exp, j, str(out), git): j for j in todo}
        try:
            for f in as_completed(futs):
                try:
                    report(*f.result())
                except BrokenProcessPool:
                    raise
                except Exception:
                    report(job_id(futs[f]), 0, traceback.format_exc())
            pool.shutdown()
            break
        except BrokenProcessPool:
            pool.shutdown(wait=False, cancel_futures=True)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print("interrupted; finished results are kept, re-run to resume", flush=True)
            return failed + 1
    else:
        print(
            "giving up after repeated pool failures (see errors.log); re-run to resume",
            flush=True,
        )
        return failed + 1
    return failed
