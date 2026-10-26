"""Experiment definitions and the per-job function.

An experiment is a plain JSON-serialisable dict:

    name, seed, runs, L, scenario (kind wgn|room, nr_samples, partitions, fs, roomsize, rt60),
    grid (scenario parameters M, N_i, SNR, signal, optionally rt60), algs (name -> spec).

One *job* = one scenario realisation (run + one grid point); all algorithms of the experiment are
evaluated on exactly that realisation. Every job is seeded from
``SeedSequence(seed, spawn_key=(run, crc32(grid point)))``, so results do not depend on grid order,
number of processes or on which jobs were already finished.

Output: one file ``<exp_dir>/<job_id>/<alg>.npz`` per (job, algorithm), written atomically (a crash
loses at most the algorithm that was running). Contents: ``mis`` [T,3] (NPM, NPMA, NPMM),
``scale`` [T,M] (relative channel scale in dB), ``sec`` (wall time), ``meta`` (JSON).

Algorithm specs: ``kind`` in dabsi | dabsi_ref | dgd | dsaawet | mcqn | spmcqn, kind parameters, and
for dabsi an optional ``prior`` (None = identity, else dict(rt60, roomsize, n_rooms) of the room
ensemble the prior covariance is estimated from; only used if ``lambd`` != 0).
"""

from __future__ import annotations

import hashlib
import itertools
from contextlib import contextmanager
import json
import time
import uuid
import zlib
from pathlib import Path

import numpy as np

LIBRISPEECH_PATH = ".librispeech"  # relative to the working directory, as for .myriad
LIBRISPEECH_DATASET = "train-clean-100"

# --------------------------------------------------------------------------- configs
_DABSI = dict(
    kind="dabsi",
    rho=1.0,
    mu=0.4,
    eta=0.98,
    lambd=0.0,
    gev_reg=1e-9,
    gev_lim=4.0,
    local_iters=1,
    da=True,
)

# wgn_rooms.py parameters
_WGN_ALGS = {
    "dabsi": _DABSI,
    "dgd": dict(kind="dgd", alpha=1e-6),
    "dsaawet": dict(kind="dsaawet", alpha=0.1),
    "mcqn": dict(kind="mcqn", mu=0.2, eta=0.98, lambd=1e-6),
    "spmcqn": dict(kind="spmcqn", mu=0.1, eta=0.98, lambd=1e-6),
}

# simulated_rooms.py: baseline step sizes are tuned per lambda
_ROOM_STEPS = {  # lambda: (mu_mcqn, mu_spmcqn, alpha_dgd, alpha_dsaawet)
    0.0: (0.15, 0.05, 1e-6, 1e-4),
    1e-5: (0.2, 0.1, 1e-5, 1e-3),
    1e-4: (0.25, 0.15, 1e-4, 1e-2),
}
_PRIOR = dict(rt60=None, roomsize=(2, 6), n_rooms=2000)  # matches the generated scenarios


def _room_algs(lambds, prior=_PRIOR) -> dict:
    algs = {}
    for lam in lambds:
        tag = f"_l{lam:g}"
        mq, sp, dg, ds = _ROOM_STEPS[lam]
        algs.update(
            {
                "dabsi" + tag: {**_DABSI, "lambd": lam, "prior": prior if lam else None},
                "dgd" + tag: dict(kind="dgd", alpha=dg),
                "dsaawet" + tag: dict(kind="dsaawet", alpha=ds),
                "mcqn" + tag: dict(kind="mcqn", mu=mq, eta=0.98, lambd=1e-6),
                "spmcqn" + tag: dict(kind="spmcqn", mu=sp, eta=0.98, lambd=1e-6),
            }
        )
    return algs


_WGN = dict(kind="wgn", nr_samples=500_000, partitions=400, fs=2000)
_ROOM = dict(
    kind="room", nr_samples=800_000, partitions=400, fs=2000, roomsize=(2, 6), rt60=None
)

EXPERIMENTS = {
    # paper: WGN-RIR rooms
    "wgn_rooms": dict(
        name="wgn_rooms",
        seed=0,
        runs=20,
        L=128,
        scenario=_WGN,
        grid=dict(M=[4], N_i=[2, 4], SNR=[50, 40, 30, 20], signal=["white", "speech"]),
        algs=_WGN_ALGS,
    ),
    # paper: simulated shoebox rooms, three prior strengths (white source; speech needs LibriSpeech)
    "simulated_rooms": dict(
        name="simulated_rooms",
        seed=0,
        runs=20,
        L=128,
        scenario=_ROOM,
        grid=dict(M=[4], N_i=[2], SNR=[50, 40, 30, 20], signal=["white", "speech"]),
        algs=_room_algs([0.0, 1e-5, 1e-4]),
    ),
    # paper: topology density ablation (proposed algorithm only, no prior)
    "topology": dict(
        name="topology",
        seed=0,
        runs=20,
        L=32,
        scenario=_WGN,
        grid=dict(M=[4, 8], N_i=[2, 4], SNR=[20], signal=["white"]),
        algs={"dabsi": _DABSI},
    ),
    # revision: prior mismatch by room size (matched / slight / distinct / none)
    "prior_mismatch": dict(
        name="prior_mismatch",
        seed=0,
        runs=20,
        L=64,
        scenario=_ROOM,
        grid=dict(M=[4], N_i=[2], SNR=[40, 20], signal=["white"]),
        algs={
            "dabsi_noprior": {**_DABSI, "lambd": 0.0},
            "dabsi_matched": {**_DABSI, "lambd": 1e-5, "prior": _PRIOR},
            "dabsi_slight": {
                **_DABSI,
                "lambd": 1e-5,
                "prior": {**_PRIOR, "roomsize": (2, 7)},
            },
            "dabsi_distinct": {
                **_DABSI,
                "lambd": 1e-5,
                "prior": {**_PRIOR, "roomsize": (8, 12)},
            },
        },
    ),
    # revision: scalability in M with bounded degree (low priority, not part of the planned runs; untested for M > 8)
    "scalability": dict(
        name="scalability",
        seed=0,
        runs=10,
        L=64,
        scenario=dict(_WGN, nr_samples=300_000, partitions=240),
        grid=dict(M=[4, 8, 16, 32], N_i=[2], SNR=[30], signal=["white"]),
        algs={k: _WGN_ALGS[k] for k in ("dabsi", "dgd", "dsaawet", "spmcqn")},
    ),
    # revision: DSAAWET step size sweep, stationary vs non-stationary white source
    "dsaawet_sweep": dict(
        name="dsaawet_sweep",
        seed=0,
        runs=20,
        L=64,
        scenario=dict(_WGN, nr_samples=300_000, partitions=240),
        grid=dict(M=[4], N_i=[2], SNR=[30], signal=["white", "stationary"]),
        algs={
            "dabsi": _DABSI,
            **{
                f"dsaawet_a{a:g}": dict(kind="dsaawet", alpha=a)
                for a in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
            },
        },
    ),
    # revision: RT60 sensitivity (shoebox rooms)
    "rt60_sensitivity": dict(
        name="rt60_sensitivity",
        seed=0,
        runs=20,
        L=64,
        scenario=_ROOM,
        grid=dict(
            M=[4], N_i=[2], SNR=[30], signal=["white"], rt60=[0.08, 0.16, 0.32, 0.64]
        ),
        algs=_room_algs([0.0]),
    ),
    # revision: measured RIRs (MYRiAD, hearing-aid mics); prior from the simulated-room ensemble
    "measured_rooms": dict(
        name="measured_rooms",
        seed=0,
        runs=20,
        L=128,
        scenario=dict(
            kind="meas",
            nr_samples=500_000,
            partitions=400,
            fs=2000,
            myriad_path=".myriad",
        ),
        grid=dict(M=[4], N_i=[2], SNR=[50, 40, 30, 20], signal=["white", "speech"]),
        algs=_room_algs([0.0, 1e-5]),
    ),
    "meas_smoke": dict(
        name="meas_smoke",
        seed=0,
        runs=1,
        L=16,
        scenario=dict(
            kind="meas", nr_samples=16 * 120, partitions=4, fs=500, myriad_path=".myriad"
        ),
        grid=dict(M=[4], N_i=[2], SNR=[20], signal=["white"]),
        algs={
            "dabsi": {
                **_DABSI,
                "lambd": 1e-5,
                "prior": dict(rt60=None, roomsize=(2, 6), n_rooms=5),
            },
            "dsaawet": dict(kind="dsaawet", alpha=1e-3),
        },
    ),
    # tiny versions for quick checks / tests
    "wgn_smoke": dict(
        name="wgn_smoke",
        seed=0,
        runs=2,
        L=16,
        scenario=dict(kind="wgn", nr_samples=16 * 120, partitions=4, fs=500),
        grid=dict(M=[3], N_i=[2], SNR=[20], signal=["white"]),
        algs={**_WGN_ALGS, "dabsi_ref": {**_DABSI, "kind": "dabsi_ref"}},
    ),
    "room_smoke": dict(
        name="room_smoke",
        seed=0,
        runs=1,
        L=16,
        scenario=dict(
            kind="room",
            nr_samples=16 * 120,
            partitions=4,
            fs=500,
            roomsize=(2, 6),
            rt60=None,
        ),
        grid=dict(M=[3], N_i=[2], SNR=[20], signal=["white", "stationary"]),
        algs={
            "dabsi": {
                **_DABSI,
                "lambd": 1e-5,
                "prior": dict(rt60=None, roomsize=(2, 6), n_rooms=5),
            },
            "dsaawet": dict(kind="dsaawet", alpha=1e-3),
        },
    ),
}


# --------------------------------------------------------------------------- jobs
def make_jobs(exp: dict) -> list[dict]:
    """Run-major job list (partial results stay balanced across settings)."""
    g = exp["grid"]
    pts = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    return [dict(run=r, **p) for r in range(exp["runs"]) for p in pts]


def job_id(j: dict) -> str:
    parts = [
        f"{k}{v:g}" if isinstance(v, (int, float)) else str(v)
        for k, v in j.items()
        if k != "run"
    ]
    return "_".join(parts) + f"_run{j['run']:03d}"


def _seed_key(j: dict) -> tuple:
    point = json.dumps({k: v for k, v in sorted(j.items()) if k != "run"})
    return (j["run"], zlib.crc32(point.encode()))


@contextmanager
def _global_rng(seq: np.random.SeedSequence):
    """pyroomacoustics draws its image-source jitter from the global NumPy RNG: seed it for the block, then restore."""
    state = np.random.get_state()
    np.random.seed(seq.generate_state(1)[0])
    try:
        yield
    finally:
        np.random.set_state(state)


def topology_stats(M: int, edges) -> dict:
    """Summary of the directed topology (self loops excluded) for later analysis."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    G.add_edges_from((int(i), int(j)) for i, j in edges if i != j)
    U = G.to_undirected()
    connected = nx.is_connected(U)
    in_deg = np.array([d for _, d in G.in_degree()])
    return dict(
        n_nodes=M,
        n_edges=G.number_of_edges(),
        density=nx.density(G),
        mean_in_degree=float(in_deg.mean()),
        max_in_degree=int(in_deg.max()),
        connected=bool(connected),
        diameter=nx.diameter(U) if connected else None,
        avg_path_length=nx.average_shortest_path_length(U) if connected else None,
    )


# --------------------------------------------------------------------------- scenario and prior
def _speech_source(sc: dict, rng: np.random.Generator) -> np.ndarray:
    """LibriSpeech source [N,1] as in ``simulations/*_rooms.py`` (VAD -30 dB, random pauses)."""
    from lib import simulation_utils as su

    lsr = su.librispeechSampleRandomizer(
        path=LIBRISPEECH_PATH, dataset=LIBRISPEECH_DATASET, rng=rng
    )
    src = su.getSpeechSample(
        N=sc["nr_samples"],
        fs=sc["fs"],
        paths=next(lsr),
        vad_threshold=-30,
        padding=(1000, 1000),
        partitions=sc["partitions"],
        part_rand=True,
        rng=rng,
    )
    if not np.any(src):  # the 2 x 1000 sample padding leaves no speech in short sources
        raise ValueError("speech source is all zero; increase nr_samples")
    return src


def make_scenario(exp: dict, job: dict):
    from .harness import (
        generate_meas_scenario,
        generate_room_scenario,
        generate_wgn_scenario,
    )

    sc, L = exp["scenario"], exp["L"]
    rng = np.random.default_rng(
        np.random.SeedSequence(exp["seed"], spawn_key=_seed_key(job) + (0,))
    )
    sig = job.get("signal", "white")
    if sig not in ("white", "stationary", "speech"):
        raise ValueError(f"unknown signal {sig!r}")
    common = dict(L=L, nr_samples=sc["nr_samples"], partitions=sc["partitions"])
    if sig == "stationary":
        from lib import simulation_utils as su

        common["partitions"] = 1
        common["src_signal"] = su.getWhitenoiseSample(
            N=sc["nr_samples"],
            partitions=1,
            part_rand=False,
            amp_rand=(1.0, 1.0),
            rng=rng,
        )
    if sig == "speech":
        common["src_signal"] = _speech_source(sc, rng)
    if sc["kind"] == "wgn":
        return generate_wgn_scenario(rng, job["M"], job["N_i"], job["SNR"], **common)
    if sc["kind"] == "meas":
        return generate_meas_scenario(
            rng,
            job["M"],
            job["N_i"],
            job["SNR"],
            fs=sc["fs"],
            myriad_path=sc["myriad_path"],
            **common,
        )
    return generate_room_scenario(
        rng,
        job["M"],
        job["N_i"],
        job["SNR"],
        fs=sc["fs"],
        roomsize=sc["roomsize"],
        rt60=job.get("rt60", sc.get("rt60")),
        **common,
    )


def compute_prior(exp: dict, prior: dict | None, cache_dir: Path) -> np.ndarray:
    """Room-ensemble covariance C [L,L] (as in simulation.py), cached on disk. None -> identity."""
    L = exp["L"]
    if prior is None:
        return np.eye(L)
    fs = exp["scenario"]["fs"]
    seed = exp.get(
        "prior_seed", exp["seed"]
    )  # tuning scenarios use the prior of the experiment
    key = hashlib.sha1(
        json.dumps([prior, L, fs, seed], sort_keys=True).encode()
    ).hexdigest()[:12]
    path = Path(cache_dir) / f"prior_{key}.npy"
    if path.exists():
        return np.load(path)
    from lib import simulation_utils as su

    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(10**6, int(key, 16) % 2**32))
    )
    n_mics, C = 10, np.zeros((L, L), dtype=complex)
    with _global_rng(
        np.random.SeedSequence(seed, spawn_key=(10**6, int(key, 16) % 2**32, 1))
    ):
        for _ in range(prior["n_rooms"]):
            room, _ = su.generateRandomRoomAndGraph(
                tuple(prior["roomsize"]),
                L,
                n_mics,
                1,
                fs,
                margin=0.1,
                rng=rng,
                rt60=prior["rt60"],
            )
            rirs_f = np.fft.fft(su.removeInitialDelayFromRIRs(room.rir, L, False), axis=0)
            C += rirs_f @ rirs_f.conj().T / (prior["n_rooms"] * n_mics)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.stem}.{uuid.uuid4().hex}.tmp.npy"
    )  # unique: shards may race
    np.save(tmp, C)
    tmp.replace(path)
    return C


def precompute_priors(exp: dict, out_dir: Path) -> None:
    """Fetch the measured dataset if needed and build all prior covariances once (before workers start)."""
    if exp["scenario"]["kind"] == "meas":
        from lib import simulation_utils as su

        su.downloadMyriad(path=exp["scenario"]["myriad_path"])
    for spec in exp["algs"].values():
        if spec.get("lambd") and spec.get("prior"):
            compute_prior(exp, spec["prior"], Path(out_dir) / "_priors")


# --------------------------------------------------------------------------- algorithms
def _lib():
    """Cython backends; raises ImportError if the extensions are not built."""
    import lib.centralized_cy as cent
    import lib.distgrad_cy as dgd
    import lib.dsaawet_cy as dsaawet
    import lib.simulation_utils_cy as su

    return su, dgd, dsaawet, cent


def _run_baseline(spec: dict, scn, rng: np.random.Generator, T: int):
    import torch

    from .harness import channel_scale_db, misalignment

    su, dgd, dsaawet, cent = _lib()
    L, M, kind = scn.L, scn.M, spec["kind"]
    g = su.Graph(M, scn.edges)
    if kind == "dgd":
        fs, alg = 4 * L, dgd.DGD(g, L, 4 * L, spec["alpha"])
    elif kind == "dsaawet":
        h0 = rng.uniform(-0.2, 0.2, (L * M, 1))
        h0 = h0 / np.linalg.norm(h0) / M
        fs, alg = 2 * L, dsaawet.DSAAWET(g, L, h0, 2 * L, spec["alpha"])
    elif kind == "mcqn":
        fs, alg = L, cent.MCQN(L, M, spec["mu"], spec["eta"], spec["lambd"])
    elif kind == "spmcqn":
        fs, alg = L, cent.SPMCQN(g, L, M, spec["mu"], spec["eta"], spec["lambd"])
    else:
        raise ValueError(f"unknown algorithm kind {kind!r}")

    est = []
    for k in range(0, T * L, L):
        alg.step(scn.noisy[k : k + fs, :], scn.vad[k + fs // 2, :])
        est.append(np.asarray(alg.getEstimate()).reshape(M, L))
    est, h_true = torch.as_tensor(np.stack(est)), torch.as_tensor(scn.h_true)
    return misalignment(est, h_true).numpy(), channel_scale_db(est, h_true).numpy()


def _run_alg(name: str, spec: dict, exp: dict, scn, key: tuple, cache_dir: Path, T: int):
    from .config import DABSIConfig
    from .harness import run_dabsi, run_reference

    kind, params = spec["kind"], {
        k: v for k, v in spec.items() if k not in ("kind", "prior")
    }
    if kind in ("dabsi", "dabsi_ref"):
        C = compute_prior(exp, spec.get("prior") if params["lambd"] else None, cache_dir)
        if kind == "dabsi":
            cfg = DABSIConfig(L=scn.L, **params)
            mis, scale = run_dabsi(
                cfg, [scn], sigma_inv=np.linalg.inv(C), max_frames=T, with_scale=True
            )
            return mis[0], scale[0]
        ref = {k: params[k] for k in ("rho", "mu", "eta", "lambd", "gev_reg", "gev_lim")}
        mis = run_reference(
            scn, ref, C, da=params["da"], local_iters=params["local_iters"], max_frames=T
        )
        return mis, np.zeros((len(mis), scn.M))
    rng = np.random.default_rng(
        np.random.SeedSequence(
            exp["seed"], spawn_key=key + (1, zlib.crc32(name.encode()))
        )
    )
    return _run_baseline(spec, scn, rng, T)


# --------------------------------------------------------------------------- job
def _peak_rss_mb() -> float:
    """Peak resident memory of this process (macOS reports bytes, Linux kilobytes)."""
    import resource
    import sys

    scale = 1 / 2**20 if sys.platform == "darwin" else 1 / 2**10
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def run_job(exp: dict, job: dict, out_dir: str, git: dict) -> tuple[str, float]:
    """Run all missing algorithms of one job; write ``<out_dir>/<job_id>/<alg>.npz`` atomically."""
    t0 = time.time()
    jdir = Path(out_dir) / job_id(job)
    todo = [n for n in exp["algs"] if not (jdir / f"{n}.npz").exists()]
    if not todo:
        return job_id(job), 0.0
    jdir.mkdir(parents=True, exist_ok=True)
    key = _seed_key(job)
    with _global_rng(np.random.SeedSequence(exp["seed"], spawn_key=key + (2,))):
        scn = make_scenario(exp, job)
    T = len(
        range(0, scn.noisy.shape[0] - 4 * exp["L"], exp["L"])
    )  # common length (DGD uses 4L frames)
    degree_in = np.bincount(
        [dst for _, dst in scn.edges], minlength=scn.M
    )  # incl. self loop
    topo = topology_stats(scn.M, scn.edges)

    for name in todo:
        t1 = time.time()
        mis, scale = _run_alg(
            name, exp["algs"][name], exp, scn, key, Path(out_dir) / "_priors", T
        )
        sec = time.time() - t1
        meta = dict(
            job=job,
            alg=name,
            spec=exp["algs"][name],
            seed=exp["seed"],
            git=git,
            mic_snr_db=scn.mic_snr_db.tolist(),
            edges=scn.edges,
            degree_in=degree_in.tolist(),
            topology=topo,
            leading_zeros=scn.leading_zeros,
            vad_fraction=scn.vad.mean(0).tolist(),
            sec_per_frame=sec / T,
            peak_rss_mb=_peak_rss_mb(),
        )
        path = jdir / f"{name}.npz"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                mis=mis.astype(np.float32),
                scale=scale.astype(np.float32),
                sec=sec,
                meta=json.dumps(meta),
            )
        tmp.replace(path)
    return job_id(job), time.time() - t0
