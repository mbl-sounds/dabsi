"""Simulation harness: WGN-room scenarios, misalignment metrics, batched DABSI runs, reference runs.

Scenario generation reuses the legacy generators in ``lib.simulation_utils`` so that old and new
implementations can be fed with bit-identical data. No coding / transmission loss is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DABSIConfig
from .dabsi import DABSI
from .graph import build_graph_layout


# --------------------------------------------------------------------------- scenario
@dataclass
class Scenario:
    noisy: np.ndarray  # [N, M] real, noisy microphone signals
    vad: np.ndarray  # [N, M] bool, per-node (clean-signal based) VAD
    rirs_f: np.ndarray  # [L, M] complex, DFT of the true impulse responses
    leading_zeros: list  # [M] int, delays
    edges: list  # directed (src, dst) edges incl. self loops, as in lib Graph
    mic_snr_db: np.ndarray  # [M]
    L: int

    @property
    def M(self) -> int:
        return self.noisy.shape[1]

    @property
    def h_true(self) -> np.ndarray:
        """[M, L] channel-major, frequency domain (same layout as rirs_f.T)."""
        return self.rirs_f.T.copy()

    def lib_graph(self):
        from lib import simulation_utils as su

        return su.Graph(self.M, self.edges)


def generate_wgn_scenario(
    rng: np.random.Generator,
    M: int,
    N_i: int,
    SNR: float,
    L: int = 64,
    nr_samples: int = 500_000,
    partitions: int = 400,
    src_signal: np.ndarray | None = None,
) -> Scenario:
    """Random WGN-RIR scenario exactly as in ``simulations/wgn_rooms.py`` (single part, no dynamics).

    src_signal: optional source signal of length ``nr_samples`` (e.g. speech); default is the
    partitioned white-noise source used in the paper simulations.
    """
    from lib import simulation_utils as su

    if src_signal is None:
        src_signal = su.getWhitenoiseSample(
            N=nr_samples,
            partitions=partitions,
            part_rand=True,
            amp_rand=(0.5, 1.0),
            rng=rng,
        )
    src_signal = src_signal / np.var(src_signal) ** 0.5 * 0.1

    rirs_f, leading_zeros, mic, graph = su.signalToWGN(
        src_signal, nr_samples, [{}], L, M, N_i, rng
    )
    vad = su.getVADFromCleanSignal(mic, threshold=-60, smoothing_length=10)
    noise = su.getNoiseFromSNR(mic, vad, SNR, M, L, alpha=1.0, ir_type="room", rng=rng)
    mic_snr_db = 10 * np.log10(su.getMicSNRs(mic, vad, noise))
    return Scenario(
        noisy=mic + noise,
        vad=np.asarray(vad, dtype=bool),
        rirs_f=rirs_f[0],
        leading_zeros=[int(z) for z in leading_zeros[0]],
        edges=[tuple(map(int, e)) for e in graph.edges],
        mic_snr_db=mic_snr_db,
        L=L,
    )


def generate_room_scenario(
    rng: np.random.Generator,
    M: int,
    N_i: int,
    SNR: float,
    L: int = 64,
    nr_samples: int = 800_000,
    partitions: int = 400,
    fs: int = 2000,
    roomsize: tuple = (2, 6),
    rt60: float | None = None,
    src_signal: np.ndarray | None = None,
) -> Scenario:
    """Random shoebox-room scenario as in ``simulations/simulated_rooms.py`` (white source, single part)."""
    from lib import simulation_utils as su

    if src_signal is None:
        src_signal = su.getWhitenoiseSample(
            N=nr_samples,
            partitions=partitions,
            part_rand=True,
            amp_rand=(0.5, 1.0),
            rng=rng,
        )
    rirs_f, leading_zeros, mic, graph = su.signalToRoom(
        src_signal, nr_samples, [{}], L, M, N_i, fs, tuple(roomsize), rng, rt60=rt60
    )
    vad = su.getVADFromCleanSignal(mic, threshold=-60, smoothing_length=10)
    noise = su.getNoiseFromSNR(mic, vad, SNR, M, L, alpha=1.0, ir_type="room", rng=rng)
    mic_snr_db = 10 * np.log10(su.getMicSNRs(mic, vad, noise))
    return Scenario(
        noisy=mic + noise,
        vad=np.asarray(vad, dtype=bool),
        rirs_f=rirs_f[0],
        leading_zeros=[int(z) for z in leading_zeros[0]],
        edges=[tuple(map(int, e)) for e in graph.edges],
        mic_snr_db=mic_snr_db,
        L=L,
    )


def generate_meas_scenario(
    rng: np.random.Generator,
    M: int,
    N_i: int,
    SNR: float,
    L: int = 64,
    nr_samples: int = 800_000,
    partitions: int = 400,
    fs: int = 2000,
    myriad_path: str = ".myriad",
    src_signal: np.ndarray | None = None,
) -> Scenario:
    """Measured-RIR scenario from the local MYRiAD archive (white source, single part, M-4 external mics)."""
    from lib import simulation_utils as su

    if src_signal is None:
        src_signal = su.getWhitenoiseSample(
            N=nr_samples,
            partitions=partitions,
            part_rand=True,
            amp_rand=(0.5, 1.0),
            rng=rng,
        )
    src_signal = src_signal / np.var(src_signal) ** 0.5 * 0.1
    rirs_f, leading_zeros, mic, graph = su.signalToMeas(
        src_signal, nr_samples, [{}], L, N_i, fs, M - 4, myriad_path, rng
    )
    vad = su.getVADFromCleanSignal(mic, threshold=-60, smoothing_length=10)
    noise = su.getNoiseFromSNR(mic, vad, SNR, M, L, alpha=1.0, ir_type="room", rng=rng)
    mic_snr_db = 10 * np.log10(su.getMicSNRs(mic, vad, noise))
    return Scenario(
        noisy=mic + noise,
        vad=np.asarray(vad, dtype=bool),
        rirs_f=rirs_f[0],
        leading_zeros=[int(z) for z in leading_zeros[0]],
        edges=[tuple(map(int, e)) for e in graph.edges],
        mic_snr_db=mic_snr_db,
        L=L,
    )


def frame_starts(N: int, L: int) -> range:
    """Frame start samples of the reference loop: ``range(0, N - 2L, L)`` (frame 2L, hop L)."""
    return range(0, N - 2 * L, L)


# --------------------------------------------------------------------------- metrics
def _npm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalised projection misalignment over the last dim; a = estimate, b = truth."""
    proj = (b.conj() * a).sum(-1, keepdim=True) / (a.conj() * a).sum(-1, keepdim=True)
    e = b - proj * a
    return torch.linalg.vector_norm(e, dim=-1) / torch.linalg.vector_norm(b, dim=-1)


def misalignment(h_est: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """
    h_est, h_true: [..., M, L] complex (frequency domain, channel-major).
    Returns [..., 3] = (NPM global, NPMA mean of per-channel NPM, NPMM min of per-channel NPM),
    identical to ``lib.simulation_utils.getMisalignment(..., ["NPM", "NPMA", "NPMM"], L)``.
    """
    flat = (h_est.flatten(-2), h_true.flatten(-2))
    npm = _npm(*flat)
    per = _npm(h_est, h_true)  # [..., M]
    return torch.stack([npm, per.mean(-1), per.amin(-1)], dim=-1)


def channel_scale_db(h_est: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """
    [..., M] per-channel LS scale of the estimate towards the truth (h_true ~ c_i h_est_i) in dB,
    relative to the mean over channels. Zero everywhere = no relative scale drift between channels.
    """
    c = (h_est.conj() * h_true).sum(-1) / (h_est.conj() * h_est).sum(-1)
    db = 20 * torch.log10(c.abs())
    return db - db.mean(-1, keepdim=True)


# --------------------------------------------------------------------------- new implementation
def _group_by_topology(scenarios: list[Scenario]) -> list[list[int]]:
    groups: dict[frozenset, list[int]] = {}
    for i, s in enumerate(scenarios):
        groups.setdefault(frozenset(s.edges), []).append(i)
    return list(groups.values())


def run_dabsi(
    cfg: DABSIConfig,
    scenarios: list[Scenario],
    sigma_inv: np.ndarray | torch.Tensor | None = None,
    align_self: bool = False,
    max_frames: int | None = None,
    with_scale: bool = False,
):
    """
    Run the batched torch DABSI on scenarios (scenarios with identical topology share a batch).
    Returns misalignment [S, T, 3] (NPM, NPMA, NPMM per frame) as float64 numpy; with ``with_scale``
    also the relative channel scales [S, T, M] in dB.
    """
    L = cfg.L
    S = len(scenarios)
    N = min(s.noisy.shape[0] for s in scenarios)
    T = len(frame_starts(N, L))
    if max_frames is not None:
        T = min(T, max_frames)
    out = np.zeros((S, T, 3))
    scale = np.zeros((S, T, scenarios[0].M))

    for idx in _group_by_topology(scenarios):
        sc = [scenarios[i] for i in idx]
        M = sc[0].M
        layout = build_graph_layout(
            M, sc[0].edges, directed=True, include_self=True, device="cpu"
        )
        x = torch.as_tensor(
            np.stack([s.noisy[:N].T for s in sc]), dtype=cfg.real_dtype, device=cfg.device
        )
        vad = torch.as_tensor(
            np.stack([s.vad[:N] for s in sc]), device=cfg.device
        )  # [B,N,M]
        h_true = torch.as_tensor(
            np.stack([s.h_true for s in sc]), dtype=cfg.complex_dtype, device=cfg.device
        )

        model = DABSI(cfg, layout, len(sc), sigma_inv=sigma_inv, align_self=align_self)
        model.set_delays(torch.as_tensor([s.leading_zeros for s in sc]))
        for t in range(T):
            k = t * L
            model.step(x[:, :, k : k + 2 * L], vad[:, k + L])
            est = model.estimate()
            out[idx, t] = misalignment(est, h_true).double().cpu().numpy()
            if with_scale:
                scale[idx, t] = channel_scale_db(est, h_true).double().cpu().numpy()
    return (out, scale) if with_scale else out


# --------------------------------------------------------------------------- reference implementation
def run_reference(
    scn: Scenario,
    params: dict,
    sigma: np.ndarray,
    da: bool = True,
    local_iters: int = 1,
    max_frames: int | None = None,
) -> np.ndarray:
    """Legacy numpy ``lib.admm.Network`` run on one scenario (as in ``wgn_rooms.py``). Returns [T, 3]."""
    from lib import admm
    from lib import simulation_utils as su

    L = scn.L
    net = admm.Network(scn.lib_graph(), L)
    net.setPriorInformation(sigma)
    net.reset()
    net.setParameters({**params, "localStep": admm._BFGSGEVP})
    net.local_iters = local_iters
    net.da = da
    net.setDelays(scn.leading_zeros)

    h_true = scn.rirs_f.T.reshape(-1, 1)
    out = []
    for n, k in enumerate(frame_starts(scn.noisy.shape[0], L)):
        if max_frames is not None and n >= max_frames:
            break
        net.step(scn.noisy[k : k + 2 * L, :], scn.vad[k + L, :])
        out.append(
            su.getMisalignment(net.getEstimate(), h_true, ["NPM", "NPMA", "NPMM"], L)
        )
    return np.asarray(out, dtype=float)
