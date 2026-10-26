"""Frame-by-frame parity of the batched torch DABSI against the reference lib.admm.Network (no coding)."""

import numpy as np
import pytest
import torch

from dbsi.config import DABSIConfig
from dbsi.dabsi import DABSI
from dbsi.graph import build_graph_layout

admm = pytest.importorskip("lib.admm")
sim_utils = pytest.importorskip("lib.simulation_utils")


def _reference(M, L, edges, params, sigma, delays, frames, vads, da):
    graph = sim_utils.Graph(M, edges)
    net = admm.Network(graph, L)
    if sigma is not None:
        net.setPriorInformation(sigma)
    net.reset()
    net.setParameters({**params, "localStep": admm._BFGSGEVP})
    net.local_iters = 1
    net.da = da
    net.setDelays(list(delays))
    out = []
    for s, v in zip(frames, vads):
        net.step(s, v)
        out.append(net.getEstimate().reshape(M, L).copy())
    return np.stack(out)


@pytest.mark.parametrize("da", [True, False])
@pytest.mark.parametrize("lambd", [0.0, 1e-3])
def test_parity_with_reference(da, lambd):
    rng = np.random.default_rng(0)
    M, L, T, B = 4, 8, 30, 2
    und = [(0, 1), (1, 2), (2, 3), (0, 2)]
    edges = sorted(
        {(a, b) for a, b in und} | {(b, a) for a, b in und} | {(i, i) for i in range(M)}
    )
    layout = build_graph_layout(M, und, directed=False, include_self=True, device="cpu")
    assert sorted(layout.edges) == edges

    params = dict(rho=1.0, mu=0.4, eta=0.9, lambd=lambd, gev_reg=1e-9, gev_lim=4.0)
    cfg = DABSIConfig(L=L, eta=params["eta"], rho=1.0, mu=0.4, lambd=lambd, da=da)

    A_ = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
    sigma = A_ @ A_.conj().T / L + np.eye(L)
    sigma_inv = np.linalg.inv(sigma)

    # SIMO signal (white source through random FIRs + sensor noise), hop L, frames of 2L
    N = (T + 1) * L
    src = rng.normal(size=N + L - 1)
    h = rng.normal(size=(M, L)) * np.exp(-np.arange(L) / (L / 3))
    x = np.stack([np.convolve(src, h[m])[:N] for m in range(M)], axis=1)  # [N, M]
    x = x[None].repeat(B, 0) + 1e-2 * rng.normal(size=(B, N, M))
    frames = np.stack([[x[b, t * L : t * L + 2 * L] for t in range(T)] for b in range(B)])
    vads = rng.random((B, T, M)) > 0.35
    vads[:, :3] = True
    delays = rng.integers(0, 3, size=(B, M))

    model = DABSI(cfg, layout, B, sigma_inv=torch.as_tensor(sigma_inv))
    model.set_delays(torch.as_tensor(delays))

    ref = [
        _reference(M, L, edges, params, sigma, delays[b], frames[b], vads[b], da)
        for b in range(B)
    ]
    ref = np.stack(ref, axis=1)  # [T,B,M,L]

    for t in range(T):
        s = torch.as_tensor(frames[:, t].transpose(0, 2, 1).copy())  # [B,M,2L]
        model.step(s, torch.as_tensor(vads[:, t]))
        est = model.estimate().numpy()
        np.testing.assert_allclose(
            est, ref[t], rtol=1e-6, atol=1e-6, err_msg=f"frame {t}"
        )
