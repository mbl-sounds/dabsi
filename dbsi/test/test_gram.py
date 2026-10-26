"""Tests: FFT/Toeplitz-based Gram blocks and C*w versus dense reference (old lib/admm.py construction)."""

import numpy as np
import torch

from dbsi.graph import build_graph_layout
from dbsi.gram import (
    build_gram_layout,
    frame_gram,
    init_gram_state,
    update_gram_state,
    gram_matvec,
    gram_trace,
    quad_form,
)

DT = torch.float64
CDT = torch.complex128


# ------------------------------------------------------------------ dense reference (old code)
def _dft(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    return np.exp(-2j * np.pi / N) ** (i * j) / np.sqrt(N)


def _os_matrices(L):
    F_L, F_2L = _dft(L), _dft(2 * L)
    W01 = np.concatenate([np.zeros((L, L)), np.eye(L)]).T
    W10_2LxL = np.concatenate([np.eye(L), np.zeros((L, L))])
    W01_fq = F_L @ W01 @ np.linalg.inv(F_2L)
    W10_2LxL_fq = F_2L @ W10_2LxL @ np.linalg.inv(F_L)
    W10_Lx2L = W10_2LxL.T
    W10_Lx2L_fq = F_L @ W10_Lx2L @ np.linalg.inv(F_2L)
    return W10_Lx2L_fq, W01_fq.conj().T @ W01_fq, W10_2LxL_fq


def _dense_R(block, i, j, W):
    W10a, W_R, W10b = W
    return W10a @ ((block[:, i, None].conj() * W_R).T * block[:, j, None]).T @ W10b


def _dense_cr_update(R, block, eta, N_r, L, W):
    """Old Node._constructCrossRelationMatrix."""
    R *= eta
    for i in range(N_r):
        for j in range(i, N_r):
            R_ij = _dense_R(block, i, j, W) * (1.0 - eta)
            if i != j:
                si, sj = slice(i * L, (i + 1) * L), slice(j * L, (j + 1) * L)
                R[sj, si] += -R_ij
                R[si, sj] += -R_ij.conj().T
            else:
                for n in range(N_r):
                    if n != i:
                        sn = slice(n * L, (n + 1) * L)
                        R[sn, sn] += R_ij


def _setup(L=8, M=4, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    edges = [(0, 1), (1, 2), (2, 3)]  # path -> heterogeneous receive neighbourhoods
    layout = build_graph_layout(M, edges, directed=False, include_self=True, device="cpu")
    glay = build_gram_layout(layout)
    return g, layout, glay


def test_ema_gram_matches_dense_cr_matrices():
    L, M, B, eta = 8, 4, 2, 0.9
    g, layout, glay = _setup(L, M, B)
    state = init_gram_state(B, glay, L, DT, "cpu")
    W = _os_matrices(L)
    R_sn = {
        (b, n): np.zeros((layout.R_deg[n].item() * L,) * 2, complex)
        for b in range(B)
        for n in range(M)
    }
    R_n = {k: v.copy() for k, v in R_sn.items()}

    for _ in range(6):
        s = torch.randn(B, M, 2 * L, generator=g, dtype=DT)
        vad = torch.rand(B, M, generator=g) > 0.4
        update_gram_state(state, frame_gram(s, glay, L), vad, eta)
        for b in range(B):
            blk_all = np.fft.fft(s[b].numpy().T, axis=0)  # [2L, M]
            for n in range(M):
                R_ids = layout.R_idx[n][layout.R_mask[n]].tolist()
                block = blk_all[:, R_ids]
                tgt = R_sn if vad[b, n] else R_n
                _dense_cr_update(tgt[(b, n)], block, eta, len(R_ids), L, W)

    # compare C*w and quadratic form / trace for random filters
    w = torch.randn(B, M, glay.R_max, L, generator=g, dtype=CDT) * glay.R_mask.view(
        1, M, -1, 1
    )
    for stream, ref in ((state.Gx, R_sn), (state.Gv, R_n)):
        Cw = gram_matvec(stream, w, glay)
        tr = gram_trace(stream, glay)
        for b in range(B):
            for n in range(M):
                Rn = layout.R_deg[n].item()
                wv = w[b, n, :Rn].reshape(-1).numpy()
                y = ref[(b, n)] @ wv
                np.testing.assert_allclose(
                    Cw[b, n, :Rn].reshape(-1).numpy(), y, atol=1e-10
                )
                np.testing.assert_allclose(
                    tr[b, n].item(), np.trace(ref[(b, n)]).real, atol=1e-10
                )
        q = quad_form(w, Cw)
        for b in range(B):
            for n in range(M):
                Rn = layout.R_deg[n].item()
                wv = w[b, n, :Rn].reshape(-1).numpy()
                np.testing.assert_allclose(
                    q[b, n].item(), (wv.conj() @ ref[(b, n)] @ wv).real, atol=1e-9
                )
    assert not torch.any(Cw[:, :, :, :][~layout.R_mask.view(1, M, -1, 1).expand_as(Cw)])


def test_gating_freezes_other_stream():
    L, M, B = 8, 4, 1
    g, layout, glay = _setup(L, M, B)
    state = init_gram_state(B, glay, L, DT, "cpu")
    s = torch.randn(B, M, 2 * L, generator=g, dtype=DT)
    Gn = frame_gram(s, glay, L)
    update_gram_state(state, Gn, torch.ones(B, M, dtype=torch.bool), 0.5)
    assert state.Gv.abs().max() == 0
    gx = state.Gx.clone()
    update_gram_state(state, Gn, torch.zeros(B, M, dtype=torch.bool), 0.5)
    assert torch.equal(state.Gx, gx)
    assert state.Gv.abs().max() > 0
