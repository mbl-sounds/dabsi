"""
Pairwise Gram blocks, gated EMA and matrix-free C*w products (paper math, no dense RLxRL matrices).

Conventions (identical to the reference implementation in lib/admm.py)
---------------------------------------------------------------------
* A frame is s in R^{2L} (overlap-save, hop L).  The paper's X_a = W01 diag(F_2L s) W10 with the
  *unnormalised* 2L-DFT equals  F_L T_a F_L^H  with the unitary L-DFT F_L and the L x L Toeplitz
  matrix  T_a[m, n] = s_a[L + m - n].  No DFT / overlap-save matrices are ever formed: the
  Toeplitz matrix is gathered from the time-domain frame and filters are transformed with FFTs.
* Pairwise Gram block (time domain):  G_ab = T_a^H T_b   (frequency domain: F_L G_ab F_L^H).
* Cross-relation matrix C of a node with receive-slots R:
      block (a, a) = sum_{m != a} G_mm,      block (a, m) = -G_ma   (m != a).
  Hence  (C w)_a = (sum_{m != a} G_mm) w_a - sum_{m != a} G_ma w_m  and  w^H C w = sum_{pairs} ||r||^2.
* Per node n, only slot pairs (p <= q) of its receive neighbourhood are stored: G[b, n, s].
  G_qp = G_pq^H is implied.
* Filters live in the frequency domain ([.., L], complex); products are evaluated in the time domain
  with  w_t = ifft(w_f, norm="ortho"),  (C w)_f = fft((C w)_t, norm="ortho")  (F_L is unitary).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import torch

from .graph import GraphLayout


@dataclass
class GramLayout:
    """Index tensors for the stored slot pairs (p <= q) of every node."""

    M: int
    R_max: int
    S_max: int
    R_idx: torch.Tensor  # long [M, R_max] (clamped >= 0): global channel id of slot
    R_mask: torch.Tensor  # bool [M, R_max]
    R_deg: torch.Tensor  # long [M]
    pa: torch.Tensor  # long [M, S_max]  slot p
    pb: torch.Tensor  # long [M, S_max]  slot q (>= p)
    smask: torch.Tensor  # bool [M, S_max]
    is_diag: torch.Tensor  # bool [M, S_max]  (p == q) and valid
    off_mask: torch.Tensor  # bool [M, S_max]  (p < q) and valid
    diag_slot: torch.Tensor  # long [M, R_max]  slot index of (r, r) (0 for invalid slots)


def build_gram_layout(layout: GraphLayout) -> GramLayout:
    M = layout.M
    device = layout.R_idx.device
    R_deg = layout.R_deg.cpu().tolist()
    R_max = layout.R_idx.shape[1]
    pairs = [list(combinations_with_replacement(range(r), 2)) for r in R_deg]
    S_max = max(len(p) for p in pairs)

    pa = torch.zeros(M, S_max, dtype=torch.long)
    pb = torch.zeros(M, S_max, dtype=torch.long)
    smask = torch.zeros(M, S_max, dtype=torch.bool)
    diag_slot = torch.zeros(M, R_max, dtype=torch.long)
    for n, plist in enumerate(pairs):
        for s, (p, q) in enumerate(plist):
            pa[n, s], pb[n, s], smask[n, s] = p, q, True
            if p == q:
                diag_slot[n, p] = s
    is_diag = smask & (pa == pb)
    off_mask = smask & (pa != pb)
    return GramLayout(
        M=M,
        R_max=R_max,
        S_max=S_max,
        R_idx=layout.R_idx.clamp_min(0),
        R_mask=layout.R_mask,
        R_deg=layout.R_deg,
        pa=pa.to(device),
        pb=pb.to(device),
        smask=smask.to(device),
        is_diag=is_diag.to(device),
        off_mask=off_mask.to(device),
        diag_slot=diag_slot.to(device),
    )


# --------------------------------------------------------------------------------------
# Gram blocks of a single frame
# --------------------------------------------------------------------------------------
def toeplitz_frames(s: torch.Tensor, L: int) -> torch.Tensor:
    """s: [..., 2L] (time domain) -> T: [..., L, L],  T[m, n] = s[L + m - n]."""
    m = torch.arange(L, device=s.device)
    idx = L + m[:, None] - m[None, :]
    return s[..., idx]


def frame_gram(s: torch.Tensor, glay: GramLayout, L: int) -> torch.Tensor:
    """
    s: [B, M, 2L] time-domain frame of every channel (M = number of channels = nodes).
    Returns G_new: [B, M, S_max, L, L] with G_new[b, n, s] = T_p^H T_q for slot pair s of node n.
    """
    T = toeplitz_frames(s, L)  # [B, M, L, L]
    M, S = glay.M, glay.S_max
    T_loc = T[:, glay.R_idx]  # [B, M, R, L, L] node-local slots
    node = torch.arange(M, device=s.device).view(M, 1)
    Ta = T_loc[:, node, glay.pa]  # [B, M, S, L, L]
    Tb = T_loc[:, node, glay.pb]
    G = Ta.mH @ Tb
    return G * glay.smask.view(1, M, S, 1, 1)


# --------------------------------------------------------------------------------------
# Gated EMA state
# --------------------------------------------------------------------------------------
@dataclass
class GramState:
    Gx: torch.Tensor  # [B, M, S, L, L]  signal(+noise) stream, updated when VAD = 1
    Gv: torch.Tensor  # [B, M, S, L, L]  noise-only stream,    updated when VAD = 0


def init_gram_state(
    B: int, glay: GramLayout, L: int, dtype: torch.dtype, device
) -> GramState:
    """dtype: real dtype (float32/float64); Gram blocks of real signals are real in the time domain."""
    shape = (B, glay.M, glay.S_max, L, L)
    return GramState(
        Gx=torch.zeros(shape, dtype=dtype, device=device),
        Gv=torch.zeros(shape, dtype=dtype, device=device),
    )


def update_gram_state(
    state: GramState, G_new: torch.Tensor, vad: torch.Tensor, eta: float
) -> None:
    """
    Gated EMA (reference: Node._constructCrossRelationMatrix, called only for the matching VAD state):
      VAD_n = 1:  Gx <- eta Gx + (1-eta) G_new ;  Gv unchanged
      VAD_n = 0:  Gv <- eta Gv + (1-eta) G_new ;  Gx unchanged
    vad: [B, M] bool (per node).
    """
    v = vad.view(*vad.shape, 1, 1, 1)
    state.Gx = torch.where(v, eta * state.Gx + (1.0 - eta) * G_new, state.Gx)
    state.Gv = torch.where(v, state.Gv, eta * state.Gv + (1.0 - eta) * G_new)


# --------------------------------------------------------------------------------------
# Products with the implied cross-relation matrix
# --------------------------------------------------------------------------------------
def _mv(G: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """G [..., L, L] (real or complex) times w [..., L] (complex or real) -> [..., L]."""
    if w.is_complex() and not G.is_complex():
        wr = torch.view_as_real(w.contiguous())  # [..., L, 2]
        return torch.view_as_complex((G @ wr).contiguous())
    return (G @ w.unsqueeze(-1)).squeeze(-1)


def gram_matvec_time(
    G: torch.Tensor, w_t: torch.Tensor, glay: GramLayout
) -> torch.Tensor:
    """
    C w with C implied by the slot blocks G ([B,M,S,L,L]); w_t: [B,M,R,L] (time-domain filters).
    Returns [B,M,R,L]; padded slots are zero.
    """
    B, M, S, L, _ = G.shape
    R = glay.R_max
    pidx = glay.pa.view(1, M, S, 1).expand(B, M, S, L)
    qidx = glay.pb.view(1, M, S, 1).expand(B, M, S, L)
    off = glay.off_mask.view(1, M, S, 1)

    w_p = torch.gather(w_t, 2, pidx)
    w_q = torch.gather(w_t, 2, qidx)
    # off-diagonal contributions:  (C w)_q -= G_pq w_p ,  (C w)_p -= G_pq^H w_q
    y_q = -_mv(G, w_p) * off
    y_p = -_mv(G.mH, w_q) * off
    out = torch.zeros_like(w_t)
    out.scatter_add_(2, qidx, y_q)
    out.scatter_add_(2, pidx, y_p)

    # diagonal contributions: (sum_{m != a} G_mm) w_a
    dslot = glay.diag_slot.view(1, M, R, 1, 1).expand(B, M, R, L, L)
    Gd = torch.gather(G, 2, dslot) * glay.R_mask.view(1, M, R, 1, 1)
    Dsum = Gd.sum(dim=2, keepdim=True)
    out = out + _mv(Dsum - Gd, w_t)
    return out * glay.R_mask.view(1, M, R, 1)


def gram_trace(G: torch.Tensor, glay: GramLayout) -> torch.Tensor:
    """tr(C) per node = (R_n - 1) * sum_p tr(G_pp).  Returns [B, M] real."""
    d = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)  # [B, M, S]
    d = (d * glay.is_diag.view(1, *glay.is_diag.shape)).sum(-1)
    return (d.real if d.is_complex() else d) * (glay.R_deg.view(1, -1) - 1)


def freq_to_time(w_f: torch.Tensor) -> torch.Tensor:
    """w_t = F_L^H w_f (unitary)."""
    return torch.fft.ifft(w_f, dim=-1, norm="ortho")


def time_to_freq(w_t: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft(w_t, dim=-1, norm="ortho")


def gram_matvec(G: torch.Tensor, w_f: torch.Tensor, glay: GramLayout) -> torch.Tensor:
    """(C w) for frequency-domain filters w_f [B,M,R,L] -> [B,M,R,L] (frequency domain)."""
    return time_to_freq(gram_matvec_time(G, freq_to_time(w_f), glay))


def quad_form(w_f: torch.Tensor, Cw_f: torch.Tensor) -> torch.Tensor:
    """w^H C w per node, [B, M] real."""
    return (w_f.conj() * Cw_f).sum(dim=(-1, -2)).real
