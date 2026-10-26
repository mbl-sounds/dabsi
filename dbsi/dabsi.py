"""
Batched DABSI (distributed adaptive blind SIMO identification) on top of the pairwise Gram state.

Mirrors the data flow of the reference `lib.admm.Network.step` (BFGS GEV local step, ADMM consensus,
adaptive distributed averaging, per-node delay alignment, per-node VAD gating), but

* batched over independent simulations (leading dim B) and over nodes (dim M),
* no transmission loss / quantisation / coding (all exchanges are ideal),
* covariance information is only stored as EMA of pairwise Gram blocks (see gram.py), the big
  (R L) x (R L) cross-relation matrices are never formed.

Tensor layout per node n: slot dimension R (receive neighbourhood, padded to R_max, includes self),
filters in the frequency domain, shape [B, M, R, L], complex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from . import gram as G
from .config import DABSIConfig
from .data import make_frames
from .graph import GraphLayout


@dataclass
class DABSIState:
    x: torch.Tensor  # [B,M,R,L] local estimates (primal)
    u: torch.Tensor  # [B,M,R,L] dual
    z: torch.Tensor  # [B,M,R,L] consensus copies (own slot = own consensus estimate)
    y: torch.Tensor  # [B,M,R,L] x + u/rho (after local update)
    y_r: torch.Tensor  # [B,M,T,L] received local estimates of own channel (incl. self)
    H: torch.Tensor  # [B,M,RL,RL] BFGS inverse Hessian
    first_iter: torch.Tensor  # [B,M] bool
    n_inst: torch.Tensor  # [B,M,R] ||z_slot||^2 (previous frame)
    phi_da: torch.Tensor  # [B,M,R] neighbour norm estimates
    phi_own: torch.Tensor  # [B,M]   own norm estimate
    Bph: torch.Tensor  # [B,M,L] delay alignment phases (same for all slots of a node)
    gram: G.GramState


class DABSI:
    def __init__(
        self,
        cfg: DABSIConfig,
        layout: GraphLayout,
        batch: int,
        sigma_inv: torch.Tensor | None = None,
        align_self: bool = False,
    ):
        """
        sigma_inv: [L, L] inverse prior covariance (frequency domain) or None.
        align_self: apply conj(B) also on the node's own entry of the consensus average.
            The reference implementation does NOT (self entry is y_own without alignment).
        """
        self.cfg = cfg
        self.L = cfg.L
        self.B = batch
        self.M = layout.M
        self.align_self = align_self
        dev = torch.device(cfg.device)
        self.device = dev
        self.rdt, self.cdt = cfg.real_dtype, cfg.complex_dtype

        self.glay = _to_device(G.build_gram_layout(layout), dev)
        M = self.M
        self.R_max = layout.R_idx.shape[1]
        self.T_max = layout.T_idx.shape[1]
        self.node = torch.arange(M, device=dev)
        self.R_idx = layout.R_idx.clamp_min(0).to(dev)  # [M,R]
        self.R_mask = layout.R_mask.to(dev)
        self.R_deg = layout.R_deg.to(dev)
        self.self_slot = layout.self_slot_in_R.to(dev)
        self.T_idx = layout.T_idx.clamp_min(0).to(dev)  # [M,T]
        self.T_mask = layout.T_mask.to(dev)
        self.T_deg = layout.T_deg.to(dev)
        self.T_slot = layout.T_slot_in_R.clamp_min(0).to(dev)
        self.is_self_R = self.R_mask & (self.R_idx == self.node.view(M, 1))
        self.not_self_R = self.R_mask & ~self.is_self_R
        self.is_self_T = self.T_mask & (self.T_idx == self.node.view(M, 1))
        self.not_self_T = self.T_mask & ~self.is_self_T

        self.sigma_inv = None
        self.tr_sigma_inv = None
        if sigma_inv is not None and cfg.lambd != 0.0:
            self.sigma_inv = torch.as_tensor(sigma_inv).to(device=dev, dtype=self.cdt)
            self.tr_sigma_inv = torch.trace(self.sigma_inv).real
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        B, M, R, T, L = self.B, self.M, self.R_max, self.T_max, self.L
        dev, cdt, rdt = self.device, self.cdt, self.rdt
        rm = self.R_mask.view(1, M, R, 1)
        x = (torch.ones(B, M, R, L, dtype=cdt, device=dev) / L) * rm
        z = torch.zeros(B, M, R, L, dtype=cdt, device=dev)
        z[:, self.node, self.self_slot] = 1.0 / math.sqrt(L)
        n = R * L
        H = torch.eye(n, dtype=cdt, device=dev).expand(B, M, n, n).clone()
        self.st = DABSIState(
            x=x,
            u=torch.zeros_like(x),
            z=z,
            y=torch.zeros_like(x),
            y_r=torch.zeros(B, M, T, L, dtype=cdt, device=dev),
            H=H,
            first_iter=torch.ones(B, M, dtype=torch.bool, device=dev),
            n_inst=torch.zeros(B, M, R, dtype=rdt, device=dev),
            phi_da=torch.full((B, M, R), 1.0 / M, dtype=rdt, device=dev),
            phi_own=torch.full((B, M), 1.0 / M, dtype=rdt, device=dev),
            Bph=torch.ones(B, M, L, dtype=cdt, device=dev),
            gram=G.init_gram_state(B, self.glay, L, rdt, dev),
        )

    def set_delays(self, delays: torch.Tensor):
        """delays: integer [M] or [B,M]."""
        d = torch.as_tensor(delays, device=self.device).to(self.rdt)
        if d.dim() == 1:
            d = d.expand(self.B, self.M)
        k = torch.arange(self.L, device=self.device, dtype=self.rdt)
        ang = -2.0 * math.pi * d.unsqueeze(-1) * k / self.L
        self.st.Bph = torch.polar(torch.ones_like(ang), ang).to(self.cdt)

    def estimate(self) -> torch.Tensor:
        """Local estimate of every node's own channel, frequency domain, [B,M,L]."""
        return self.st.x[:, self.node, self.self_slot]

    # ------------------------------------------------------------------ helpers
    def _prior(self, w: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("kl,bmrl->bmrk", self.sigma_inv, w)
        return lam.view(*lam.shape, 1, 1) * out

    def _mask_slots(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.R_mask.view(1, self.M, self.R_max, 1)

    # ------------------------------------------------------------------ one frame
    def step(self, s: torch.Tensor, vad: torch.Tensor):
        """
        s:   [B, M, 2L] real time-domain frame (one 2L block per channel/node)
        vad: [B, M] bool, per-node VAD decision for this frame
        """
        cfg, st = self.cfg, self.st
        s = s.to(device=self.device, dtype=self.rdt)
        vad = vad.to(device=self.device, dtype=torch.bool)

        G.update_gram_state(st.gram, G.frame_gram(s, self.glay, self.L), vad, cfg.eta)

        for _ in range(cfg.local_iters):
            self._local_update(vad)
        self._transmit_local(vad)
        self._update_consensus(vad)
        self._normalize_consensus(vad)
        if cfg.da:
            st.phi_da = st.phi_own[:, self.R_idx] * self.R_mask.view(
                1, self.M, self.R_max
            )
        self._transmit_consensus(vad)
        self._update_dual(vad)

    # ------------------------------------------------------------------ local BFGS-GEV step
    def _local_update(self, vad: torch.Tensor):
        cfg, st, glay = self.cfg, self.st, self.glay
        B, M, R, L = self.B, self.M, self.R_max, self.L
        Gx, Gv = st.gram.Gx, st.gram.Gv
        x = st.x
        v4 = vad.view(B, M, 1, 1)

        Cx_x = G.gram_matvec(Gx, x, glay)
        Cv_x = G.gram_matvec(Gv, x, glay)
        a = G.quad_form(x, Cx_x)
        b = G.quad_form(x, Cv_x)
        sig = ((a + cfg.gev_reg) / (b + cfg.gev_reg)).clamp(0.0, cfg.gev_lim)  # [B,M]
        sig4 = sig.view(B, M, 1, 1)

        lam = None
        grad = Cx_x - sig4 * Cv_x
        if self.sigma_inv is not None:
            trCR = G.gram_trace(Gx, glay) - sig * G.gram_trace(Gv, glay)
            lam = (
                cfg.lambd
                * trCR
                / (self.R_deg.view(1, M).to(self.rdt) ** 2 * self.tr_sigma_inv)
            )
            grad = grad + self._prior(x, lam)
        Bz = st.Bph.unsqueeze(2) * st.z
        grad = self._mask_slots(grad + st.u + cfg.rho * (x - Bz))

        n = R * L
        g = grad.reshape(B, M, n)
        Hg = (st.H @ g.unsqueeze(-1)).squeeze(-1)
        direction = -torch.where(st.first_iter.unsqueeze(-1), g, Hg)
        s_step = (cfg.mu * direction).view(B, M, R, L)

        # gradient difference with frozen sigma / prior:  y_diff = (C_R + rho I) s
        yd = G.gram_matvec(Gx, s_step, glay) - sig4 * G.gram_matvec(Gv, s_step, glay)
        if lam is not None:
            yd = yd + self._prior(s_step, lam)
        yd = self._mask_slots(yd + cfg.rho * s_step)

        sv = s_step.reshape(B, M, n)
        yv = yd.reshape(B, M, n)
        gamma = (sv.conj() * yv).sum(-1).real  # [B,M]
        ok = (gamma > cfg.bfgs_tol) & vad
        gam = torch.where(ok, gamma, torch.ones_like(gamma))
        a_ = (st.H @ yv.unsqueeze(-1)).squeeze(-1)  # H y
        b_ = (st.H.mH @ yv.unsqueeze(-1)).squeeze(
            -1
        )  # (y^H H)^H, exactly as in the reference
        c = ((yv.conj() * a_).sum(-1) / gam).view(B, M, 1, 1)
        gam = gam.view(B, M, 1, 1)
        s_col = sv.unsqueeze(-1)
        s_row = sv.conj().unsqueeze(-2)
        a_col = a_.unsqueeze(-1)
        b_row = b_.conj().unsqueeze(-2)
        H_new = (
            st.H
            - (a_col @ s_row) / gam
            - (s_col @ b_row) / gam
            + (c + 1) * (s_col @ s_row) / gam
        )
        st.H = torch.where(ok.view(B, M, 1, 1), H_new, st.H)

        st.x = torch.where(v4, x + s_step, x)
        st.first_iter = st.first_iter & ~vad
        st.y = torch.where(v4, st.x + st.u / cfg.rho, st.y)

    # ------------------------------------------------------------------ communication steps
    def _transmit_local(self, vad: torch.Tensor):
        """y_r[n, t] <- conj(B_rc) * y_rc[slot of n]  (both nodes active); self entry if n active."""
        st = self.st
        y_from = st.y[:, self.T_idx, self.T_slot]  # [B,M,T,L]
        y_from = st.Bph[:, self.T_idx].conj() * y_from
        send = vad.unsqueeze(-1) & vad[:, self.T_idx] & self.not_self_T.unsqueeze(0)
        y_r = torch.where(send.unsqueeze(-1), y_from, st.y_r)

        y_self = st.y[:, self.node, self.self_slot]  # [B,M,L]
        if self.align_self:
            y_self = st.Bph.conj() * y_self
        selfm = vad.unsqueeze(-1) & self.is_self_T.unsqueeze(0)
        st.y_r = torch.where(selfm.unsqueeze(-1), y_self.unsqueeze(2), y_r)

    def _update_consensus(self, vad: torch.Tensor):
        st = self.st
        tm = self.T_mask.view(1, self.M, self.T_max, 1)
        z_new = (st.y_r * tm).sum(2) / self.T_deg.view(1, self.M, 1)
        z_own = st.z[:, self.node, self.self_slot]
        z = st.z.clone()
        z[:, self.node, self.self_slot] = torch.where(vad.unsqueeze(-1), z_new, z_own)
        st.z = z

    def _normalize_consensus(self, vad: torch.Tensor):
        st, cfg = self.st, self.cfg
        B, M = self.B, self.M
        z = st.z.clone()
        z_own = z[:, self.node, self.self_slot]
        if cfg.da:
            n_new = (st.z.abs() ** 2).sum(-1) * self.R_mask.view(1, M, self.R_max)
            n_old = st.n_inst
            num = torch.linalg.vector_norm(n_old - n_new, dim=-1)
            den = torch.linalg.vector_norm(n_old, dim=-1) + torch.linalg.vector_norm(
                n_new, dim=-1
            )
            gamma = (num / den.clamp_min(torch.finfo(self.rdt).tiny)).unsqueeze(-1)
            mix = (gamma * n_new + (1.0 - gamma) * st.phi_da) * self.R_mask.view(
                1, M, self.R_max
            )
            phi = mix.sum(-1) / self.R_deg.view(1, M)  # uniform weights 1/N_r
            st.n_inst = torch.where(vad.unsqueeze(-1), n_new, n_old)
            st.phi_own = torch.where(vad, phi, st.phi_own)
            scale = torch.sqrt(phi * M).unsqueeze(-1)
        else:
            n_z = torch.sqrt(
                (z_own.abs() ** 2).sum(dim=(-1, -2), keepdim=True)
            )  # [B,1,1]
            scale = n_z.expand(B, M, 1)
        z[:, self.node, self.self_slot] = torch.where(
            vad.unsqueeze(-1), z_own / scale, z_own
        )
        st.z = z

    def _transmit_consensus(self, vad: torch.Tensor):
        st = self.st
        z_own = st.z[:, self.node, self.self_slot]  # [B,M,L]
        z_recv = z_own[:, self.R_idx]  # [B,M,R,L]
        send = vad.unsqueeze(-1) & vad[:, self.R_idx] & self.not_self_R.unsqueeze(0)
        st.z = torch.where(send.unsqueeze(-1), z_recv, st.z)

    def _update_dual(self, vad: torch.Tensor):
        st, cfg = self.st, self.cfg
        res = cfg.rho * (st.x - st.Bph.unsqueeze(2) * st.z)
        res = self._mask_slots(res)
        st.u = st.u + res * vad.view(self.B, self.M, 1, 1)


def _to_device(glay: G.GramLayout, dev) -> G.GramLayout:
    for name, val in vars(glay).items():
        if isinstance(val, torch.Tensor):
            setattr(glay, name, val.to(dev))
    return glay


def run_bsi(
    cfg: DABSIConfig,
    layout: GraphLayout,
    x_raw: torch.Tensor,
    vad_frames: torch.Tensor,
    sigma_inv: torch.Tensor | None = None,
    delays: torch.Tensor | None = None,
    callback: Callable[[int, DABSI], None] | None = None,
    align_self: bool = False,
) -> DABSI:
    """
    x_raw: [B, M, N] real; vad_frames: [B, T, M] bool (T = (N-2L)//L + 1, VAD at each frame centre).
    Frames of length 2L with hop L, exactly like the reference simulation loop.
    """
    L = cfg.L
    frames = make_frames(x_raw.to(cfg.device), L)  # [B,T,M,2L]
    B, T, M, _ = frames.shape
    model = DABSI(cfg, layout, B, sigma_inv=sigma_inv, align_self=align_self)
    if delays is not None:
        model.set_delays(delays)
    for t in range(T):
        model.step(frames[:, t], vad_frames[:, t])
        if callback is not None:
            callback(t, model)
    return model
