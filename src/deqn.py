from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional
import time
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import trange

from .config import ModelParams, TrainConfig, set_seeds, PolicyName
from .quadrature import gauss_hermite
from .model_common import unpack_state, shock_laws_of_motion, identities
from .transforms import decode_outputs
from .policy_rules import i_taylor, i_taylor_zlb, i_modified_taylor, i_modified_taylor_zlb
from .residuals_a1 import residuals_a1
from .residuals_a2 import residuals_a2
from .residuals_a3 import residuals_a3
from .residuals_a2_zlb import residuals_a2_zlb
from .residuals_a3_zlb import residuals_a3_zlb
from .io_utils import save_csv, ensure_dir, make_run_dir, save_run_metadata, _normalize_artifacts_root, pack_config, save_torch


# Torch thread pools can only be safely configured once per process.
_CPU_THREAD_KNOBS_APPLIED = False


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: Tuple[int, int] = (512, 512),
        activation: str = "selu",
        *,
        init_mode: str = "default",
        init_scale: float = 0.01,
    ):
        super().__init__()
        act = nn.SELU if activation.lower() == "selu" else nn.ReLU
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), act(),
            nn.Linear(h1, h2), act(),
            nn.Linear(h2, d_out)
        )
        self._apply_init(init_mode=init_mode, init_scale=float(init_scale))

    def _apply_init(self, *, init_mode: str, init_scale: float) -> None:
        mode = str(init_mode).strip().lower()
        if mode != "author_variance_scaling":
            return
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            fan_in = float(m.weight.shape[1])
            fan_out = float(m.weight.shape[0])
            fan_avg = max(1.0, 0.5 * (fan_in + fan_out))
            # TensorFlow VarianceScaling(scale=s, mode='fan_avg', distribution='uniform'):
            # var = s / fan_avg, bound = sqrt(3*var)
            bound = float(np.sqrt(3.0 * float(init_scale) / fan_avg))
            nn.init.uniform_(m.weight, -bound, bound)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stack_residuals(res: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.stack([res[k] for k in keys], dim=-1)


def loss_from_residuals(
    resid: torch.Tensor,
    *,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    lt = str(loss_type).lower().strip()
    if lt == "mse":
        return (resid ** 2).mean()
    if lt == "huber":
        d = float(huber_delta)
        absr = torch.abs(resid)
        hub = torch.where(absr <= d, 0.5 * resid * resid, d * (absr - 0.5 * d))
        return hub.mean()
    raise ValueError(f"Unsupported loss_type={loss_type!r}; expected 'huber' or 'mse'")


def huber_elementwise(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    d = float(delta)
    ax = torch.abs(x)
    return torch.where(ax <= d, 0.5 * x * x, d * (ax - 0.5 * d))



def residual_metrics(resid: torch.Tensor, keys: List[str], *, tol: float) -> Dict[str, float]:
    """Compute scalar diagnostics for a residual matrix (B,K)."""
    with torch.no_grad():
        absr = torch.abs(resid)
        out: Dict[str, float] = {}
        out["loss"] = float((resid ** 2).mean().detach().cpu())
        out["rms"] = float(torch.sqrt(torch.mean(resid ** 2)).detach().cpu())
        out["max"] = float(torch.max(absr).detach().cpu())
        out["share_all_lt_tol"] = float((absr < tol).all(dim=1).float().mean().detach().cpu())
        # per-equation
        rms_i = torch.sqrt(torch.mean(resid ** 2, dim=0)).detach().cpu().numpy()
        max_i = torch.max(absr, dim=0).values.detach().cpu().numpy()
        viol_i = (absr >= tol).float().mean(dim=0).detach().cpu().numpy()
        for k, r, m, v in zip(keys, rms_i, max_i, viol_i):
            out[f"eq_rms__{k}"] = float(r)
            out[f"eq_max__{k}"] = float(m)
            out[f"eq_viol__{k}"] = float(v)
        return out

def residual_metrics_by_regime(x: torch.Tensor, resid: torch.Tensor, keys: List[str], *, tol: float, policy: PolicyName) -> Dict[str, float]:
    """Same metrics split by regime s in {0,1}."""
    st = unpack_state(x, policy)
    s = st.s
    out: Dict[str, float] = {}
    for r in [0, 1]:
        mask = (s == r)
        if mask.any():
            m = residual_metrics(resid[mask], keys, tol=tol)
            for kk, vv in m.items():
                out[f"r{r}__{kk}"] = float(vv)
        else:
            # keep keys stable
            out[f"r{r}__n"] = 0.0
    return out
def _gh_grid_3d(params: ModelParams, gh_n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute 3D Gauss–Hermite nodes and weights.

    Returns:
        eps_grid: (Q,3) tensor of (epsA, epsg, epst)
        w_grid:   (Q,)  tensor of product weights
    """
    gh = gauss_hermite(int(gh_n), params.device, params.dtype)
    nodes, weights = gh.nodes, gh.weights
    eps_grid = torch.cartesian_prod(nodes, nodes, nodes)  # (Q,3)
    w_grid = torch.cartesian_prod(weights, weights, weights).prod(dim=-1)  # (Q,)
    return eps_grid, w_grid


def _transition_probs_to_next(params: ModelParams, st) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return transition probabilities (to regime 0, to regime 1) as vectors of shape (B,).
    """
    p0 = params.P[st.s, 0]
    # Author dsge_taylor_para keeps p21 as a state component (path-specific bad->normal prob).
    p21_state = getattr(st, "p21", None)
    if p21_state is not None:
        p21_clamped = torch.clamp(
            p21_state.to(device=p0.device, dtype=p0.dtype),
            min=1e-8,
            max=1.0 - 1e-8,
        )
        p0_bad = p21_clamped
        p0_good = torch.full_like(p0_bad, 1.0 - float(params.p12))
        p0 = torch.where(st.s == 0, p0_good, p0_bad)
    p1 = 1.0 - p0
    return p0, p1


def expectation_operator_appendixB(
    params: ModelParams,
    st,
    gh_n: int,
    f,
    *,
    eps_grid: Optional[torch.Tensor] = None,
    w_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Appendix B expectation operator (vectorized)."""
    if eps_grid is None or w_grid is None:
        eps_grid, w_grid = _gh_grid_3d(params, gh_n)

    dev = params.device

    B = st.Delta_prev.shape[0]
    Q = eps_grid.shape[0]

    # shocks: (B,Q,2)
    epsA = eps_grid[:, 0].view(1, Q, 1).expand(B, Q, 2)
    epsg = eps_grid[:, 1].view(1, Q, 1).expand(B, Q, 2)
    epst = eps_grid[:, 2].view(1, Q, 1).expand(B, Q, 2)

    # next regime values: (B,Q,2) with r=0 => 0, r=1 => 1
    s_next = torch.stack(
        [
            torch.zeros(B, Q, device=dev, dtype=torch.long),
            torch.ones(B, Q, device=dev, dtype=torch.long),
        ],
        dim=-1,
    )

    val = f(epsA, epsg, epst, s_next)  # (B,Q,2) or (B,Q,2,K)

    wq = w_grid.view(1, Q, 1)  # (1,Q,1)
    p0, p1 = _transition_probs_to_next(params, st)
    pi = torch.stack([p0, p1], dim=-1).view(B, 1, 2)  # (B,1,2)

    if val.ndim == 3:
        return (val * wq * pi).sum(dim=1).sum(dim=-1)  # (B,)
    if val.ndim == 4:
        wq4 = w_grid.view(1, Q, 1, 1)
        pi4 = torch.stack([p0, p1], dim=-1).view(B, 1, 2, 1)
        return (val * wq4 * pi4).sum(dim=1).sum(dim=1)  # (B,K)

    raise ValueError(f"Unexpected f output ndim={val.ndim}; expected 3 or 4")


def implied_nominal_rate_from_euler(
    params: ModelParams,
    policy: PolicyName,
    x_t: torch.Tensor,
    out_t: Dict[str, torch.Tensor],
    gh_n: int,
    trainer: "Trainer",
) -> torch.Tensor:
    """
    Recover i_t from the Fisher/Euler equation post-training:
        1 = E_t[ beta * (1+i_t)/(1+pi_{t+1}) * lam_{t+1}/lam_t ]
    => i_t = ( E_t[ beta * lam_{t+1}/(lam_t*(1+pi_{t+1})) ] )^{-1} - 1
    """
    st = unpack_state(x_t, policy)
    # Ensure broadcastability against next-period objects shaped (B,Q,2).
    # out_t["lam"] is (B,), while on["lam"] inside the expectation is (B,Q,2).
    lam_t = out_t["lam"].view(-1, 1, 1)

    def x_next(epsA, epsg, epst, s_next):
        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, epsA, epsg, epst, s_next)
        Delta_cur = out_t["Delta"].view(-1, 1, 1).expand_as(logA_n)
        if policy == "commitment":
            vartheta_cur = out_t["vartheta"].view(-1, 1, 1).expand_as(logA_n)
            varrho_cur = out_t["varrho"].view(-1, 1, 1).expand_as(logA_n)
            has_c_prev = bool(getattr(trainer, "_commitment_has_c_prev", False)) or (st.c_prev is not None)
            if has_c_prev:
                c_prev_cur = out_t["c"].view(-1, 1, 1).expand_as(logA_n)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype), vartheta_cur, varrho_cur, c_prev_cur], dim=-1)
            return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype), vartheta_cur, varrho_cur], dim=-1)
        if policy == "commitment_zlb":
            vartheta_cur = out_t["vartheta"].view(-1, 1, 1).expand_as(logA_n)
            varrho_cur = out_t["varrho"].view(-1, 1, 1).expand_as(logA_n)
            c_prev_cur = out_t["c"].view(-1, 1, 1).expand_as(logA_n)
            i_nom_cur = out_t["i_nom"].view(-1, 1, 1).expand_as(logA_n)
            varphi_cur = out_t["varphi"].view(-1, 1, 1).expand_as(logA_n)
            return torch.stack(
                [Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype), vartheta_cur, varrho_cur, c_prev_cur, i_nom_cur, varphi_cur],
                dim=-1,
            )
        if policy == "taylor_para":
            if not bool(getattr(trainer, "_taylor_para_has_extended_state", False)):
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype)], dim=-1)
            i_old_cur = out_t["i_nom"].view(-1, 1, 1).expand_as(logA_n)
            if st.p21 is not None:
                p21_cur = st.p21.view(-1, 1, 1).expand_as(logA_n)
            else:
                p21_cur = torch.full_like(logA_n, float(params.p21))
            return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype), i_old_cur, p21_cur], dim=-1)
        return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype)], dim=-1)

    def f_term(epsA, epsg, epst, s_next):
        xn = x_next(epsA, epsg, epst, s_next)
        on = trainer._policy_outputs(xn)
        return params.beta * (on["lam"] / lam_t) * (1.0 / on["one_plus_pi"])

    eps_grid = getattr(trainer, "_eps_grid", None)
    w_grid = getattr(trainer, "_w_grid", None)
    Et = expectation_operator_appendixB(params, st, gh_n, f_term, eps_grid=eps_grid, w_grid=w_grid)
    return (1.0 / Et) - 1.0



@dataclass
class Trainer:
    params: ModelParams
    cfg: TrainConfig
    policy: PolicyName
    net: PolicyNetwork
    gh_n: Optional[int] = None
    rbar_by_regime: torch.Tensor | None = None

    _eps_grid: Optional[torch.Tensor] = None
    _w_grid: Optional[torch.Tensor] = None

    _t_accum: Dict[str, float] | None = None
    _t_count: int = 0
    last_train_stats: Dict[str, float] | None = None
    train_log_df: pd.DataFrame | None = None

    def __post_init__(self):
        # CPU knobs (safe; do not change equations)
        if self.params.device == "cpu":
            global _CPU_THREAD_KNOBS_APPLIED
            if not _CPU_THREAD_KNOBS_APPLIED:
                try:
                    if self.cfg.cpu_num_threads is not None:
                        torch.set_num_threads(int(self.cfg.cpu_num_threads))
                    if self.cfg.cpu_num_interop_threads is not None:
                        torch.set_num_interop_threads(int(self.cfg.cpu_num_interop_threads))
                except RuntimeError:
                    # Too late to change thread pools in this Python process (common in notebooks).
                    pass
                finally:
                    _CPU_THREAD_KNOBS_APPLIED = True
            mp = getattr(self.cfg, "matmul_precision", None)
            if mp is not None and hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision(str(mp))
                except Exception:
                    pass

        self.params = self.params.to_torch()
        p12_override = getattr(self.cfg, "author_commitment_zlb_p12", None)
        if (
            self.policy == "commitment_zlb"
            and str(getattr(self.cfg, "training_mode", "author")).strip().lower() == "author"
            and p12_override is not None
        ):
            self.params = replace(self.params, p12=float(p12_override)).to_torch()
        set_seeds(self.cfg.seed)

        # Keep model tensors on the same device/dtype as params to avoid
        # CPU/CUDA mismatches when notebooks instantiate net on default CPU.
        self.net = self.net.to(device=self.params.device, dtype=self.params.dtype)
        if self.rbar_by_regime is not None:
            self.rbar_by_regime = self.rbar_by_regime.to(device=self.params.device, dtype=self.params.dtype)

        # Detect optional state extensions expected by the current network.
        self._commitment_has_c_prev = False
        self._commitment_zlb_has_full_state = False
        self._taylor_para_has_extended_state = False
        if self.policy == "commitment":
            try:
                self._commitment_has_c_prev = int(self.net.net[0].in_features) >= 8
            except Exception:
                self._commitment_has_c_prev = False
        if self.policy == "commitment_zlb":
            try:
                self._commitment_zlb_has_full_state = int(self.net.net[0].in_features) >= 10
            except Exception:
                self._commitment_zlb_has_full_state = False
        if self.policy == "taylor_para":
            try:
                self._taylor_para_has_extended_state = int(self.net.net[0].in_features) >= 7
            except Exception:
                self._taylor_para_has_extended_state = False

        if self.policy in ("mod_taylor", "mod_taylor_zlb"):
            if self.rbar_by_regime is None:
                raise ValueError(f"policy={self.policy} requires rbar_by_regime from flex-price SSS")
            if self.rbar_by_regime.numel() != 2:
                raise ValueError("rbar_by_regime must have 2 elements (one per regime)")

        if self.policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            self.res_keys = ["res_euler", "res_XiN", "res_XiD", "res_pstar_def"]
        elif self.policy == "taylor_para":
            # Author dsge_taylor_para includes eq_5 (dispersion law) and eq_8 (rule residual).
            self.res_keys = ["res_euler", "res_XiN", "res_XiD", "res_Delta", "res_pstar_def", "res_i_rule"]
        elif self.policy == "discretion":
            self.res_keys = ["res_c_foc", "res_Delta_foc", "res_XiN_rec", "res_XiD_rec", "res_pstar_def"]
        elif self.policy == "discretion_zlb":
            self.res_keys = ["res_c_foc", "res_Delta_foc", "res_XiN_rec", "res_XiD_rec", "res_zlb_comp", "res_pstar_def"]
        elif self.policy == "commitment":
            self.res_keys = ["res_c_foc", "res_Delta_foc", "res_XiN_rec", "res_XiD_rec", "res_pstar_def"]
        elif self.policy == "commitment_zlb":
            self.res_keys = ["res_c_foc", "res_Delta_foc", "res_XiN_rec", "res_XiD_rec", "res_euler_i", "res_zlb_comp", "res_pstar_def"]
        else:
            raise ValueError(self.policy)

        self.eq_labels = list(self.res_keys)

        # timers
        self._t_accum = {}
        self.last_train_stats = None
        # initialize GH grid to phase1 by default unless user overrides gh_n
        if self.gh_n is None:
            self.gh_n = int(self.cfg.phase1.gh_n_train)
        self._set_gh(int(self.gh_n))

    def _set_gh(self, gh_n: int) -> None:
        """Switch GH order + refresh cached 3D grids (does not change equations)."""
        self.gh_n = int(gh_n)
        self._eps_grid, self._w_grid = _gh_grid_3d(self.params, self.gh_n)

    # --------------------
    # State init / stepping
    # --------------------
    def simulate_initial_state(self, B: int, commitment_sss: Dict | None = None) -> torch.Tensor:
        dev, dt = self.params.device, self.params.dtype
        rho_A, sig_A = float(self.params.rho_A), float(self.params.sigma_A)
        rho_g, sig_g = float(self.params.rho_g), float(self.params.sigma_g)
        rho_xi, sig_xi = float(self.params.rho_tau), float(self.params.sigma_tau)

        # Public Keras Hooks.py semantics:
        # - exogenous states centered at zero with author scaling
        # - 50/50 initial regime mix
        # - small noise around disp_old_x = 1
        sd_logA = (sig_A ** 2) / max(1e-12, (1.0 - rho_A ** 2)) if abs(rho_A) < 1.0 else sig_A ** 2
        sd_logg = (sig_g ** 2) / max(1e-12, (1.0 - rho_g ** 2)) if abs(rho_g) < 1.0 else sig_g ** 2
        sd_xi = (sig_xi ** 2) / max(1e-12, (1.0 - rho_xi ** 2)) if abs(rho_xi) < 1.0 else sig_xi ** 2
        logA = sd_logA * torch.randn(B, device=dev, dtype=dt)
        logg = sd_logg * torch.randn(B, device=dev, dtype=dt)
        xi = sd_xi * torch.randn(B, device=dev, dtype=dt)
        s = torch.where(
            torch.rand(B, device=dev, dtype=dt) < 0.5,
            torch.zeros(B, device=dev, dtype=torch.long),
            torch.ones(B, device=dev, dtype=torch.long),
        )
        if self.policy == "taylor_para":
            Delta_prev = torch.ones(B, device=dev, dtype=dt) + 2e-3 * torch.randn(B, device=dev, dtype=dt)
            i_nom_ss = (1.0 + float(self.params.pi_bar)) / float(self.params.beta) - 1.0
            i_old = torch.tensor(i_nom_ss, device=dev, dtype=dt) + 8e-3 * torch.randn(B, device=dev, dtype=dt)
            p21_l = float(getattr(self.params, "p21_l", self.params.p21))
            p21_u = float(getattr(self.params, "p21_u", self.params.p21))
            p21 = torch.empty(B, device=dev, dtype=dt).uniform_(p21_l, p21_u)
            if bool(getattr(self, "_taylor_para_has_extended_state", False)):
                return torch.stack([Delta_prev, logA, logg, xi, s.to(dt), i_old, p21], dim=-1)
            return torch.stack([Delta_prev, logA, logg, xi, s.to(dt)], dim=-1)

        disp_std = 1e-4 if self.policy in ("commitment", "commitment_zlb") else 1e-3
        Delta_prev = torch.ones(B, device=dev, dtype=dt) + disp_std * torch.randn(B, device=dev, dtype=dt)

        if self.policy not in ("commitment", "commitment_zlb"):
            return torch.stack([Delta_prev, logA, logg, xi, s.to(dt)], dim=-1)

        # Keep API compatibility, but this path intentionally ignores commitment_sss.
        _ = commitment_sss

        if self.policy == "commitment":
            # Keras commitment Hooks.py constants:
            # vartheta_old=-0.019182, rho_old=0.016500, c_old=0.921336 + Gaussian noise.
            normal_noise = torch.randn(B, 3, device=dev, dtype=dt) * torch.tensor(
                [0.027, 0.023, 0.052], device=dev, dtype=dt
            )
            vartheta_old = torch.tensor(-0.019182, device=dev, dtype=dt) + normal_noise[:, 0]
            rho_old = torch.tensor(0.016500, device=dev, dtype=dt) + normal_noise[:, 1]
            c_old = torch.clamp(torch.tensor(0.921336, device=dev, dtype=dt) + normal_noise[:, 2], min=1e-8)

            if self._commitment_has_c_prev:
                return torch.stack([Delta_prev, logA, logg, xi, s.to(dt), vartheta_old, rho_old, c_old], dim=-1)
            return torch.stack([Delta_prev, logA, logg, xi, s.to(dt), vartheta_old, rho_old], dim=-1)

        # commitment_zlb hooks constants (author dsge_zlb_commitment/Hooks.py)
        normal_noise = torch.randn(B, 6, device=dev, dtype=dt) * torch.tensor(
            [0.027, 0.023, 0.052, 0.0001, 0.001, 0.001], device=dev, dtype=dt
        )
        vartheta_old = torch.tensor(-0.019182, device=dev, dtype=dt) + normal_noise[:, 0]
        rho_old = torch.tensor(0.016500, device=dev, dtype=dt) + normal_noise[:, 1]
        c_old = torch.clamp(torch.tensor(0.921336, device=dev, dtype=dt) + normal_noise[:, 2], min=1e-8)
        Delta_prev = torch.tensor(1.0, device=dev, dtype=dt) + normal_noise[:, 3]
        i_nom_old = torch.clamp(torch.tensor(0.002461, device=dev, dtype=dt) + normal_noise[:, 4], min=0.0)
        varphi_old = torch.minimum(
            torch.zeros(B, device=dev, dtype=dt),
            torch.tensor(-0.000012, device=dev, dtype=dt) + normal_noise[:, 5],
        )
        return torch.stack(
            [Delta_prev, logA, logg, xi, s.to(dt), vartheta_old, rho_old, c_old, i_nom_old, varphi_old],
            dim=-1,
        )

    def _author_post_init_state(self, x: torch.Tensor, episode_idx: int) -> torch.Tensor:
        """
        Approximate author Hooks.post_init() behavior after each episode.
        """
        dev, dt = self.params.device, self.params.dtype
        B = int(x.shape[0])
        x_new = x.clone()

        rho_A, sig_A = float(self.params.rho_A), float(self.params.sigma_A)
        rho_g, sig_g = float(self.params.rho_g), float(self.params.sigma_g)
        rho_xi, sig_xi = float(self.params.rho_tau), float(self.params.sigma_tau)
        sd_logA = (sig_A ** 2) / max(1e-12, (1.0 - rho_A ** 2)) if abs(rho_A) < 1.0 else sig_A ** 2
        sd_logg = (sig_g ** 2) / max(1e-12, (1.0 - rho_g ** 2)) if abs(rho_g) < 1.0 else sig_g ** 2
        sd_xi = (sig_xi ** 2) / max(1e-12, (1.0 - rho_xi ** 2)) if abs(rho_xi) < 1.0 else sig_xi ** 2

        def _reset_exogenous_if_needed() -> None:
            if int(episode_idx) >= 2:
                return
            x_new[:, 1] = sd_logA * torch.randn(B, device=dev, dtype=dt)
            x_new[:, 2] = sd_logg * torch.randn(B, device=dev, dtype=dt)
            x_new[:, 3] = sd_xi * torch.randn(B, device=dev, dtype=dt)
            s = torch.where(
                torch.rand(B, device=dev, dtype=dt) < 0.5,
                torch.zeros(B, device=dev, dtype=torch.long),
                torch.ones(B, device=dev, dtype=torch.long),
            )
            x_new[:, 4] = s.to(dt)

        if self.policy == "taylor_para":
            # Always refresh p21_x in author hooks.
            p21_l = float(getattr(self.params, "p21_l", self.params.p21))
            p21_u = float(getattr(self.params, "p21_u", self.params.p21))
            if x_new.shape[1] >= 7:
                x_new[:, 6] = torch.empty(B, device=dev, dtype=dt).uniform_(p21_l, p21_u)
            if int(episode_idx) < 2:
                x_new[:, 0] = 1.0 + 2e-3 * torch.randn(B, device=dev, dtype=dt)
                i_nom_ss = (1.0 + float(self.params.pi_bar)) / float(self.params.beta) - 1.0
                if x_new.shape[1] >= 6:
                    x_new[:, 5] = torch.tensor(i_nom_ss, device=dev, dtype=dt) + 8e-3 * torch.randn(B, device=dev, dtype=dt)
            _reset_exogenous_if_needed()
            return x_new

        if self.policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "discretion", "discretion_zlb"):
            if int(episode_idx) < 2:
                x_new[:, 0] = 1.0 + 1e-3 * torch.randn(B, device=dev, dtype=dt)
            _reset_exogenous_if_needed()
            return x_new

        if self.policy == "commitment":
            noise = torch.randn(B, 4, device=dev, dtype=dt) * torch.tensor(
                [0.027, 0.023, 0.052, 0.0001], device=dev, dtype=dt
            )
            x_new[:, 5] = torch.tensor(-0.019182, device=dev, dtype=dt) + noise[:, 0]
            x_new[:, 6] = torch.tensor(0.016500, device=dev, dtype=dt) + noise[:, 1]
            if x_new.shape[1] >= 8:
                x_new[:, 7] = torch.clamp(torch.tensor(0.921336, device=dev, dtype=dt) + noise[:, 2], min=1e-8)
            x_new[:, 0] = 1.0 + noise[:, 3]
            _reset_exogenous_if_needed()
            return x_new

        if self.policy == "commitment_zlb":
            noise = torch.randn(B, 6, device=dev, dtype=dt) * torch.tensor(
                [0.027, 0.023, 0.052, 0.0001, 0.001, 0.001], device=dev, dtype=dt
            )
            x_new[:, 5] = torch.tensor(-0.019182, device=dev, dtype=dt) + noise[:, 0]
            x_new[:, 6] = torch.tensor(0.016500, device=dev, dtype=dt) + noise[:, 1]
            x_new[:, 7] = torch.clamp(torch.tensor(0.921336, device=dev, dtype=dt) + noise[:, 2], min=1e-8)
            x_new[:, 0] = 1.0 + noise[:, 3]
            x_new[:, 8] = torch.clamp(torch.tensor(0.002461, device=dev, dtype=dt) + noise[:, 4], min=0.0)
            x_new[:, 9] = torch.minimum(
                torch.zeros(B, device=dev, dtype=dt),
                torch.tensor(-0.000012, device=dev, dtype=dt) + noise[:, 5],
            )
            _reset_exogenous_if_needed()
            return x_new

        return x_new

    def _apply_author_hard_bounds(self, st, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Mirror author-side hard clipping (State/PolicyState/Definitions wrappers)
        on decoded economic objects.
        """
        if not bool(getattr(self.cfg, "use_author_bounds", True)):
            return out

        out = dict(out)

        def _clip(name: str, lo: float | None = None, hi: float | None = None) -> None:
            if name not in out:
                return
            z = out[name]
            if lo is not None:
                z = torch.clamp(z, min=float(lo))
            if hi is not None:
                z = torch.clamp(z, max=float(hi))
            out[name] = z

        eps = float(self.params.eps)
        gamma = float(self.params.gamma)
        omega = float(self.params.omega)
        pstar_low = float(getattr(self.cfg, "pstar_low", 0.9))
        pstar_high = float(getattr(self.cfg, "pstar_high", 1.1))
        pi_low = float(getattr(self.cfg, "pi_low", -0.1))
        pi_high = float(getattr(self.cfg, "pi_high", 0.1))

        # Common core bounds from author Variables.py definitions.
        _clip("c", 0.6, 1.4)
        one_plus_pi_low = max(1e-8, 1.0 + pi_low)
        one_plus_pi_high = max(one_plus_pi_low + 1e-8, 1.0 + pi_high)
        _clip("pi_aux", one_plus_pi_low ** eps, one_plus_pi_high ** eps)

        taylor_family = ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb")
        if self.policy in taylor_family:
            _clip("XiN", 1.0, 8.0)
            _clip("XiD", 1.0, 8.0)
            delta_lo, delta_hi = 0.9, 1.1
        elif self.policy == "commitment_zlb":
            _clip("XiN", 0.0, 7.0)
            _clip("XiD", 0.0, 7.0)
            _clip("p_star_aux", max(1e-12, pstar_high ** (-eps)), max(1e-12, pstar_low ** (-eps)))
            _clip("vartheta", -1.0, 1.0)
            _clip("varrho", -1.0, 1.0)
            _clip("i_nom", None, 0.1)
            _clip("varphi", -2.0, None)
            delta_lo, delta_hi = 0.6, 1.4
        elif self.policy == "commitment":
            _clip("XiN", 0.0, 12.0)
            _clip("XiD", 0.0, 12.0)
            _clip("p_star_aux", max(1e-12, pstar_high ** (-eps)), max(1e-12, pstar_low ** (-eps)))
            _clip("vartheta", -2.0, 1.0)
            _clip("varrho", -1.0, 1.0)
            delta_lo, delta_hi = 0.6, 1.4
        elif self.policy == "discretion_zlb":
            _clip("XiN", 0.0, 12.0)
            _clip("XiD", 0.0, 12.0)
            _clip("varphi", -1.0, None)
            delta_lo, delta_hi = 0.6, 1.4
        else:
            _clip("XiN", 0.0, 12.0)
            _clip("XiD", 0.0, 12.0)
            delta_lo, delta_hi = 0.6, 1.4

        # Rebuild derived objects from clipped primitives.
        out["one_plus_pi"] = torch.clamp(out["pi_aux"], min=1e-12).pow(1.0 / eps)
        out["pi"] = out["one_plus_pi"] - 1.0
        out["pstar"] = self.params.M * out["XiN"] / torch.clamp(out["XiD"], min=1e-12)
        _clip("pstar", pstar_low, pstar_high)
        out["Delta"] = self.params.theta * out["pi_aux"] * st.Delta_prev + (1.0 - self.params.theta) * out["p_star_aux"]
        _clip("Delta", delta_lo, delta_hi)

        out["A"] = torch.exp(st.logA)
        out["g"] = self.params.g_bar * torch.exp(st.loggtilde)
        out["y"] = out["c"] + out["g"]
        out["lam"] = torch.clamp(out["c"], min=1e-12).pow(-gamma)
        out["h"] = out["y"] * out["Delta"] / torch.clamp(out["A"], min=1e-12)
        out["w"] = out["h"].pow(omega) / torch.clamp(out["lam"], min=1e-12)

        # Keep policy-rule objects consistent with clipped inflation.
        if self.policy == "taylor":
            out["i_nom"] = i_taylor(self.params, out["pi"])
            out["i_rule_target"] = out["i_nom"]
        elif self.policy == "taylor_zlb":
            out["i_nom"] = i_taylor_zlb(self.params, out["pi"], zlb_floor=0.0)
            out["i_rule_target"] = out["i_nom"]
        elif self.policy == "mod_taylor":
            if self.rbar_by_regime is None:
                raise ValueError("policy=mod_taylor requires rbar_by_regime from flex-price SSS")
            out["i_nom"] = i_modified_taylor(self.params, out["pi"], self.rbar_by_regime, st.s)
            out["i_rule_target"] = out["i_nom"]
        elif self.policy == "mod_taylor_zlb":
            if self.rbar_by_regime is None:
                raise ValueError("policy=mod_taylor_zlb requires rbar_by_regime from flex-price SSS")
            out["i_nom"] = i_modified_taylor_zlb(self.params, out["pi"], self.rbar_by_regime, st.s, zlb_floor=0.0)
            out["i_rule_target"] = out["i_nom"]
        elif self.policy == "taylor_para":
            i_old = st.i_old if st.i_old is not None else torch.zeros_like(out["pi"])
            out["i_rule_target"] = self.params.rho_i * i_old + (1.0 - self.params.rho_i) * i_taylor(self.params, out["pi"])

        return out

    def _policy_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # If x was created under torch.inference_mode(), it is an 'inference tensor' and
        # cannot be used in autograd-tracked computations. Clone it when grad is enabled.
        if torch.is_grad_enabled() and getattr(x, 'is_inference', False):
            x = x.clone()
        st = unpack_state(x, self.policy)
        raw = self.net(x)
        floors = {
            "c": self.cfg.c_floor,
            "Delta": self.cfg.delta_floor,
            "pstar": self.cfg.pstar_floor,
            "pi_low": float(getattr(self.cfg, "pi_low", -0.1)),
            "pi_high": float(getattr(self.cfg, "pi_high", 0.1)),
            "pstar_low": float(getattr(self.cfg, "pstar_low", 0.9)),
            "pstar_high": float(getattr(self.cfg, "pstar_high", 1.1)),
        }
        out = decode_outputs(
            self.policy,
            raw,
            floors=floors,
            params=self.params,
            st=st,
            rbar_by_regime=self.rbar_by_regime if self.policy in ("mod_taylor", "mod_taylor_zlb") else None,
        )
        return self._apply_author_hard_bounds(st, out)

    def _bounds_penalty(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Author-like soft penalties for bound violations (Huber)."""
        delta = float(getattr(self.cfg, "huber_delta", 1.0))
        terms: List[torch.Tensor] = []

        def _add_bound(name: str, lo: float | None, hi: float | None) -> None:
            if name not in out:
                return
            if lo is not None:
                below = torch.clamp(float(lo) - out[name], min=0.0)
                terms.append(huber_elementwise(below, delta).sum())
            if hi is not None:
                above = torch.clamp(out[name] - float(hi), min=0.0)
                terms.append(huber_elementwise(above, delta).sum())

        # Common nominal bounds
        pi_low = float(getattr(self.cfg, "pi_low", -0.1))
        pi_high = float(getattr(self.cfg, "pi_high", 0.1))
        pstar_low = float(getattr(self.cfg, "pstar_low", 0.9))
        pstar_high = float(getattr(self.cfg, "pstar_high", 1.1))

        _add_bound("pi", pi_low, pi_high)
        _add_bound("pstar", pstar_low, pstar_high)

        # Author Definitions also penalize auxiliary nominal objects.
        one_plus_pi_low = max(1e-8, 1.0 + pi_low)
        one_plus_pi_high = max(one_plus_pi_low + 1e-8, 1.0 + pi_high)
        _add_bound(
            "pi_aux",
            one_plus_pi_low ** float(self.params.eps),
            one_plus_pi_high ** float(self.params.eps),
        )

        # Wider author-like variable penalties
        _add_bound("c", 0.6, 1.4)
        if self.policy in ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            _add_bound("Delta", 0.9, 1.1)
            _add_bound("XiN", 1.0, 8.0)
            _add_bound("XiD", 1.0, 8.0)
        elif self.policy == "commitment_zlb":
            _add_bound("Delta", 0.6, 1.4)
            _add_bound("XiN", 0.0, 7.0)
            _add_bound("XiD", 0.0, 7.0)
            _add_bound(
                "p_star_aux",
                max(1e-12, pstar_high ** (-float(self.params.eps))),
                max(1e-12, pstar_low ** (-float(self.params.eps))),
            )
            _add_bound("vartheta", -1.0, 1.0)
            _add_bound("varrho", -1.0, 1.0)
            _add_bound("i_nom", None, 0.1)
            _add_bound("varphi", -2.0, None)
        elif self.policy == "commitment":
            _add_bound("Delta", 0.6, 1.4)
            _add_bound("XiN", 0.0, 12.0)
            _add_bound("XiD", 0.0, 12.0)
            _add_bound(
                "p_star_aux",
                max(1e-12, pstar_high ** (-float(self.params.eps))),
                max(1e-12, pstar_low ** (-float(self.params.eps))),
            )
            _add_bound("vartheta", -2.0, 1.0)
            _add_bound("varrho", -1.0, 1.0)
        elif self.policy == "discretion_zlb":
            _add_bound("Delta", 0.6, 1.4)
            _add_bound("XiN", 0.0, 12.0)
            _add_bound("XiD", 0.0, 12.0)
            _add_bound("varphi", -1.0, None)
        else:
            _add_bound("Delta", 0.6, 1.4)
            _add_bound("XiN", 0.0, 12.0)
            _add_bound("XiD", 0.0, 12.0)

        if not terms:
            # keep graph/device consistent
            return torch.zeros((), device=self.params.device, dtype=self.params.dtype)
        return torch.stack(terms).sum()

    def _raw_policy_penalty(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Author-style penalty on policy outputs in raw/policy space.
        This mirrors Equilibrium.py spirit (bounds in policy/definition spaces).
        """
        if raw.ndim != 2:
            return torch.zeros((), device=self.params.device, dtype=self.params.dtype)
        delta = float(getattr(self.cfg, "huber_delta", 1.0))
        terms: List[torch.Tensor] = []

        def _add(col: int, lo: float | None, hi: float | None) -> None:
            if col < 0 or col >= int(raw.shape[-1]):
                return
            z = raw[:, col]
            if lo is not None:
                below = torch.clamp(float(lo) - z, min=0.0)
                terms.append(huber_elementwise(below, delta).sum())
            if hi is not None:
                above = torch.clamp(z - float(hi), min=0.0)
                terms.append(huber_elementwise(above, delta).sum())

        # Taylor-family policy vectors in current code:
        # - taylor/mod_taylor/*zlb: [num, den, p_star_aux_shift, cons_shift]
        # - taylor_para: [num, den, disp, p_star_aux_shift, cons_shift, i_nom_shift]
        #   (disp output is not used in author dsge_taylor_para Definitions/Dynamics)
        if self.policy in ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            c_ss = float((1.0 / self.params.M) ** (1.0 / (self.params.omega + self.params.gamma)))
            _add(0, 1.0, 8.0)  # XiN
            _add(1, 1.0, 8.0)  # XiD
            if self.policy == "taylor_para":
                _add(3, float(max(1e-12, 0.9 ** (-self.params.eps)) - 1.0), float(1.1 ** (-self.params.eps) - 1.0))
                _add(4, 0.6 - c_ss, 1.4 - c_ss)  # c = c_ss + cons_shift
            else:
                _add(2, float(max(1e-12, 0.9 ** (-self.params.eps)) - 1.0), float(1.1 ** (-self.params.eps) - 1.0))
                _add(3, 0.6 - c_ss, 1.4 - c_ss)  # c = c_ss + cons_shift

        if not terms:
            return torch.zeros((), device=self.params.device, dtype=self.params.dtype)
        return torch.stack(terms).sum()

    def _training_loss_from_states(self, x: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        """Combined training objective under selected training_mode."""
        base = loss_from_residuals(
            resid,
            loss_type=getattr(self.cfg, "loss_type", "huber"),
            huber_delta=float(getattr(self.cfg, "huber_delta", 1.0)),
        )
        if bool(getattr(self.cfg, "use_penalty_bounds", True)):
            out = self._policy_outputs(x)
            w = float(getattr(self.cfg, "bounds_penalty_weight", 1.0))
            base = base + w * self._bounds_penalty(out)
            use_raw_penalty = bool(getattr(self.cfg, "use_author_raw_penalty", True))
            # Avoid double-penalizing Taylor-family bounds:
            # bounds already apply in decoded/definition space (author-style focus).
            if self.policy in ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
                use_raw_penalty = False
            if use_raw_penalty:
                raw = self.net(x)
                base = base + w * self._raw_policy_penalty(raw)
        return base

    @torch.no_grad()
    def _step_state(self, x: torch.Tensor) -> torch.Tensor:
        """One-step law of motion for x_{t+1} (Appendix B)."""
        dev, dt = self.params.device, self.params.dtype
        B = x.shape[0]
        st = unpack_state(x, self.policy)
        out = self._policy_outputs(x)

        epsA = torch.randn(B, device=dev, dtype=dt)
        epsg = torch.randn(B, device=dev, dtype=dt)
        epst = torch.randn(B, device=dev, dtype=dt)
        if self.policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            nss = int(getattr(self.cfg, "author_n_steady_state_batches", 0) or 0)
            min_b = int(getattr(self.cfg, "author_n_steady_state_min_batch", 500) or 500)
            if nss > 0 and B > min_b:
                nss_eff = min(int(B), int(nss))
                if nss_eff > 0:
                    epsA[-nss_eff:] = 0.0
                    epsg[-nss_eff:] = 0.0
                    epst[-nss_eff:] = 0.0

        u = torch.rand(B, device=dev, dtype=dt)
        p0, _ = _transition_probs_to_next(self.params, st)
        s_next = torch.where(u < p0, torch.zeros_like(st.s), torch.ones_like(st.s))

        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)

        if self.policy == "commitment":
            if st.c_prev is not None or self._commitment_has_c_prev:
                return torch.stack(
                    [out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"], out["c"]],
                    dim=-1,
                )
            return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"]], dim=-1)

        if self.policy == "commitment_zlb":
            return torch.stack(
                [
                    out["Delta"],
                    logA_n,
                    logg_n,
                    xi_n,
                    s_n.to(dt),
                    out["vartheta"],
                    out["varrho"],
                    out["c"],
                    out["i_nom"],
                    out["varphi"],
                ],
                dim=-1,
            )

        if self.policy == "taylor_para":
            if not bool(getattr(self, "_taylor_para_has_extended_state", False)):
                return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
            if st.p21 is not None:
                p21_prev = st.p21
            else:
                p21_prev = torch.full_like(out["Delta"], float(self.params.p21))
            return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["i_nom"], p21_prev], dim=-1)

        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)

    # --------------------
    # Residuals (unchanged economics)
    # --------------------
    def _residuals(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is a normal tensor when computing residuals with autograd.
        if torch.is_grad_enabled() and getattr(x, 'is_inference', False):
            x = x.clone()
        if self.policy in ("discretion", "discretion_zlb") and torch.is_grad_enabled() and (not x.requires_grad):
            x = x.clone().detach().requires_grad_(True)
        out = self._policy_outputs(x)
        st = unpack_state(x, self.policy)

        if self.policy in ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            lam_t = out["lam"]
            lam_tg = lam_t.view(-1, 1, 1)
            i_t = out["i_nom"]
            i_tg = i_t.view(-1, 1, 1)

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                if self.policy == "taylor_para":
                    if not bool(getattr(self, "_taylor_para_has_extended_state", False)):
                        return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1)
                    i_old_cur = out["i_nom"].view(-1, 1, 1).expand_as(logA_n)
                    if st.p21 is not None:
                        p21_cur = st.p21.view(-1, 1, 1).expand_as(logA_n)
                    else:
                        p21_cur = torch.full_like(logA_n, float(self.params.p21))
                    return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), i_old_cur, p21_cur], dim=-1)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1)

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)
                lam_ratio = on["lam"] / lam_tg
                pi_aux_n = on["pi_aux"]
                term_XiN = self.params.beta * self.params.theta * lam_ratio * pi_aux_n * on["XiN"]
                term_XiD = self.params.beta * self.params.theta * lam_ratio * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                term_eul = self.params.beta * (on["lam"] * (1.0 + i_tg) * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) / pi_aux_n) / lam_tg
                return torch.stack([term_XiN, term_XiD, term_eul], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_XiN = Et_all[..., 0]
            Et_XiD = Et_all[..., 1]
            Et_eul = Et_all[..., 2]

            if self.policy == "taylor_para":
                res = residuals_a1(
                    self.params,
                    st,
                    out,
                    Et_XiN,
                    Et_XiD,
                    Et_eul,
                    i_t_current=out.get("i_nom"),
                    i_rule_target=out.get("i_rule_target"),
                )
            else:
                res = residuals_a1(self.params, st, out, Et_XiN, Et_XiD, Et_eul)
            return stack_residuals(res, self.res_keys)

        if self.policy == "discretion":
            lam_t = out["lam"]
            lam_tg = lam_t.view(-1, 1, 1)

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1)

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)

                pi_aux_n = on["pi_aux"]
                lam_ratio = on["lam"] / lam_tg

                F = self.params.theta * self.params.beta * on["c"].pow(-self.params.gamma) * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                G = self.params.theta * self.params.beta * on["c"].pow(-self.params.gamma) * pi_aux_n * on["XiN"]

                theta_term = self.params.theta * pi_aux_n * on["zeta"]
                XiN_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n * on["XiN"]
                XiD_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]

                return torch.stack([F, G, theta_term, XiN_rec, XiD_rec], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_F = Et_all[..., 0]
            Et_G = Et_all[..., 1]
            Et_theta = Et_all[..., 2]
            Et_XiN = Et_all[..., 3]
            Et_XiD = Et_all[..., 4]

            # Author code computes dF/dDelta and dG/dDelta w.r.t. lagged dispersion in state.
            dEtF_dx = torch.autograd.grad(Et_F.sum(), x, create_graph=True, retain_graph=True)[0]
            dEtG_dx = torch.autograd.grad(Et_G.sum(), x, create_graph=True)[0]
            Et_dF = dEtF_dx[..., 0]
            Et_dG = dEtG_dx[..., 0]

            res = residuals_a2(self.params, st, out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD)
            return stack_residuals(res, self.res_keys)

        if self.policy == "discretion_zlb":
            lam_t = out["lam"]
            lam_tg = lam_t.view(-1, 1, 1)

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1)

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)

                pi_aux_n = on["pi_aux"]
                lam_ratio = on["lam"] / lam_tg

                F = self.params.theta * self.params.beta * on["c"].pow(-self.params.gamma) * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                G = self.params.theta * self.params.beta * on["c"].pow(-self.params.gamma) * pi_aux_n * on["XiN"]
                H = on["lam"] / pi_aux_n.pow(1.0 / self.params.eps)

                theta_term = self.params.theta * pi_aux_n * on["zeta"]
                XiN_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n * on["XiN"]
                XiD_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]

                return torch.stack([F, G, theta_term, XiN_rec, XiD_rec, H], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_F = Et_all[..., 0]
            Et_G = Et_all[..., 1]
            Et_theta = Et_all[..., 2]
            Et_XiN = Et_all[..., 3]
            Et_XiD = Et_all[..., 4]
            Et_H = Et_all[..., 5]

            dEtF_dx = torch.autograd.grad(Et_F.sum(), x, create_graph=True, retain_graph=True)[0]
            dEtG_dx = torch.autograd.grad(Et_G.sum(), x, create_graph=True)[0]
            Et_dF = dEtF_dx[..., 0]
            Et_dG = dEtG_dx[..., 0]

            res = residuals_a2_zlb(
                self.params,
                st,
                out,
                Et_F,
                Et_G,
                Et_dF,
                Et_dG,
                Et_theta,
                Et_XiN,
                Et_XiD,
                Et_H,
                eps_cc=0.0,
            )
            return stack_residuals(res, self.res_keys)

        if self.policy == "commitment":
            c_t = out["c"]
            lam_t = out["lam"]
            lam_tg = lam_t.view(-1, 1, 1)
            gamma = self.params.gamma
            c_tg = c_t.view(-1, 1, 1)

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                vartheta_cur = out["vartheta"].view(-1, 1, 1).expand_as(logA_n)
                varrho_cur = out["varrho"].view(-1, 1, 1).expand_as(logA_n)
                if st.c_prev is not None or self._commitment_has_c_prev:
                    c_prev_cur = out["c"].view(-1, 1, 1).expand_as(logA_n)
                    return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vartheta_cur, varrho_cur, c_prev_cur], dim=-1)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vartheta_cur, varrho_cur], dim=-1)

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)
                pi_aux_n = on["pi_aux"]
                lam_ratio = on["lam"] / lam_tg

                XiN_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n * on["XiN"]
                XiD_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                theta_term = self.params.theta * on["zeta"] * pi_aux_n

                termN = self.params.beta * self.params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * pi_aux_n * on["XiN"]
                termD = self.params.beta * self.params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                return torch.stack([XiN_rec, XiD_rec, termN, termD, theta_term], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_XiN = Et_all[..., 0]
            Et_XiD = Et_all[..., 1]
            Et_termN = Et_all[..., 2]
            Et_termD = Et_all[..., 3]
            Et_theta = Et_all[..., 4]

            res = residuals_a3(self.params, st, out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta)
            return stack_residuals(res, self.res_keys)

        if self.policy == "commitment_zlb":
            c_t = out["c"]
            lam_t = out["lam"]
            lam_tg = lam_t.view(-1, 1, 1)
            gamma = self.params.gamma
            c_tg = c_t.view(-1, 1, 1)

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                vartheta_cur = out["vartheta"].view(-1, 1, 1).expand_as(logA_n)
                varrho_cur = out["varrho"].view(-1, 1, 1).expand_as(logA_n)
                c_prev_cur = out["c"].view(-1, 1, 1).expand_as(logA_n)
                i_nom_cur = out["i_nom"].view(-1, 1, 1).expand_as(logA_n)
                varphi_cur = out["varphi"].view(-1, 1, 1).expand_as(logA_n)
                return torch.stack(
                    [Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vartheta_cur, varrho_cur, c_prev_cur, i_nom_cur, varphi_cur],
                    dim=-1,
                )

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)
                pi_aux_n = on["pi_aux"]
                lam_ratio = on["lam"] / lam_tg

                XiN_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n * on["XiN"]
                XiD_rec = self.params.beta * self.params.theta * lam_ratio * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                theta_term = self.params.theta * on["zeta"] * pi_aux_n
                H = on["lam"] / pi_aux_n.pow(1.0 / self.params.eps)

                termN = self.params.beta * self.params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * pi_aux_n * on["XiN"]
                termD = self.params.beta * self.params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * pi_aux_n.pow((self.params.eps - 1.0) / self.params.eps) * on["XiD"]
                return torch.stack([XiN_rec, XiD_rec, termN, termD, theta_term, H], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_XiN = Et_all[..., 0]
            Et_XiD = Et_all[..., 1]
            Et_termN = Et_all[..., 2]
            Et_termD = Et_all[..., 3]
            Et_theta = Et_all[..., 4]
            Et_H = Et_all[..., 5]

            res = residuals_a3_zlb(
                self.params,
                st,
                out,
                Et_XiN,
                Et_XiD,
                Et_termN,
                Et_termD,
                Et_theta,
                Et_H,
                eps_cc=0.0,
            )
            return stack_residuals(res, self.res_keys)

        raise ValueError(self.policy)

    # --------------------
    # Training with two phases (network fixed)
    # --------------------
    def train(
        self,
        commitment_sss: Dict | None = None,
        *,
        n_path: int | None = None,
        n_paths_per_step: int | None = None,
    ) -> List[float]:
        """
        Two-phase DEQN training:
          phase1: (gh_n, batch, lr, steps)
          phase2: (gh_n, batch, lr, steps)
        Network architecture is fixed across phases (per config contract).
        """
        dev = self.params.device


        base_npps = int(n_paths_per_step) if n_paths_per_step is not None else int(getattr(self.cfg, "n_paths_per_step", 1))
        if base_npps < 1:
            base_npps = 1

        def _phase_npps(phase_idx: int) -> int:
            # Explicit train(...) argument overrides config and applies to both phases.
            if n_paths_per_step is not None:
                return base_npps
            if phase_idx == 1:
                v = getattr(self.cfg, "n_paths_per_step_phase1", None)
            else:
                v = getattr(self.cfg, "n_paths_per_step_phase2", None)
            if v is None:
                return base_npps
            vv = int(v)
            return vv if vv >= 1 else 1

        t0 = time.time()
        log_every = int(getattr(self.cfg, "log_every", 50))
        log_every = int(getattr(self.cfg, "log_every", 50))

        # Resolve run_dir. If not provided, create one automatically under ../artifacts.
        # This keeps training reproducible with zero manual setup.
        _rd = getattr(self.cfg, "run_dir", None)
        if _rd is None or str(_rd).strip() == "":
            cfg_artifacts_root = getattr(self.cfg, "artifacts_root", None)
            if cfg_artifacts_root is not None and str(cfg_artifacts_root).strip() != "":
                artifacts_root = _normalize_artifacts_root(str(cfg_artifacts_root))
            else:
                artifacts_root = _normalize_artifacts_root(os.environ.get("DEQN_ARTIFACTS_ROOT", "../artifacts"))
            run_tag = getattr(self.cfg, "tag", None)
            if run_tag is None or str(run_tag).strip() == "":
                run_tag = getattr(self.cfg, "mode", "run")
            _rd = make_run_dir(
                artifacts_root,
                self.policy,
                tag=str(run_tag),
                seed=getattr(self.cfg, "seed", None),
            )
            try:
                setattr(self.cfg, "run_dir", _rd)
            except Exception:
                pass
            try:
                cfg_blob = pack_config(self.params, self.cfg, extra={"policy": self.policy})
                cfg_blob["policy"] = self.policy
                save_run_metadata(
                    _rd,
                    config=cfg_blob,
                )
            except Exception:
                pass

        run_dir: str | None = str(_rd) if _rd else None
        if run_dir is not None:
            ensure_dir(run_dir)
            # Always refresh run metadata from the effective trainer params/cfg.
            # This keeps config.json aligned even when run_dir was pre-created in notebooks.
            try:
                cfg_blob = pack_config(self.params, self.cfg, extra={"policy": self.policy})
                cfg_blob["policy"] = self.policy
                save_run_metadata(run_dir, config=cfg_blob)
            except Exception:
                pass
        metrics_rows: List[Dict[str, float]] | None = [] if run_dir is not None else None
        global_step = 0
        global_best_loss = float("inf")
        global_best_step = -1
        global_best_state: Dict[str, torch.Tensor] | None = None
        weights_selection = str(getattr(self.cfg, "weights_selection", "best")).strip().lower()
        if weights_selection not in ("best", "last"):
            weights_selection = "best"
        self._commitment_sss_for_init = commitment_sss if self.policy in ("commitment", "commitment_zlb") else None

        def run_stage(
            *,
            steps: int,
            lr: float,
            batch_size: int,
            minibatch_size: int,
            x_init: torch.Tensor,
            tag: str,
            eps_stop: float | None,
        ) -> Tuple[List[float], torch.Tensor]:
            """Train for one phase using the DEQN algorithm as described in Appendix B.

            At each optimizer step:
              1) Simulate a path of length N_path (without gradients).
              2) Evaluate equilibrium residuals on all states visited on the path.
              3) Minimize the mean squared residuals via Adam.

            Stopping rule (paper): iterate until loss < eps_stop.
            We also cap iterations at `steps` for safety/reproducibility.
            """
            nonlocal global_step
            opt = optim.Adam(self.net.parameters(), lr=float(lr))

            # Appendix B uses a simulated path of length N_path_length (notation in the paper).
            N_path = int(n_path) if n_path is not None else int(getattr(self.cfg, "n_path", 200))
            train_mode = str(getattr(self.cfg, "training_mode", "author")).strip().lower()
            if train_mode not in ("author", "ours"):
                train_mode = "author"

            sim_batch_size = int(batch_size)
            if train_mode == "author":
                cfg_sim_batch = getattr(self.cfg, "author_n_sim_batch", None)
                if cfg_sim_batch is not None and int(cfg_sim_batch) > 0:
                    sim_batch_size = int(cfg_sim_batch)

            # Ensure population size matches this phase's simulation batch size.
            if x_init is None or int(x_init.shape[0]) != int(sim_batch_size):
                x_pop = self.simulate_initial_state(
                    int(sim_batch_size),
                    commitment_sss=commitment_sss if self.policy in ("commitment", "commitment_zlb") else None,
                ).to(device=dev, dtype=self.params.dtype)
            else:
                x_pop = x_init

            losses: List[float] = []
            best_loss = float("inf")
            best_step = -1
            best_state: Dict[str, torch.Tensor] | None = None
            strict_eps = bool(getattr(self.cfg, "strict_eps_stop", False)) and (train_mode == "ours")
            max_steps_default = int(steps)
            max_steps_safety = getattr(self.cfg, "strict_eps_max_steps", None)
            if strict_eps and eps_stop is not None:
                if max_steps_safety is None:
                    # Keep a large but finite safety cap.
                    max_steps = max(max_steps_default, 20 * max_steps_default)
                else:
                    max_steps = int(max_steps_safety)
            else:
                max_steps = max_steps_default
            max_steps = max(0, int(max_steps))
            hit_safety_cap = eps_stop is not None
            use_author_lr_scheduler = bool(getattr(self.cfg, "use_author_lr_scheduler", False)) and (train_mode == "author")
            author_lr_decay = float(getattr(self.cfg, "author_lr_decay", 1.0))
            author_lr_min = float(getattr(self.cfg, "author_lr_min", 1e-7))
            author_lr_warmup_episodes = max(0, int(getattr(self.cfg, "author_lr_warmup_episodes", 0)))
            keys = self.res_keys
            current_lr = float(lr)
            current_episode_idx = 0

            def _set_opt_lr(v: float) -> None:
                vv = float(v)
                for pg in opt.param_groups:
                    pg["lr"] = vv

            def _author_sched_lr(ep_idx_zero_based: int) -> float:
                if author_lr_warmup_episodes > 0 and ep_idx_zero_based < author_lr_warmup_episodes:
                    frac = float(ep_idx_zero_based + 1) / float(author_lr_warmup_episodes)
                    return max(author_lr_min, float(lr) * frac)
                k = max(0, ep_idx_zero_based - author_lr_warmup_episodes)
                return max(author_lr_min, float(lr) * (author_lr_decay ** k))

            def _opt_step(X_step: torch.Tensor, need_log: bool) -> Tuple[float, torch.Tensor | None, torch.Tensor | None]:
                opt.zero_grad(set_to_none=True)

                # CUDA memory guard: evaluate residual loss in micro-batches.
                chunk_size = int(getattr(self.cfg, "residual_chunk_size", 0) or 0)
                if chunk_size <= 0 and str(dev).startswith("cuda"):
                    if self.policy in ("discretion", "discretion_zlb"):
                        chunk_size = min(int(X_step.shape[0]), 1024)
                    elif self.policy in ("commitment", "commitment_zlb") and self.params.dtype == torch.float64:
                        chunk_size = min(int(X_step.shape[0]), 512)
                use_chunks = (0 < chunk_size < int(X_step.shape[0]))

                resid_for_log: torch.Tensor | None = None
                X_for_log: torch.Tensor | None = None
                if not use_chunks:
                    resid = self._residuals(X_step)
                    loss = self._training_loss_from_states(X_step, resid)
                    lv = float(loss.detach().cpu())
                    loss.backward()
                    if need_log:
                        resid_for_log = resid.detach()
                        X_for_log = X_step.detach()
                else:
                    n_total = int(X_step.shape[0])
                    lv = 0.0
                    resid_chunks = [] if need_log else None
                    x_chunks = [] if need_log else None
                    for j in range(0, n_total, int(chunk_size)):
                        Xi = X_step[j : j + int(chunk_size)]
                        resid_i = self._residuals(Xi)
                        loss_i = self._training_loss_from_states(Xi, resid_i)
                        w = float(Xi.shape[0]) / float(n_total)
                        (loss_i * w).backward()
                        lv += float(loss_i.detach().cpu()) * w
                        if need_log:
                            with torch.no_grad():
                                assert resid_chunks is not None and x_chunks is not None
                                resid_chunks.append(resid_i.detach().cpu())
                                x_chunks.append(Xi.detach().cpu())
                        del Xi, resid_i, loss_i
                    if need_log:
                        assert resid_chunks is not None and x_chunks is not None
                        resid_for_log = torch.cat(resid_chunks, dim=0)
                        X_for_log = torch.cat(x_chunks, dim=0)

                if self.cfg.grad_clip is not None and float(self.cfg.grad_clip) > 0:
                    clip_v = float(self.cfg.grad_clip)
                    clip_mode = str(getattr(self.cfg, "grad_clip_mode", "norm")).strip().lower()
                    if clip_mode == "value":
                        torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_v)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_v)
                opt.step()
                return lv, resid_for_log, X_for_log

            def _after_step(lv: float, need_log: bool, resid_for_log: torch.Tensor | None, X_for_log: torch.Tensor | None) -> bool:
                nonlocal global_step, best_loss, best_step, best_state, hit_safety_cap
                nonlocal global_best_loss, global_best_step, global_best_state
                losses.append(lv)
                global_step += 1

                if lv < best_loss:
                    best_loss = lv
                    best_step = global_step
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                if lv < global_best_loss:
                    global_best_loss = lv
                    global_best_step = global_step
                    global_best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

                if need_log and (resid_for_log is not None) and (X_for_log is not None):
                    with torch.no_grad():
                        metr = residual_metrics(resid_for_log, keys, tol=1e-4)
                        metr.update(residual_metrics_by_regime(X_for_log, resid_for_log, keys, tol=1e-4, policy=self.policy))
                        metr.update(
                            {
                                "global_step": float(global_step),
                                "episode": float(current_episode_idx),
                                "lr": float(current_lr),
                            }
                        )
                        try:
                            rms = float(metr.get("rms", float("nan")))
                            mx = float(metr.get("max", float("nan")))
                            share = float(metr.get("share_all_lt_tol", float("nan")))
                        except Exception:
                            rms, mx, share = float("nan"), float("nan"), float("nan")
                        print(
                            f"[{self.policy} | {tag}] step={global_step:>6d} "
                            f"loss={lv:.3e}  rms={rms:.3e}  max={mx:.3e}  share(|res|<1e-4)={share:.3f}"
                        )
                        if metrics_rows is not None:
                            metrics_rows.append(metr)
                            if (len(metrics_rows) % 20) == 0:
                                save_csv(os.path.join(run_dir, "train_metrics.csv"), pd.DataFrame(metrics_rows))

                if eps_stop is not None and lv < float(eps_stop):
                    hit_safety_cap = False
                    return True
                if eps_stop is None:
                    hit_safety_cap = False
                return False

            # Author-like training loop: simulate one episode, shuffle states, optimize in minibatches.
            mb = max(1, int(minibatch_size))
            n_states_episode = max(1, int(N_path) * int(sim_batch_size))
            updates_per_episode = max(1, (n_states_episode + mb - 1) // mb)
            episodes_cfg = getattr(self.cfg, "n_episodes", None)
            explicit_author_episodes = int(episodes_cfg) if episodes_cfg is not None else None
            if explicit_author_episodes is not None and explicit_author_episodes <= 0:
                explicit_author_episodes = None

            if train_mode == "author":
                if explicit_author_episodes is not None:
                    max_episodes = int(explicit_author_episodes)
                    # Author Main.py uses episode count as the true stop criterion.
                    # Keep steps only as an optional explicit safety guard.
                    author_step_cap = getattr(self.cfg, "author_step_cap", None)
                    if author_step_cap is None:
                        max_steps = 0
                    else:
                        max_steps = max(0, int(author_step_cap))
                else:
                    # Backward compatibility: derive episode budget from phase step budget.
                    max_episodes = max(1, (int(max_steps) + updates_per_episode - 1) // updates_per_episode)
            else:
                max_episodes = None

            if train_mode == "author" and max_episodes is not None:
                pbar_total = int(max_episodes) * int(updates_per_episode)
            elif int(max_steps) > 0:
                pbar_total = int(max_steps)
            else:
                pbar_total = 1
            pbar = trange(max(1, pbar_total), desc=f"{self.policy} | train | {tag} | {x_pop.dtype}", leave=False)
            use_limited_shuffle = bool(getattr(self.cfg, "author_limited_shuffle", True)) and (train_mode == "author")

            def _limited_shuffle_indices(n_total: int, buffer_size: int, device: torch.device) -> torch.Tensor:
                n = int(n_total)
                k = max(1, int(buffer_size))
                if n <= 1:
                    return torch.arange(n, device=device, dtype=torch.long)
                if k >= n:
                    return torch.randperm(n, device=device)

                buf = list(range(k))
                out_idx: List[int] = []
                for i_in in range(k, n):
                    j = random.randrange(0, len(buf))
                    out_idx.append(buf[j])
                    buf[j] = i_in
                random.shuffle(buf)
                out_idx.extend(buf)
                return torch.tensor(out_idx, device=device, dtype=torch.long)

            try:
                episode_count = 0
                reached_step_cap = False
                if train_mode == "author" and bool(getattr(self.cfg, "author_initialize_each_episode", True)):
                    x_pop = self._author_post_init_state(x_pop, episode_idx=0)
                while True:
                    if train_mode == "author":
                        if max_episodes is None or episode_count >= int(max_episodes):
                            break
                    else:
                        if global_step >= int(max_steps):
                            break

                    # Author Hooks.py-style per-episode scheduler.
                    if use_author_lr_scheduler:
                        current_lr = _author_sched_lr(episode_count)
                    else:
                        current_lr = float(lr)
                    _set_opt_lr(current_lr)
                    current_episode_idx = int(episode_count + 1)

                    with torch.no_grad():
                        xs = []
                        cur = x_pop
                        for __ in range(N_path):
                            xs.append(cur)
                            cur = self._step_state(cur)
                        x_pop = cur
                        X = torch.cat(xs, dim=0)
                        if use_limited_shuffle:
                            eff = int(X.shape[0])
                            # Author Main.py: shuffle(buffer_size = effective_size / minibatch_size)
                            buf = max(1, eff // max(1, int(mb)))
                            perm = _limited_shuffle_indices(eff, buf, X.device)
                        else:
                            perm = torch.randperm(int(X.shape[0]), device=X.device)
                        X = X.index_select(0, perm)
                    episode_count += 1

                    stop_now = False
                    for j in range(0, int(X.shape[0]), mb):
                        if int(max_steps) > 0 and global_step >= int(max_steps):
                            reached_step_cap = True
                            break
                        Xi = X[j : j + mb]
                        # Author Main.py batches use drop_remainder=True.
                        if train_mode == "author" and int(Xi.shape[0]) < int(mb):
                            continue
                        need_log = (global_step % log_every) == 0
                        lv, resid_for_log, X_for_log = _opt_step(Xi, need_log)
                        stop_now = _after_step(lv, need_log, resid_for_log, X_for_log)
                        pbar.update(1)
                        if stop_now:
                            break
                    if train_mode == "author" and bool(getattr(self.cfg, "author_initialize_each_episode", True)):
                        x_pop = self._author_post_init_state(x_pop, episode_idx=episode_count)
                    if stop_now or reached_step_cap:
                        break
            finally:
                pbar.close()

            # Optional phase-wise best restore (kept for "best" checkpoint mode).
            if weights_selection == "best" and best_state is not None:
                try:
                    self.net.load_state_dict(best_state, strict=True)
                except Exception:
                    pass

            if strict_eps and eps_stop is not None and hit_safety_cap:
                print(
                    f"[{self.policy} | {tag}] reached safety cap (max_steps={max_steps}) "
                    f"before hitting eps_stop={float(eps_stop):.3e}."
                )
            if train_mode == "author":
                if explicit_author_episodes is not None:
                    print(
                        f"[{self.policy} | {tag}] author episodes completed: "
                        f"{episode_count}/{int(max_episodes)}"
                    )
                else:
                    print(
                        f"[{self.policy} | {tag}] author episodes (derived): "
                        f"{episode_count} from step budget={int(max_steps)}"
                    )
                if reached_step_cap and explicit_author_episodes is not None:
                    print(
                        f"[{self.policy} | {tag}] WARNING: hit step safety cap "
                        f"(max_steps={int(max_steps)}) before finishing episodes={int(max_episodes)}."
                    )
            print(f"[{self.policy} | {tag}] best_loss={best_loss:.3e} at step={best_step}")
            return losses, x_pop


        # init population
        npps_p1 = _phase_npps(1)
        x = self.simulate_initial_state(
            int(self.cfg.phase1.batch_size) * npps_p1,
            commitment_sss=commitment_sss if self.policy in ("commitment", "commitment_zlb") else None,
        ).to(device=dev, dtype=self.params.dtype)

        losses_all: List[float] = []

        for pidx, phase in enumerate(self.cfg.phases(), start=1):
            # Switch GH for this phase
            self._set_gh(int(phase.gh_n_train))

            # Optional dtype switch (compute-only)
            if bool(getattr(phase, "use_float64", False)) and self.params.dtype != torch.float64:
                self.params = ModelParams(
                    beta=self.params.beta, gamma=self.params.gamma, omega=self.params.omega,
                    theta=self.params.theta, eps=self.params.eps, tau_bar=self.params.tau_bar,
                    rho_A=self.params.rho_A, rho_tau=self.params.rho_tau, rho_g=self.params.rho_g,
                    sigma_A=self.params.sigma_A, sigma_tau=self.params.sigma_tau, sigma_g=self.params.sigma_g,
                    g_bar=self.params.g_bar, eta_bar=self.params.eta_bar,
                    bad_state=self.params.bad_state,
                    p12=self.params.p12, p21=self.params.p21, p21_l=self.params.p21_l, p21_u=self.params.p21_u,
                    pi_bar=self.params.pi_bar, psi=self.params.psi, rho_i=self.params.rho_i,
                    device=self.params.device, dtype=torch.float64
                ).to_torch()
                self.net = self.net.to(dev).double()
                x = x.to(device=dev, dtype=torch.float64)
                self._set_gh(int(phase.gh_n_train))
                # Ensure quadrature grids and inputs follow dtype switch (single -> double)
                assert self._eps_grid is None or self._eps_grid.dtype == self.params.dtype
                assert self._w_grid is None or self._w_grid.dtype == self.params.dtype

            npps_phase = _phase_npps(pidx)
            train_mode_here = str(getattr(self.cfg, "training_mode", "author")).strip().lower()
            if train_mode_here == "author" and getattr(self.cfg, "author_n_sim_batch", None) is not None:
                b_tag = int(getattr(self.cfg, "author_n_sim_batch"))
            else:
                b_tag = int(phase.batch_size) * npps_phase
            tag = f"phase{pidx}(gh={int(phase.gh_n_train)},B={b_tag})"
            lp, x = run_stage(
                steps=int(phase.steps),
                lr=float(phase.lr),
                batch_size=int(phase.batch_size) * npps_phase,
                minibatch_size=int(phase.batch_size),
                x_init=x,
                tag=tag,
                eps_stop=getattr(phase, "eps_stop", None),
            )
            losses_all.extend(lp)

        # Final checkpoint selection and persistence:
        # - weights_last.pt: final iterate
        # - weights_best.pt: minimum training loss iterate
        # - weights.pt (or cfg.best_weights_name): canonical file according to cfg.weights_selection
        state_last = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        selected_state = state_last
        selected_label = "last"
        if weights_selection == "best":
            if global_best_state is not None:
                selected_state = global_best_state
                selected_label = "best"
            else:
                print(f"[{self.policy}] WARNING: weights_selection='best' but best snapshot is unavailable; using last.")
        try:
            self.net.load_state_dict(selected_state, strict=True)
        except Exception:
            pass

        if run_dir is not None:
            try:
                save_torch(os.path.join(run_dir, "weights_last.pt"), state_last)
            except Exception:
                pass
            if global_best_state is not None:
                try:
                    save_torch(os.path.join(run_dir, "weights_best.pt"), global_best_state)
                except Exception:
                    pass
            canonical_name = str(getattr(self.cfg, "best_weights_name", "weights.pt") or "weights.pt")
            try:
                save_torch(os.path.join(run_dir, canonical_name), selected_state)
                if canonical_name != "weights.pt":
                    save_torch(os.path.join(run_dir, "weights.pt"), selected_state)
            except Exception:
                pass
        print(
            f"[{self.policy}] checkpoint selection: {selected_label} "
            f"(global_best={global_best_loss:.3e} at step={global_best_step})"
        )

        # Save training diagnostics (quality monitoring)
        if metrics_rows is not None and run_dir is not None:
            try:
                df = pd.DataFrame(metrics_rows)
                self.train_log_df = df
                save_csv(os.path.join(run_dir, "train_metrics.csv"), df)
            except Exception:
                # Logging must never crash training
                pass

        # Final flush + pointers
        if run_dir is not None:
            try:
                save_csv(
                    os.path.join(run_dir, "train_metrics.csv"),
                    pd.DataFrame(metrics_rows or []),
                )
            except Exception:
                pass
            # Author post_process starts from Parameters.starting_state[0].
            # Persist the final author-loop starting state for reproducible post-processing.
            try:
                save_torch(os.path.join(run_dir, "starting_state.pt"), x.detach().cpu())
            except Exception:
                pass
            print(f"[{self.policy}] training artifacts in: {run_dir}")
            print(f"[{self.policy}] metrics: {os.path.join(run_dir, 'train_metrics.csv')}")
            print(f"[{self.policy}] next: run post-training residual check via scripts/trajectory_quality_report.py (uses existing sanity_checks).")
        return losses_all


@torch.inference_mode()
def simulate_paths(
    params: ModelParams,
    policy: PolicyName,
    net: PolicyNetwork,
    T: int,
    burn_in: int,
    x0: torch.Tensor,
    rbar_by_regime: torch.Tensor | None = None,
    *,
    compute_implied_i: bool = False,
    gh_n: int = 3,
    thin: int = 1,
    store_states: bool = False,
    show_progress: bool = False,
    force_regime: int | None = None,
    force_logA: float | None = None,
    force_loggtilde: float | None = None,
    force_xi: float | None = None,
) -> Dict[str, np.ndarray]:
    """Forward simulation under a trained policy network."""

    # For discretion/discretion_zlb/commitment, nominal rate is implied by Euler unless requested otherwise.
    if (policy in ("discretion", "discretion_zlb", "commitment")) and (not compute_implied_i):
        print("[simulate_paths] compute_implied_i was False for policy=%s; enabling it to compute nominal rates from Euler." % policy)
        compute_implied_i = True

    # Keep params/net dtype consistent. Training may end in float64 (phase2), while callers
    # can still pass float32 params from notebook setup.
    net_dtype = next(net.parameters()).dtype
    params_sim = params
    if net_dtype != params.dtype:
        params_sim = ModelParams(
            beta=params.beta, gamma=params.gamma, omega=params.omega,
            theta=params.theta, eps=params.eps, tau_bar=params.tau_bar,
            rho_A=params.rho_A, rho_tau=params.rho_tau, rho_g=params.rho_g,
            sigma_A=params.sigma_A, sigma_tau=params.sigma_tau, sigma_g=params.sigma_g,
            g_bar=params.g_bar, eta_bar=params.eta_bar,
            bad_state=params.bad_state,
            p12=params.p12, p21=params.p21, p21_l=params.p21_l, p21_u=params.p21_u,
            pi_bar=params.pi_bar, psi=params.psi, rho_i=params.rho_i,
            device=params.device, dtype=net_dtype,
        ).to_torch()
        if rbar_by_regime is not None:
            rbar_by_regime = rbar_by_regime.to(device=params_sim.device, dtype=params_sim.dtype)

    dev, dt = params_sim.device, params_sim.dtype
    net.eval()
    B = x0.shape[0]
    x = x0.to(device=dev, dtype=dt)

    # Build a minimal cfg for simulation (no training). For commitment_zlb we need
    # author-style p12 override parity (1/28) used in the public code.
    if policy == "commitment_zlb":
        cfg_sim = TrainConfig.author_like(
            policy=policy,
            seed=0,
            cpu_num_threads=None,
            cpu_num_interop_threads=None,
        )
    elif params_sim.device == "cpu":
        cfg_sim = TrainConfig.dev(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    else:
        cfg_sim = TrainConfig.full(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    trainer = Trainer(params=params_sim, cfg=cfg_sim, policy=policy, net=net, gh_n=int(gh_n), rbar_by_regime=rbar_by_regime)

    assert thin >= 1

    keep = (T - burn_in + thin - 1) // thin
    store: Dict[str, np.ndarray] = {
        "c": np.zeros((keep, B)),
        "pi": np.zeros((keep, B)),
        "pstar": np.zeros((keep, B)),
        "Delta": np.zeros((keep, B)),
        "y": np.zeros((keep, B)),
        "h": np.zeros((keep, B)),
        "g": np.zeros((keep, B)),
        "A": np.zeros((keep, B)),
        "tau": np.zeros((keep, B)),
        "s": np.zeros((keep, B), dtype=np.int64),
    }
    explicit_i_policies = ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "commitment_zlb")
    if (policy in explicit_i_policies) or compute_implied_i:
        store["i"] = np.zeros((keep, B))
    if store_states:
        store["logA"] = np.zeros((keep, B))
        store["loggtilde"] = np.zeros((keep, B))
        store["xi"] = np.zeros((keep, B))

    if policy in ("commitment", "commitment_zlb"):
        store["vartheta_prev"] = np.zeros((keep, B))
        store["varrho_prev"] = np.zeros((keep, B))
    if policy == "commitment_zlb":
        store["i_nom_prev"] = np.zeros((keep, B))
        store["varphi_prev"] = np.zeros((keep, B))

    k = 0
    if force_regime is not None:
        fr = int(force_regime)
        if fr not in (0, 1):
            raise ValueError(f"force_regime must be 0/1 or None, got {force_regime!r}")
    iterator = trange(T, desc=f"simulate[{policy}]", leave=False) if show_progress else range(T)
    for t in iterator:
        out = trainer._policy_outputs(x)
        st = unpack_state(x, policy)
        ids = identities(params_sim, st, out)

        if policy in explicit_i_policies:
            if "i_nom" not in out:
                raise RuntimeError(f"policy={policy} expected explicit i_nom in decoded outputs.")
            i_t = out["i_nom"]
        else:
            if compute_implied_i:
                i_t = implied_nominal_rate_from_euler(params_sim, policy, x, out, int(gh_n), trainer)
            else:
                i_t = torch.full_like(out["pi"], float("nan"))

        if (t >= burn_in) and ((t - burn_in) % thin == 0):
            store["c"][k] = out["c"].cpu().numpy()
            store["pi"][k] = out["pi"].cpu().numpy()
            store["pstar"][k] = out["pstar"].cpu().numpy()
            store["Delta"][k] = out["Delta"].cpu().numpy()
            store["y"][k] = ids["y"].cpu().numpy()
            store["h"][k] = ids["h"].cpu().numpy()
            store["g"][k] = ids["g"].cpu().numpy()
            store["A"][k] = ids["A"].cpu().numpy()
            store["tau"][k] = (ids["one_plus_tau"] - 1.0).cpu().numpy()
            store["s"][k] = st.s.cpu().numpy()
            if "i" in store:
                store["i"][k] = i_t.cpu().numpy()
            if store_states:
                store["logA"][k] = st.logA.cpu().numpy()
                store["loggtilde"][k] = st.loggtilde.cpu().numpy()
                store["xi"][k] = st.xi.cpu().numpy()
            if policy in ("commitment", "commitment_zlb"):
                store["vartheta_prev"][k] = st.vartheta_prev.cpu().numpy()
                store["varrho_prev"][k] = st.varrho_prev.cpu().numpy()
            if policy == "commitment_zlb":
                store["i_nom_prev"][k] = st.i_nom_prev.cpu().numpy()
                store["varphi_prev"][k] = st.varphi_prev.cpu().numpy()
            k += 1

        if force_regime is None:
            x = trainer._step_state(x)
        else:
            # Author-like fixed-regime branches used in post-processing (NT/SS).
            Bcur = x.shape[0]
            epsA = torch.randn(Bcur, device=dev, dtype=dt)
            epsg = torch.randn(Bcur, device=dev, dtype=dt)
            epst = torch.randn(Bcur, device=dev, dtype=dt)
            s_next = torch.full_like(st.s, int(force_regime))
            logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params_sim, st, epsA, epsg, epst, s_next)
            if policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "discretion", "discretion_zlb"):
                x = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
            elif policy == "taylor_para":
                if bool(getattr(trainer, "_taylor_para_has_extended_state", False)):
                    p21_prev = st.p21 if st.p21 is not None else torch.full_like(out["Delta"], float(params_sim.p21))
                    x = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["i_nom"], p21_prev], dim=-1)
                else:
                    x = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
            elif policy == "commitment":
                if st.c_prev is not None or bool(getattr(trainer, "_commitment_has_c_prev", False)):
                    x = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"], out["c"]], dim=-1)
                else:
                    x = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"]], dim=-1)
            elif policy == "commitment_zlb":
                x = torch.stack(
                    [
                        out["Delta"],
                        logA_n,
                        logg_n,
                        xi_n,
                        s_n.to(dt),
                        out["vartheta"],
                        out["varrho"],
                        out["c"],
                        out["i_nom"],
                        out["varphi"],
                    ],
                    dim=-1,
                )
            else:
                raise ValueError(f"Unsupported policy for simulate_paths: {policy}")

        # Author post_process branch controls: optionally pin selected exogenous states.
        if force_logA is not None:
            x[:, 1] = float(force_logA)
        if force_loggtilde is not None:
            x[:, 2] = float(force_loggtilde)
        if force_xi is not None:
            x[:, 3] = float(force_xi)

    return store
