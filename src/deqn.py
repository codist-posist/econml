from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import os

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
from .policy_rules import i_taylor, i_modified_taylor, fisher_euler_term
from .residuals_a1 import residuals_a1
from .residuals_a2 import residuals_a2
from .residuals_a3 import residuals_a3
from .io_utils import save_csv, ensure_dir, make_run_dir, save_run_metadata, _normalize_artifacts_root, pack_config


# Torch thread pools can only be safely configured once per process.
_CPU_THREAD_KNOBS_APPLIED = False


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: Tuple[int, int] = (512, 512),
        activation: str = "selu",
    ):
        super().__init__()
        act = nn.SELU if activation.lower() == "selu" else nn.ReLU
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), act(),
            nn.Linear(h1, h2), act(),
            nn.Linear(h2, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stack_residuals(res: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.stack([res[k] for k in keys], dim=-1)


def loss_from_residuals(resid: torch.Tensor) -> torch.Tensor:
    return (resid ** 2).mean()



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

    P = params.P
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
    pi = torch.stack([P[st.s, 0], P[st.s, 1]], dim=-1).view(B, 1, 2)  # (B,1,2)

    if val.ndim == 3:
        return (val * wq * pi).sum(dim=1).sum(dim=-1)  # (B,)
    if val.ndim == 4:
        wq4 = w_grid.view(1, Q, 1, 1)
        pi4 = torch.stack([P[st.s, 0], P[st.s, 1]], dim=-1).view(B, 1, 2, 1)
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
            vp_cur = (out_t["vartheta"] * out_t["c"].pow(params.gamma)).view(-1, 1, 1).expand_as(logA_n)
            rp_cur = (out_t["varrho"] * out_t["c"].pow(params.gamma)).view(-1, 1, 1).expand_as(logA_n)
            return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x_t.dtype), vp_cur, rp_cur], dim=-1)
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
        set_seeds(self.cfg.seed)

        if self.policy == "mod_taylor":
            if self.rbar_by_regime is None:
                raise ValueError("mod_taylor requires rbar_by_regime from export_rbar_tensor(solve_flexprice_sss(params))")
            if self.rbar_by_regime.numel() != 2:
                raise ValueError("rbar_by_regime must have 2 elements (one per regime)")

        if self.policy in ["taylor", "mod_taylor"]:
            self.res_keys = ["res_c_lam", "res_labor", "res_euler", "res_XiN", "res_XiD", "res_pstar_def", "res_calvo", "res_Delta"]
        elif self.policy == "discretion":
            self.res_keys = ["res_c_foc", "res_pi_foc", "res_pstar_foc", "res_Delta_foc",
                             "res_c_lam", "res_labor", "res_XiN_rec", "res_XiD_rec",
                             "res_pstar_def", "res_calvo", "res_Delta_law"]
        elif self.policy == "commitment":
            self.res_keys = ["res_c_foc", "res_Delta_foc", "res_pi_foc", "res_pstar_foc", "res_XiN_foc", "res_XiD_foc",
                             "res_c_lam", "res_labor", "res_XiN_rec", "res_XiD_rec",
                             "res_pstar_def", "res_calvo", "res_Delta_law"]
        else:
            raise ValueError(self.policy)

        self.eq_labels = list(self.res_keys)

        # timers
        self._t_accum = {}
        self.last_train_stats = None        # initialize GH grid to phase1 by default unless user overrides gh_n
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

        # Initialize exogenous states consistent with the model's laws of motion (Appendix B).
        # logA and logg are drift-corrected log-AR(1) processes; xi is AR(1) around zero.
        # Draw from the implied stationary distributions to start training samples in the
        # ergodic region (up to burn-in), avoiding off-model transients.
        rho_A, sig_A = float(self.params.rho_A), float(self.params.sigma_A)
        rho_g, sig_g = float(self.params.rho_g), float(self.params.sigma_g)
        rho_xi, sig_xi = float(self.params.rho_tau), float(self.params.sigma_tau)

        mu_logA = -(sig_A**2) / (2.0 * (1.0 - rho_A**2)) if abs(rho_A) < 1.0 else 0.0
        mu_logg = -(sig_g**2) / (2.0 * (1.0 - rho_g**2)) if abs(rho_g) < 1.0 else 0.0
        sd_logA = sig_A / max(1e-12, (1.0 - rho_A**2))**0.5 if abs(rho_A) < 1.0 else sig_A
        sd_logg = sig_g / max(1e-12, (1.0 - rho_g**2))**0.5 if abs(rho_g) < 1.0 else sig_g
        sd_xi   = sig_xi / max(1e-12, (1.0 - rho_xi**2))**0.5 if abs(rho_xi) < 1.0 else sig_xi

        logA = torch.tensor(mu_logA, device=dev, dtype=dt) + sd_logA * torch.randn(B, device=dev, dtype=dt)
        logg = torch.tensor(mu_logg, device=dev, dtype=dt) + sd_logg * torch.randn(B, device=dev, dtype=dt)
        xi   = sd_xi * torch.randn(B, device=dev, dtype=dt)

        # Markov regime: draw from stationary distribution unless an SSS init is provided.
        p12, p21 = float(self.params.p12), float(self.params.p21)
        denom = p12 + p21
        if denom <= 0.0:
            raise ValueError(f"Invalid Markov switching parameters: p12+p21 must be > 0. Got p12={p12}, p21={p21}.")
        pi_bad = (p12 / denom)  # stationary prob of bad regime
        u = torch.rand(B, device=dev, dtype=dt)
        s = torch.where(u < (1.0 - pi_bad),
                        torch.zeros(B, device=dev, dtype=torch.long),
                        torch.ones(B, device=dev, dtype=torch.long))

        # Backward-looking endogenous state: initialize at undistorted value.
        Delta_prev = torch.ones(B, device=dev, dtype=dt)

        if self.policy != "commitment":
            return torch.stack([Delta_prev, logA, logg, xi, s.to(dt)], dim=-1)

        # Timeless commitment: lagged Ramsey multipliers are part of the state.
        # If caller provides commitment SSS (per-regime or pooled), use it; otherwise start at 0.
        if commitment_sss is not None:
            if isinstance(commitment_sss, dict) and 0 in commitment_sss and 1 in commitment_sss:
                vp0 = float(commitment_sss[0]["vartheta_prev"]); vp1 = float(commitment_sss[1]["vartheta_prev"])
                rp0 = float(commitment_sss[0]["varrho_prev"]);   rp1 = float(commitment_sss[1]["varrho_prev"])
                vp = torch.where(s == 0,
                                 torch.full((B,), vp0, device=dev, dtype=dt),
                                 torch.full((B,), vp1, device=dev, dtype=dt))
                rp = torch.where(s == 0,
                                 torch.full((B,), rp0, device=dev, dtype=dt),
                                 torch.full((B,), rp1, device=dev, dtype=dt))
            else:
                vp = torch.full((B,), float(commitment_sss["vartheta_prev"]), device=dev, dtype=dt)
                rp = torch.full((B,), float(commitment_sss["varrho_prev"]), device=dev, dtype=dt)
        else:
            std = float(getattr(self.cfg, 'commitment_init_multiplier_std', 0.0) or 0.0)
            clip = float(getattr(self.cfg, 'commitment_init_multiplier_clip', 0.0) or 0.0)
            if std > 0.0:
                vp = std * torch.randn(B, device=dev, dtype=dt)
                rp = std * torch.randn(B, device=dev, dtype=dt)
                if clip > 0.0:
                    vp = torch.clamp(vp, -clip, clip)
                    rp = torch.clamp(rp, -clip, clip)
            else:
                vp = torch.zeros(B, device=dev, dtype=dt)
                rp = torch.zeros(B, device=dev, dtype=dt)

        return torch.stack([Delta_prev, logA, logg, xi, s.to(dt), vp, rp], dim=-1)
    def _policy_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # If x was created under torch.inference_mode(), it is an 'inference tensor' and
        # cannot be used in autograd-tracked computations. Clone it when grad is enabled.
        if torch.is_grad_enabled() and getattr(x, 'is_inference', False):
            x = x.clone()
        raw = self.net(x)
        floors = {"c": self.cfg.c_floor, "Delta": self.cfg.delta_floor, "pstar": self.cfg.pstar_floor}
        return decode_outputs(self.policy, raw, floors=floors)

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

        u = torch.rand(B, device=dev, dtype=dt)
        P = self.params.P
        p0 = P[st.s, 0]
        s_next = torch.where(u < p0, torch.zeros_like(st.s), torch.ones_like(st.s))

        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)

        if self.policy == "commitment":
            vp = out["vartheta"] * out["c"].pow(self.params.gamma)
            rp = out["varrho"] * out["c"].pow(self.params.gamma)
            return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), vp, rp], dim=-1)

        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)

    # --------------------
    # Residuals (unchanged economics)
    # --------------------
    def _residuals(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is a normal tensor when computing residuals with autograd.
        if torch.is_grad_enabled() and getattr(x, 'is_inference', False):
            x = x.clone()
        out = self._policy_outputs(x)
        st = unpack_state(x, self.policy)

        if self.policy in ["taylor", "mod_taylor"]:
            lam_t = out["lam"]
            lam_tg = lam_t.view(-1, 1, 1)

            if self.policy == "taylor":
                i_t = i_taylor(self.params, out["pi"])
            else:
                assert self.rbar_by_regime is not None
                i_t = i_modified_taylor(self.params, out["pi"], self.rbar_by_regime, st.s)

            i_tg = i_t.view(-1, 1, 1)

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1)

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)
                Lambda = self.params.beta * on["lam"] / lam_tg

                one_plus_pi = on["one_plus_pi"]
                term_XiN = self.params.theta * Lambda * one_plus_pi.pow(self.params.eps) * on["XiN"]
                term_XiD = self.params.theta * Lambda * one_plus_pi.pow(self.params.eps - 1.0) * on["XiD"]
                term_eul = self.params.beta * ((1.0 + i_tg) / one_plus_pi) * (on["lam"] / lam_tg)
                return torch.stack([term_XiN, term_XiD, term_eul], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_XiN = Et_all[..., 0]
            Et_XiD = Et_all[..., 1]
            Et_eul = Et_all[..., 2]

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

                one_plus_pi = on["one_plus_pi"]
                Lambda = self.params.beta * on["lam"] / lam_tg

                F = self.params.theta * self.params.beta * on["c"].pow(-self.params.gamma) * one_plus_pi.pow(self.params.eps - 1.0) * on["XiD"]
                G = self.params.theta * self.params.beta * on["c"].pow(-self.params.gamma) * one_plus_pi.pow(self.params.eps) * on["XiN"]

                theta_term = self.params.theta * one_plus_pi.pow(self.params.eps) * on["zeta"]
                XiN_rec = self.params.theta * Lambda * one_plus_pi.pow(self.params.eps) * on["XiN"]
                XiD_rec = self.params.theta * Lambda * one_plus_pi.pow(self.params.eps - 1.0) * on["XiD"]

                return torch.stack([F, G, theta_term, XiN_rec, XiD_rec], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_F = Et_all[..., 0]
            Et_G = Et_all[..., 1]
            Et_theta = Et_all[..., 2]
            Et_XiN = Et_all[..., 3]
            Et_XiD = Et_all[..., 4]

            Et_dF = torch.autograd.grad(Et_F.sum(), out["Delta"], create_graph=True)[0]
            Et_dG = torch.autograd.grad(Et_G.sum(), out["Delta"], create_graph=True)[0]

            res = residuals_a2(self.params, st, out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD)
            return stack_residuals(res, self.res_keys)

        if self.policy == "commitment":
            c_t = out["c"]
            lam_t = out["lam"]
            c_tg = c_t.view(-1, 1, 1)
            lam_tg = lam_t.view(-1, 1, 1)
            gamma = self.params.gamma

            def x_next(epsA, epsg, epst, s_next):
                logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)
                Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
                vp_cur = (out["vartheta"] * out["c"].pow(gamma)).view(-1, 1, 1).expand_as(logA_n)
                rp_cur = (out["varrho"] * out["c"].pow(gamma)).view(-1, 1, 1).expand_as(logA_n)
                return torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vp_cur, rp_cur], dim=-1)

            def f_all(epsA, epsg, epst, s_next):
                xn = x_next(epsA, epsg, epst, s_next)
                on = self._policy_outputs(xn)
                one_plus_pi = on["one_plus_pi"]
                Lambda = self.params.beta * on["lam"] / lam_tg

                XiN_rec = self.params.theta * Lambda * one_plus_pi.pow(self.params.eps) * on["XiN"]
                XiD_rec = self.params.theta * Lambda * one_plus_pi.pow(self.params.eps - 1.0) * on["XiD"]
                theta_term = self.params.theta * one_plus_pi.pow(self.params.eps) * on["zeta"]

                termN = self.params.beta * self.params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * one_plus_pi.pow(self.params.eps) * on["XiN"]
                termD = self.params.beta * self.params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * one_plus_pi.pow(self.params.eps - 1.0) * on["XiD"]
                return torch.stack([XiN_rec, XiD_rec, termN, termD, theta_term], dim=-1)

            Et_all = expectation_operator_appendixB(self.params, st, self.gh_n, f_all, eps_grid=self._eps_grid, w_grid=self._w_grid)
            Et_XiN = Et_all[..., 0]
            Et_XiD = Et_all[..., 1]
            Et_termN = Et_all[..., 2]
            Et_termD = Et_all[..., 3]
            Et_theta = Et_all[..., 4]

            res = residuals_a3(self.params, st, out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta)
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


        npps = int(n_paths_per_step) if n_paths_per_step is not None else int(getattr(self.cfg, "n_paths_per_step", 1))

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
        metrics_rows: List[Dict[str, float]] | None = [] if run_dir is not None else None
        global_step = 0
        self._commitment_sss_for_init = commitment_sss if self.policy == "commitment" else None

        def run_stage(*, steps: int, lr: float, batch_size: int, x_init: torch.Tensor, tag: str, eps_stop: float | None) -> Tuple[List[float], torch.Tensor]:
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

            # Ensure population size matches this phase's batch_size (config should be meaningful).
            if x_init is None or int(x_init.shape[0]) != int(batch_size):
                x_pop = self.simulate_initial_state(
                    int(batch_size),
                    commitment_sss=commitment_sss if self.policy == "commitment" else None,
                ).to(device=dev, dtype=self.params.dtype)
            else:
                x_pop = x_init

            losses: List[float] = []

            for _ in trange(int(steps), desc=f"{self.policy} | train | {tag} | {x_pop.dtype}", leave=False):
                # --- 1) Simulate a path (stop-gradient / sampling step) ---
                with torch.no_grad():
                    xs = []
                    cur = x_pop
                    for __ in range(N_path):
                        xs.append(cur)
                        cur = self._step_state(cur)
                    x_pop = cur  # continue from last state next iteration
                    X = torch.cat(xs, dim=0)  # (N_path*B, d)

                # --- 2) Compute residuals on the simulated path ---
                opt.zero_grad(set_to_none=True)
                resid = self._residuals(X)  # (N_path*B, K)
                keys = self.res_keys
                loss = loss_from_residuals(resid)

                # --- 3) Gradient step (Adam) ---
                loss.backward()
                if self.cfg.grad_clip is not None and float(self.cfg.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(self.cfg.grad_clip))
                opt.step()

                lv = float(loss.detach().cpu())
                losses.append(lv)
                global_step += 1

                # Optional logging to disk (unchanged; does not affect equations)
                if (global_step % log_every) == 0:
                    with torch.no_grad():
                        metr = residual_metrics(resid, keys, tol=1e-4)
                        metr.update(residual_metrics_by_regime(X, resid, keys, tol=1e-4, policy=self.policy))
                        metr.update({"global_step": float(global_step)})

                        # Console-friendly training diagnostics (does not affect equations)
                        # Show a compact view of residual quality while training.
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
                                save_csv(
                                    os.path.join(run_dir, "train_metrics.csv"),
                                    pd.DataFrame(metrics_rows),
                                )

                # Paper stopping rule: stop once loss is below epsilon
                if eps_stop is not None and lv < float(eps_stop):
                    break

            return losses, x_pop


        # init population
        x = self.simulate_initial_state(
            int(self.cfg.phase1.batch_size) * npps,
            commitment_sss=commitment_sss if self.policy == "commitment" else None,
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
                    p12=self.params.p12, p21=self.params.p21,
                    pi_bar=self.params.pi_bar, psi=self.params.psi,
                    device=self.params.device, dtype=torch.float64
                ).to_torch()
                self.net = self.net.to(dev).double()
                x = x.to(device=dev, dtype=torch.float64)
                self._set_gh(int(phase.gh_n_train))
                # Ensure quadrature grids and inputs follow dtype switch (single -> double)
                assert self._eps_grid is None or self._eps_grid.dtype == self.params.dtype
                assert self._w_grid is None or self._w_grid.dtype == self.params.dtype

            tag = f"phase{pidx}(gh={int(phase.gh_n_train)},B={int(phase.batch_size)})"
            lp, x = run_stage(
                steps=int(phase.steps),
                lr=float(phase.lr),
                batch_size=int(phase.batch_size) * npps,
                x_init=x,
                tag=tag,
                eps_stop=getattr(phase, "eps_stop", None),
            )
            losses_all.extend(lp)


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
) -> Dict[str, np.ndarray]:
    """Forward simulation under a trained policy network."""

    # Safety: for discretion/commitment, nominal rate i_t is not an explicit control; it is implied by the Euler equation. If compute_implied_i is left False, downstream Table-2 moments for nominal/real rates will be invalid.
    if (policy in ("discretion", "commitment")) and (not compute_implied_i):
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
            p12=params.p12, p21=params.p21,
            pi_bar=params.pi_bar, psi=params.psi,
            device=params.device, dtype=net_dtype,
        ).to_torch()
        if rbar_by_regime is not None:
            rbar_by_regime = rbar_by_regime.to(device=params_sim.device, dtype=params_sim.dtype)

    dev, dt = params_sim.device, params_sim.dtype
    net.eval()
    B = x0.shape[0]
    x = x0.to(device=dev, dtype=dt)

    # Build a minimal cfg for simulation (no training). Must be compatible with new TrainConfig.
    if params_sim.device == "cpu":
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
    if (policy not in ["taylor", "mod_taylor"]) and compute_implied_i:
        store["i"] = np.zeros((keep, B))
    elif policy in ["taylor", "mod_taylor"]:
        store["i"] = np.zeros((keep, B))
    if store_states:
        store["logA"] = np.zeros((keep, B))
        store["loggtilde"] = np.zeros((keep, B))
        store["xi"] = np.zeros((keep, B))

    if policy == "commitment":
        store["vartheta_prev"] = np.zeros((keep, B))
        store["varrho_prev"] = np.zeros((keep, B))

    k = 0
    iterator = trange(T, desc=f"simulate[{policy}]", leave=False) if show_progress else range(T)
    for t in iterator:
        out = trainer._policy_outputs(x)
        st = unpack_state(x, policy)
        ids = identities(params, st, out)

        if policy in ["taylor", "mod_taylor"]:
            if policy == "taylor":
                i_t = i_taylor(params_sim, out["pi"])
            else:
                assert rbar_by_regime is not None
                i_t = i_modified_taylor(params_sim, out["pi"], rbar_by_regime, st.s)
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
            if policy == "commitment":
                store["vartheta_prev"][k] = st.vartheta_prev.cpu().numpy()
                store["varrho_prev"][k] = st.varrho_prev.cpu().numpy()
            k += 1

        x = trainer._step_state(x)

    return store
