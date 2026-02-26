
"""
Experiments utilities for reproducing paper figures (except Fig 11/12).

Design principles:
- Do NOT change model equations.
- Use the trained policy network for controls.
- Build x_{t+1} via the model's laws of motion (zero innovations for IRF-style paths when requested).
- Allow deterministic regime paths (forced switches) for transition experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Callable

import numpy as np
import torch

from .config import ModelParams, PolicyName, TrainConfig
from .deqn import PolicyNetwork, Trainer, implied_nominal_rate_from_euler
from .model_common import unpack_state, shock_laws_of_motion, identities

@dataclass
class DeterministicPathSpec:
    """
    Deterministic innovations + optional forced regime path.

    If regime_path is provided, it must be length T+1 (including initial s_0).
    The step uses s_{t+1}=regime_path[t+1] regardless of Markov probabilities.
    """
    T: int
    epsA: float = 0.0
    epsg: float = 0.0
    epst: float = 0.0
    regime_path: Optional[Sequence[int]] = None  # length T+1

def _deterministic_step(
    trainer: Trainer,
    x: torch.Tensor,
    *,
    epsA: float,
    epsg: float,
    epst: float,
    s_next: Optional[torch.Tensor],
) -> torch.Tensor:
    """One-step transition with specified innovations and specified next regime."""
    params = trainer.params
    policy = trainer.policy
    st = unpack_state(x, policy)
    out = trainer._policy_outputs(x)

    B = x.shape[0]
    dev, dt = params.device, params.dtype
    epsA_t = torch.full((B,), float(epsA), device=dev, dtype=dt)
    epsg_t = torch.full((B,), float(epsg), device=dev, dtype=dt)
    epst_t = torch.full((B,), float(epst), device=dev, dtype=dt)

    if s_next is None:
        # default: follow Markov draw (stochastic). For deterministic IRFs, pass s_next explicitly.
        u = torch.rand(B, device=dev, dtype=dt)
        P = params.P
        p0 = P[st.s, 0]
        s_next = torch.where(u < p0, torch.zeros_like(st.s), torch.ones_like(st.s))

    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, epsA_t, epsg_t, epst_t, s_next)

    if policy == "commitment":
        vp = out["vartheta"] * out["c"].pow(params.gamma)
        rp = out["varrho"] * out["c"].pow(params.gamma)
        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), vp, rp], dim=-1)

    return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)

@torch.inference_mode()
def simulate_deterministic_path(
    params: ModelParams,
    policy: PolicyName,
    net: PolicyNetwork,
    *,
    x0: torch.Tensor,
    spec: DeterministicPathSpec,
    rbar_by_regime: Optional[torch.Tensor] = None,
    compute_implied_i: bool = True,
    gh_n: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Deterministic simulation for IRF/transition experiments.
    By default, computes implied i_t for discretion/commitment (cheaply via GH with gh_n=3).
    """
    net.eval()
    # minimal cfg for trainer; no training
    cfg_sim = TrainConfig.dev(seed=0) if params.device == "cpu" else TrainConfig.full(seed=0)
    tr = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, gh_n=int(gh_n), rbar_by_regime=rbar_by_regime)

    T = int(spec.T)
    B = x0.shape[0]
    x = x0.to(device=params.device, dtype=params.dtype)

    out_store = {k: np.zeros((T+1, B)) for k in ["c","pi","Delta","pstar","y","h","g","tau","i"]}
    out_store["s"] = np.zeros((T+1, B), dtype=np.int64)

    for t in range(T+1):
        out = tr._policy_outputs(x)
        st = unpack_state(x, policy)
        ids = identities(params, st, out)

        if policy in ["taylor","mod_taylor"]:
            if policy == "taylor":
                i_t = tr.params.beta**(-1) - 1.0 + tr.params.psi*(out["pi"] - tr.params.pi_bar) + (1.0+tr.params.pi_bar)/tr.params.beta - 1.0 - (tr.params.beta**(-1) - 1.0)
                # above reduces to (1+pi_bar)/beta -1 + psi*(pi-pi_bar) but keep safe
                from .policy_rules import i_taylor
                i_t = i_taylor(params, out["pi"])
            else:
                from .policy_rules import i_modified_taylor
                assert rbar_by_regime is not None
                i_t = i_modified_taylor(params, out["pi"], rbar_by_regime, st.s)
        else:
            if compute_implied_i:
                i_t = implied_nominal_rate_from_euler(params, policy, x, out, int(gh_n), tr)
            else:
                i_t = torch.full_like(out["pi"], float("nan"))

        out_store["c"][t] = out["c"].cpu().numpy()
        out_store["pi"][t] = out["pi"].cpu().numpy()
        out_store["Delta"][t] = out["Delta"].cpu().numpy()
        out_store["pstar"][t] = out["pstar"].cpu().numpy()
        out_store["y"][t] = ids["y"].cpu().numpy()
        out_store["h"][t] = ids["h"].cpu().numpy()
        out_store["g"][t] = ids["g"].cpu().numpy()
        out_store["tau"][t] = (ids["one_plus_tau"] - 1.0).cpu().numpy()
        out_store["s"][t] = st.s.cpu().numpy()
        out_store["i"][t] = i_t.cpu().numpy()

        if t == T:
            break

        # forced regime path?
        if spec.regime_path is not None:
            s_next = torch.full((B,), int(spec.regime_path[t+1]), device=params.device, dtype=torch.long)
        else:
            s_next = None

        x = _deterministic_step(tr, x, epsA=spec.epsA, epsg=spec.epsg, epst=spec.epst, s_next=s_next)

    return out_store

def calibrate_xi_jump_to_match_pi_impact(
    params: ModelParams,
    policy: PolicyName,
    net: PolicyNetwork,
    *,
    x0: torch.Tensor,
    target_pi0: float,
    horizon_T: int = 1,
    rho_tau_override: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """
    Find an initial xi jump (added to xi state at t=0) such that pi impact matches target_pi0,
    with zero innovations thereafter. This supports Fig 3 style comparisons.
    """
    # We'll do bisection on xi0 in a wide bracket.
    # Implementation: modify x0's xi component by +xi_jump, then simulate 0->1 step and read pi at t=0 (impact).
    # If rho override provided, we temporarily override params.rho_tau (copy params).

    p = params
    if rho_tau_override is not None:
        # create shallow copy of params with different rho_tau
        p = ModelParams(
            beta=p.beta, gamma=p.gamma, omega=p.omega, theta=p.theta, eps=p.eps, tau_bar=p.tau_bar,
            rho_A=p.rho_A, rho_tau=float(rho_tau_override), rho_g=p.rho_g,
            sigma_A=p.sigma_A, sigma_tau=p.sigma_tau, sigma_g=p.sigma_g,
            g_bar=p.g_bar, eta_bar=p.eta_bar, bad_state=p.bad_state,
            p12=p.p12, p21=p.p21,
            pi_bar=p.pi_bar, psi=p.psi,
            device=p.device, dtype=p.dtype
        ).to_torch()

    # indices: x = [Delta_prev, logA, loggtilde, xi, s] (or + multipliers for commitment)
    xi_idx = 3

    def impact_pi(xi_jump: float) -> float:
        x = x0.clone()
        x[:, xi_idx] = x[:, xi_idx] + float(xi_jump)
        spec = DeterministicPathSpec(T=horizon_T, epsA=0.0, epsg=0.0, epst=0.0, regime_path=None)
        out = simulate_deterministic_path(p, policy, net, x0=x, spec=spec, compute_implied_i=False)
        # impact at t=0 (first stored point)
        return float(np.mean(out["pi"][0]))

    # bracket
    lo, hi = -2.0, 2.0
    f_lo = impact_pi(lo) - target_pi0
    f_hi = impact_pi(hi) - target_pi0
    # expand if needed
    k = 0
    while f_lo * f_hi > 0 and k < 10:
        lo *= 2.0
        hi *= 2.0
        f_lo = impact_pi(lo) - target_pi0
        f_hi = impact_pi(hi) - target_pi0
        k += 1
    if f_lo * f_hi > 0:
        # fallback: return 0 jump if not bracketed
        return 0.0

    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        f_mid = impact_pi(mid) - target_pi0
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5*(lo+hi))
