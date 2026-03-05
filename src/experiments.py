
"""
Experiments utilities for reproducing paper figures (except Fig 11/12).

Design principles:
- Do NOT change model equations.
- Use the trained policy network for controls.
- Build x_{t+1} via the model's laws of motion (zero innovations for IRF-style paths when requested).
- Allow deterministic regime paths (forced switches) for transition experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple, Callable

import numpy as np
import torch

from .config import ModelParams, PolicyName, TrainConfig
from .deqn import PolicyNetwork, Trainer, implied_nominal_rate_from_euler, _transition_probs_to_next
from .model_common import unpack_state, shock_laws_of_motion, identities, transition_probs_to_next_regimes, regime_sigma_tau

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
        probs = _transition_probs_to_next(params, st)  # (B,R)
        cdf = torch.cumsum(probs, dim=-1)
        cdf[:, -1] = 1.0
        s_next = torch.sum(u.view(-1, 1) > cdf, dim=-1).to(torch.long)

    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, epsA_t, epsg_t, epst_t, s_next)

    if policy == "commitment":
        if st.c_prev is not None:
            return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"], out["c"]], dim=-1)
        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"]], dim=-1)

    if policy == "commitment_zlb":
        return torch.stack(
            [out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"], out["c"], out["i_nom"], out["varphi"]],
            dim=-1,
        )
    if policy == "taylor_para":
        if st.p21 is None and int(x.shape[-1]) < 7:
            return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
        p21_prev = st.p21 if st.p21 is not None else torch.full_like(out["Delta"], float(params.p21))
        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["i_nom"], p21_prev], dim=-1)

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
    if params.device == "cpu":
        cfg_sim = TrainConfig.dev(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    else:
        cfg_sim = TrainConfig.full(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    tr = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, gh_n=int(gh_n), rbar_by_regime=rbar_by_regime)

    T = int(spec.T)
    B = x0.shape[0]
    x = x0.to(device=params.device, dtype=params.dtype)

    out_store = {k: np.zeros((T+1, B)) for k in ["c","pi","Delta","pstar","y","h","g","A","tau","i","logA","loggtilde","xi","p12_eff","p21_eff","sigma_tau_t"]}
    out_store["s"] = np.zeros((T+1, B), dtype=np.int64)

    for t in range(T+1):
        out = tr._policy_outputs(x)
        st = unpack_state(x, policy)
        ids = identities(params, st, out)

        explicit_i_policies = ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "commitment_zlb")
        if policy in explicit_i_policies:
            if "i_nom" not in out:
                raise RuntimeError(f"policy={policy} expected explicit i_nom in decoded outputs.")
            i_t = out["i_nom"]
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
        out_store["A"][t] = ids["A"].cpu().numpy()
        out_store["tau"][t] = (ids["one_plus_tau"] - 1.0).cpu().numpy()
        out_store["logA"][t] = st.logA.cpu().numpy()
        out_store["loggtilde"][t] = st.loggtilde.cpu().numpy()
        out_store["xi"][t] = st.xi.cpu().numpy()
        out_store["s"][t] = st.s.cpu().numpy()
        out_store["i"][t] = i_t.cpu().numpy()
        p21_state = st.p21 if st.p21 is not None else None
        probs_s0 = transition_probs_to_next_regimes(
            params,
            torch.zeros_like(st.s),
            xi=st.xi,
            p21_state=p21_state,
        )
        probs_s1 = transition_probs_to_next_regimes(
            params,
            torch.ones_like(st.s),
            xi=st.xi,
            p21_state=p21_state,
        )
        out_store["p12_eff"][t] = probs_s0[:, 1].cpu().numpy()
        out_store["p21_eff"][t] = probs_s1[:, 0].cpu().numpy()
        out_store["sigma_tau_t"][t] = regime_sigma_tau(params, st.s).cpu().numpy()

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
    regime_path: Optional[Sequence[int]] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """
    Find an initial xi jump (added to xi state at t=0) such that pi impact matches target_pi0,
    with zero innovations thereafter. This supports Fig 3 style comparisons.
    """
    # We do bisection on xi0 in a wide bracket.
    # Implementation: modify x0's xi component by +xi_jump, then run a deterministic
    # path (zero innovations) and read pi at t=0 (impact).
    # If rho override is provided, we temporarily override params.rho_tau (copy params).

    p = params
    if rho_tau_override is not None:
        # create shallow copy of params with different rho_tau
        p = replace(p, rho_tau=float(rho_tau_override)).to_torch()

    # indices: x = [Delta_prev, logA, loggtilde, xi, s] (or + multipliers for commitment)
    xi_idx = 3

    # Default to a fixed-regime deterministic path (no random Markov draws) to keep
    # figure calibration reproducible and paper-consistent.
    if regime_path is None:
        s0 = int(float(x0[0, 4].detach().cpu()))
        reg_path = [s0] * (int(horizon_T) + 1)
    else:
        reg_path = [int(v) for v in regime_path]
        if len(reg_path) != int(horizon_T) + 1:
            raise ValueError(
                f"regime_path must have length horizon_T+1={int(horizon_T)+1}, got {len(reg_path)}"
            )

    def impact_pi(xi_jump: float) -> float:
        x = x0.clone()
        x[:, xi_idx] = x[:, xi_idx] + float(xi_jump)
        spec = DeterministicPathSpec(T=horizon_T, epsA=0.0, epsg=0.0, epst=0.0, regime_path=reg_path)
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
