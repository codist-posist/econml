from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .config import ModelParams, PolicyName
from .model_common import unpack_state, shock_laws_of_motion
from .transforms import decode_outputs


@dataclass
class PolicySSS:
    """Stochastic steady state computed as a fixed point of a (trained) policy."""

    by_regime: Dict[int, Dict[str, float]]


def _uncond_mean_log_ar1(rho: float, sigma: float) -> float:
    """Mean of log state under the drift-corrected AR(1) used in shock_laws_of_motion."""
    # With drift correction in model_common.shock_laws_of_motion, the stationary mean is:
    #   E[log x] = -(sigma^2)/(2*(1-rho^2))
    # (when innovations are N(0,1)).
    if abs(1.0 - rho) < 1e-12:
        return 0.0
    return float(-(sigma**2) / (2.0 * (1.0 - rho**2)))





# NOTE: Frozen-regime ("regime held fixed") SSS is *not* used in the paper's Table-2 objects,
# but we provide a diagnostic routine that computes it with the regime transition matrix set to identity.
# This is printed for debugging/comparison only and is never used in any downstream calculations.

def frozen_policy_sss_by_regime_from_policy(
    params: "ModelParams",
    net: "torch.nn.Module",
    *,
    policy: "PolicyName",
    **kwargs,
) -> "PolicySSS":
    """Diagnostic SSS with regimes held fixed (P = I).

    Returns an object with by_regime[0] and by_regime[1] computed under frozen regimes.
    """
    P_I = torch.eye(2, device=params.device, dtype=params.dtype)
    return switching_policy_sss_by_regime_from_policy(params, net, policy=policy, P_override=P_I, **kwargs)


def _stationary_dist_2state(P: torch.Tensor) -> torch.Tensor:
    """Stationary distribution for 2x2 Markov matrix with convention P[current, next] (row-stochastic)."""
    # Solve (I - P.T) pi = 0 with sum(pi)=1  (row-stochastic: pi = pi P)
    dev, dt = P.device, P.dtype
    I = torch.eye(2, device=dev, dtype=dt)
    A = (I - P.T)  # 2x2 (row-stochastic: solve pi = pi P)
    A = torch.vstack([A, torch.ones((1, 2), device=dev, dtype=dt)])  # 3x2
    b = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt).unsqueeze(-1)  # 3x1
    pi = torch.linalg.lstsq(A, b).solution.squeeze(-1)  # (2,)
    return pi

def _backward_weights_2state(P: torch.Tensor, pi_stat: torch.Tensor, curr_s: int) -> torch.Tensor:
    """w_prev[j] = Pr(s_{t-1}=j | s_t=curr_s)."""
    dev, dt = P.device, P.dtype
    w = torch.stack([P[0, curr_s] * pi_stat[0], P[1, curr_s] * pi_stat[1]])
    denom = w.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError(
            "Invalid Markov transition matrix: cannot form backward weights. "
            f"curr_s={curr_s}, denom={float(denom.item())}, P={P.detach().cpu().numpy().tolist()}, pi_stat={pi_stat.detach().cpu().numpy().tolist()}"
        )
    return w / denom



@torch.no_grad()
def switching_policy_sss_by_regime_from_policy(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    P_override: Optional[torch.Tensor] = None,
    max_iter: int = 50_000,
    tol: float = 1e-12,
    damping: float = 0.5,
    floors: Optional[Dict[str, float]] = None,
) -> PolicySSS:
    """Switching-consistent SSS-by-regime as a fixed point of the trained policy.

    This matches the paper's Table-2 'SSS by regime' definition:
      - innovations set to zero
      - continuous shocks fixed at their drift-corrected stationary means
      - regime switching enters through expectations/backward weights via params.P

    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    # Keep state tensors dtype-compatible with the policy network. After phase-2
    # training, the net can be float64 while caller-side params are still float32.
    try:
        net_dt = next(net.parameters()).dtype
    except StopIteration:
        net_dt = params.dtype

    dev, dt = params.device, net_dt
    P = (P_override.to(device=dev, dtype=dt) if P_override is not None else params.P.to(device=dev, dtype=dt))
    pi_stat = _stationary_dist_2state(P)

    # Exogenous states fixed at unconditional means (drift-corrected AR(1))
    logA0 = _uncond_mean_log_ar1(float(params.rho_A), float(params.sigma_A))
    logg0 = _uncond_mean_log_ar1(float(params.rho_g), float(params.sigma_g))
    xi0 = 0.0

    if policy == "commitment":
        # State: (Delta_prev, logA, logg, xi, s, vp_prev, rp_prev)
        x0 = torch.tensor([1.0, logA0, logg0, xi0, 0.0, 0.0, 0.0], device=dev, dtype=dt).view(1, -1)
        x1 = torch.tensor([1.0, logA0, logg0, xi0, 1.0, 0.0, 0.0], device=dev, dtype=dt).view(1, -1)
    else:
        # State: (Delta_prev, logA, logg, xi, s)
        x0 = torch.tensor([1.0, logA0, logg0, xi0, 0.0], device=dev, dtype=dt).view(1, -1)
        x1 = torch.tensor([1.0, logA0, logg0, xi0, 1.0], device=dev, dtype=dt).view(1, -1)

    for _ in range(int(max_iter)):
        out0 = decode_outputs(policy, net(x0), floors=floors)
        out1 = decode_outputs(policy, net(x1), floors=floors)

        Delta0 = out0["Delta"].view(())
        Delta1 = out1["Delta"].view(())

        # backward weights for previous regime given current regime
        wprev0 = _backward_weights_2state(P, pi_stat, 0)
        wprev1 = _backward_weights_2state(P, pi_stat, 1)

        DeltaPrev0 = (wprev0[0] * Delta0 + wprev0[1] * Delta1).view(())
        DeltaPrev1 = (wprev1[0] * Delta0 + wprev1[1] * Delta1).view(())

        if policy == "commitment":
            gamma = params.gamma
            vp0 = (out0["vartheta"] * out0["c"].pow(gamma)).view(())
            vp1 = (out1["vartheta"] * out1["c"].pow(gamma)).view(())
            rp0 = (out0["varrho"] * out0["c"].pow(gamma)).view(())
            rp1 = (out1["varrho"] * out1["c"].pow(gamma)).view(())

            vpPrev0 = (wprev0[0] * vp0 + wprev0[1] * vp1).view(())
            vpPrev1 = (wprev1[0] * vp0 + wprev1[1] * vp1).view(())
            rpPrev0 = (wprev0[0] * rp0 + wprev0[1] * rp1).view(())
            rpPrev1 = (wprev1[0] * rp0 + wprev1[1] * rp1).view(())

            x0_next = torch.tensor([float(DeltaPrev0.item()), logA0, logg0, xi0, 0.0, float(vpPrev0.item()), float(rpPrev0.item())], device=dev, dtype=dt).view(1, -1)
            x1_next = torch.tensor([float(DeltaPrev1.item()), logA0, logg0, xi0, 1.0, float(vpPrev1.item()), float(rpPrev1.item())], device=dev, dtype=dt).view(1, -1)
        else:
            x0_next = torch.tensor([float(DeltaPrev0.item()), logA0, logg0, xi0, 0.0], device=dev, dtype=dt).view(1, -1)
            x1_next = torch.tensor([float(DeltaPrev1.item()), logA0, logg0, xi0, 1.0], device=dev, dtype=dt).view(1, -1)

        diff = max(float((x0_next - x0).abs().max().item()), float((x1_next - x1).abs().max().item()))
        x0 = (1.0 - damping) * x0 + damping * x0_next
        x1 = (1.0 - damping) * x1 + damping * x1_next
        if diff < tol:
            break

    # report
    out0 = decode_outputs(policy, net(x0), floors=floors)
    out1 = decode_outputs(policy, net(x1), floors=floors)

    def _pack_out(x: torch.Tensor, out: Dict[str, torch.Tensor]) -> Dict[str, float]:
        base = {
            "c": float(out["c"].item()),
            "pi": float(out["pi"].item()),
            "pstar": float(out["pstar"].item()),
            "lam": float(out["lam"].item()),
            "w": float(out["w"].item()),
            "XiN": float(out["XiN"].item()),
            "XiD": float(out["XiD"].item()),
            "Delta_prev": float(x[0, 0].item()),
            "Delta": float(out["Delta"].item()),
            "mu": float(out.get("mu", torch.tensor(0.0, device=dev, dtype=dt)).item()),
            "nu": float(out.get("nu", torch.tensor(0.0, device=dev, dtype=dt)).item()),
            "zeta": float(out.get("zeta", torch.tensor(0.0, device=dev, dtype=dt)).item()),
            "logA": float(x[0, 1].item()),
            "loggtilde": float(x[0, 2].item()),
            "xi": float(x[0, 3].item()),
        }
        if policy == "commitment":
            base.update({
                "vartheta": float(out["vartheta"].item()),
                "varrho": float(out["varrho"].item()),
                "vartheta_prev": float(x[0, 5].item()),
                "varrho_prev": float(x[0, 6].item()),
            })
        return base

    return PolicySSS(by_regime={0: _pack_out(x0, out0), 1: _pack_out(x1, out1)})


