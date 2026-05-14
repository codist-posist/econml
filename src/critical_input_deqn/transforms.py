from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable

import math
import torch
import torch.nn.functional as F

from .config import BaselineParams


def positive(raw: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    """Smooth nonnegative transform with a small strictly positive floor."""

    return F.softplus(raw) + floor


def gross_from_log(raw: torch.Tensor) -> torch.Tensor:
    """Positive gross rate/inflation transform."""

    return torch.exp(raw)


def _params_key(p: BaselineParams) -> tuple[float, ...]:
    return (
        float(p.alpha),
        float(p.beta),
        float(p.epsilon),
        float(p.theta),
        float(p.rho),
        float(p.sigma),
        float(p.varphi),
        float(p.omega0),
        float(p.bar_p_m),
        float(p.p_d),
        float(p.sigma_z),
        float(p.steady_state_output),
        float(p.bar_pi),
        float(p.bar_R),
        float(p.psi_A),
    )


@lru_cache(maxsize=256)
def _steady_targets_cached(key: tuple[float, ...]) -> dict[str, float]:
    (
        alpha,
        beta,
        epsilon,
        theta,
        rho,
        sigma,
        varphi,
        omega0,
        bar_p_m,
        p_d,
        sigma_z,
        steady_state_output,
        bar_pi,
        bar_R,
        psi_A,
    ) = key
    del steady_state_output  # solved below; the field is only part of the cache key.
    Z = math.exp(-0.5 * sigma_z**2)
    desired_mc = (epsilon - 1.0) / epsilon
    mu0 = min(max(omega0, 1e-8), 1.0 - 1e-8)
    p_x = (mu0 * bar_p_m ** (1.0 - rho) + (1.0 - mu0) * p_d ** (1.0 - rho)) ** (1.0 / (1.0 - rho))
    consumption_share = 1.0 - alpha * desired_mc
    labor_base = (((1.0 - alpha) * p_x) / alpha) ** alpha
    n_const = (labor_base / Z * consumption_share ** (-alpha * sigma)) ** (1.0 / (1.0 + alpha * varphi))
    w_const = n_const**varphi * consumption_share**sigma
    mc_const = (1.0 / Z) * (w_const / (1.0 - alpha)) ** (1.0 - alpha) * (p_x / alpha) ** alpha
    y_power = (1.0 - alpha) * (varphi * (1.0 - alpha * sigma) / (1.0 + alpha * varphi) + sigma)
    Y = (desired_mc / mc_const) ** (1.0 / y_power)
    C = consumption_share * Y
    continuation_F = max(1.0 - theta * beta * bar_pi ** (epsilon - 1.0), 1e-8)
    continuation_S = max(1.0 - theta * beta * bar_pi**epsilon, 1e-8)
    F_p = Y / continuation_F
    S_p = desired_mc * Y / continuation_S
    return {
        "C": max(C, 1e-8),
        "Y": max(Y, 1e-8),
        "Pi": max(bar_pi, 1e-8),
        "R": max(bar_R, 1e-8),
        "S_p": max(S_p, 1e-8),
        "F_p": max(F_p, 1e-8),
        "Q_A_scale": max(2.0, 4.0 * abs(psi_A)),
    }


def steady_decode_targets(params: BaselineParams | None = None) -> dict[str, float]:
    """Author-style calm-branch centers used by bounded output transforms."""

    p = params or BaselineParams()
    return _steady_targets_cached(_params_key(p))


def _bounded_log_center(raw: torch.Tensor, center: float | torch.Tensor, width: float, floor: float = 1e-10) -> torch.Tensor:
    if not torch.is_tensor(center):
        center_t = torch.as_tensor(float(center), device=raw.device, dtype=raw.dtype)
    else:
        center_t = center.to(device=raw.device, dtype=raw.dtype)
    return torch.clamp(center_t, min=floor) * torch.exp(float(width) * torch.tanh(raw))


def _bounded_signed(raw: torch.Tensor, scale: float) -> torch.Tensor:
    return float(scale) * torch.tanh(raw)


def _bounded_identity(raw: torch.Tensor, bound: float) -> torch.Tensor:
    """Approximately identity near zero, smoothly bounded in the tails."""

    b = max(float(bound), 1e-8)
    return b * torch.tanh(raw / b)


def calvo_index_implied_pstar(
    Pi: torch.Tensor,
    params: BaselineParams | None = None,
    *,
    floor: float = 1e-10,
) -> torch.Tensor:
    """Reset price implied by the Calvo price-index identity."""

    p = params or BaselineParams()
    theta = float(p.theta)
    epsilon = float(p.epsilon)
    if theta <= 0.0:
        return torch.ones_like(Pi)
    lhs = (1.0 - theta * Pi.pow(epsilon - 1.0)) / max(1.0 - theta, floor)
    return torch.clamp(lhs, min=floor).pow(1.0 / (1.0 - epsilon))


def _calvo_admissible_pi_width(params: BaselineParams | None = None) -> float:
    """Keep inflation inside the hard Calvo-index admissible region."""

    p = params or BaselineParams()
    theta = float(p.theta)
    epsilon = float(p.epsilon)
    if theta <= 0.0:
        return math.log(1.30)
    # Keep the implied reset price finite without clipping ordinary inflation
    # responses.  In the baseline this permits about +20.8% annualized inflation,
    # close to the Calvo-index admissibility ceiling of about +21.1%.
    pstar_hi = 2.0
    pi_hi = ((1.0 - (1.0 - theta) * pstar_hi ** (1.0 - epsilon)) / theta) ** (1.0 / (epsilon - 1.0))
    return max(math.log(min(pi_hi, 1.30)), math.log(1.005))


def decode_rule_outputs(
    raw: torch.Tensor,
    names: Iterable[str],
    *,
    params: BaselineParams | None = None,
    y_ref: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    targets = steady_decode_targets(params)
    out: Dict[str, torch.Tensor] = {}
    names_tuple = tuple(names)
    for i, name in enumerate(names_tuple):
        x = raw[..., i]
        if name == "C":
            out[name] = _bounded_log_center(x, targets["C"], math.log(3.0))
        elif name == "Y":
            center = y_ref if y_ref is not None else targets["Y"]
            out[name] = _bounded_log_center(x, center, 2.0)
        elif name == "Pi":
            out[name] = _bounded_log_center(x, targets["Pi"], _calvo_admissible_pi_width(params))
        elif name == "Q_A":
            out[name] = _bounded_signed(x, targets["Q_A_scale"])
        elif name == "S_p":
            out[name] = _bounded_log_center(x, targets["S_p"], math.log(4.0))
        elif name == "F_p":
            out[name] = _bounded_log_center(x, targets["F_p"], math.log(4.0))
        else:
            out[name] = positive(x)
    return out


def decode_natural_outputs(
    raw: torch.Tensor,
    names: Iterable[str],
    *,
    params: BaselineParams | None = None,
) -> Dict[str, torch.Tensor]:
    targets = steady_decode_targets(params)
    out: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        x = raw[..., i]
        if name == "C_n":
            out[name] = _bounded_log_center(x, targets["C"], math.log(3.0))
        elif name == "Y_n":
            out[name] = _bounded_log_center(x, targets["Y"], 2.0)
        elif name == "R_n_real":
            out[name] = _bounded_log_center(x, targets["R"], math.log(1.50))
        else:
            out[name] = positive(x)
    return out


def decode_optimal_outputs(
    raw: torch.Tensor,
    names: Iterable[str],
    *,
    params: BaselineParams | None = None,
) -> Dict[str, torch.Tensor]:
    targets = steady_decode_targets(params)
    out: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        x = raw[..., i]
        if name == "C":
            out[name] = _bounded_log_center(x, targets["C"], math.log(3.0))
        elif name == "Y":
            out[name] = _bounded_log_center(x, targets["Y"], 2.0)
        elif name == "R":
            out[name] = _bounded_log_center(x, targets["R"], math.log(1.50))
        elif name == "Pi":
            out[name] = _bounded_log_center(x, targets["Pi"], _calvo_admissible_pi_width(params))
        elif name == "Q_A":
            out[name] = _bounded_signed(x, targets["Q_A_scale"])
        elif name == "S_p":
            out[name] = _bounded_log_center(x, targets["S_p"], math.log(4.0))
        elif name == "F_p":
            out[name] = _bounded_log_center(x, targets["F_p"], math.log(4.0))
        elif name.startswith("mu_"):
            out[name] = _bounded_identity(x, 50.0)
        elif name.startswith("promise_"):
            out[name] = _bounded_identity(x, 5.0)
        else:
            out[name] = x
    return out
