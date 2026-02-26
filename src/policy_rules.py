from __future__ import annotations
import torch
from .config import ModelParams


def i_taylor(params: ModelParams, pi: torch.Tensor) -> torch.Tensor:
    # i_t = (1+pi_bar)/beta - 1 + psi*(pi - pi_bar)
    return (1.0 + params.pi_bar) / params.beta - 1.0 + params.psi * (pi - params.pi_bar)


def i_modified_taylor(params: ModelParams, pi: torch.Tensor, rbar_by_regime: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    rbar = rbar_by_regime[s]
    return rbar + params.psi * (pi - params.pi_bar)


def fisher_euler_term(
    params: ModelParams,
    i_t: torch.Tensor,
    pi_next: torch.Tensor,
    lam_next: torch.Tensor,
    lam_t: torch.Tensor,
) -> torch.Tensor:
    # beta * (1+i_t)/(1+pi_{t+1}) * (lam_{t+1}/lam_t)
    return params.beta * ((1.0 + i_t) / (1.0 + pi_next)) * (lam_next / lam_t)

