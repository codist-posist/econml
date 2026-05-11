from __future__ import annotations

import math

import torch

from .config import BaselineParams


def sample_rule_states(
    n: int,
    *,
    params: BaselineParams,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> torch.Tensor:
    """Mixture sampler over normal, crisis, relief, and boundary-like states."""

    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    else:
        gen = None
    u = torch.rand((int(n),), generator=gen, device=device, dtype=dtype)
    D = torch.zeros_like(u)
    X = torch.zeros_like(u)
    A = torch.zeros_like(u)

    normal = u < 0.50
    crisis = (u >= 0.50) & (u < 0.75)
    relief = (u >= 0.75) & (u < 0.90)
    boundary = u >= 0.90

    D = torch.where(normal, 0.05 * torch.rand_like(u), D)
    X = torch.where(normal, 0.03 * torch.rand_like(u), X)
    A = torch.where(normal, 0.05 * torch.rand_like(u), A)

    D = torch.where(crisis, 0.5 + 1.5 * torch.rand_like(u), D)
    X = torch.where(crisis, 0.15 * torch.rand_like(u), X)
    A = torch.where(crisis, 0.6 * torch.rand_like(u), A)

    D = torch.where(relief, 0.5 + torch.rand_like(u), D)
    X = torch.where(relief, 0.25 + 0.75 * torch.rand_like(u), X)
    A = torch.where(relief, 0.1 + 1.1 * torch.rand_like(u), A)

    # Boundary-like states include high-adaptation support reachable under sustained repair.
    D = torch.where(boundary, 0.2 + 0.5 * torch.rand_like(u), D)
    X = torch.where(boundary, 0.1 + 0.4 * torch.rand_like(u), X)
    A = torch.where(boundary, 0.05 + 0.95 * torch.rand_like(u), A)

    ell_D = torch.full_like(u, float(params.log_bar_lambda_D)) + 0.15 * torch.randn_like(u)
    ell_X = torch.full_like(u, float(params.log_bar_lambda_X)) + 0.15 * torch.randn_like(u)
    log_Z = float(-0.5 * params.sigma_z**2) + 0.02 * torch.randn_like(u)
    log_Delta_prev = torch.log1p(0.02 * torch.rand_like(u))
    return torch.stack([D, X, ell_D, ell_X, log_Z, A, log_Delta_prev], dim=-1)


def natural_from_rule_states(z: torch.Tensor) -> torch.Tensor:
    return z[..., :6]
