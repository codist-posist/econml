from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import QMCConfig


@dataclass(frozen=True)
class QMCNodes:
    eps_z: torch.Tensor
    eps_lam_D: torch.Tensor
    eps_lam_X: torch.Tensor
    u_N_D: torch.Tensor
    u_N_X: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.eps_z.numel())


def _normal_icdf(u: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(u, 1e-12, 1.0 - 1e-12)
    return torch.sqrt(torch.tensor(2.0, dtype=u.dtype, device=u.device)) * torch.erfinv(2.0 * u - 1.0)


def make_qmc_nodes(
    n: int,
    *,
    cfg: QMCConfig = QMCConfig(),
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> QMCNodes:
    """Fixed Sobol nodes for the baseline transition.

    Dimensions:
    0 productivity shock,
    1 disruption-intensity shock,
    2 relief-intensity shock,
    3 disruption-arrival uniform,
    4 relief-arrival uniform.
    """

    engine = torch.quasirandom.SobolEngine(dimension=5, scramble=cfg.scramble, seed=cfg.seed)
    u = engine.draw(int(n)).to(device=device, dtype=dtype)
    return QMCNodes(
        eps_z=_normal_icdf(u[:, 0]),
        eps_lam_D=_normal_icdf(u[:, 1]),
        eps_lam_X=_normal_icdf(u[:, 2]),
        u_N_D=u[:, 3],
        u_N_X=u[:, 4],
    )


def poisson_icdf(u: torch.Tensor, lam: torch.Tensor, max_count: int) -> torch.Tensor:
    """Vectorized inverse-CDF sampler for small Poisson counts.

    Values above the truncated CDF are assigned to max_count. This is used for
    fixed QMC expectation nodes, not for differentiating with respect to the
    arrival intensity.
    """

    u = torch.clamp(u, 1e-12, 1.0 - 1e-12)
    lam = torch.clamp(lam, min=1e-12)
    p = torch.exp(-lam)
    cdf = p.clone()
    count = torch.zeros_like(lam)
    assigned = u <= cdf
    for k in range(1, int(max_count) + 1):
        p = p * lam / float(k)
        cdf = cdf + p
        take = (~assigned) & (u <= cdf)
        count = torch.where(take, torch.full_like(count, float(k)), count)
        assigned = assigned | take
    count = torch.where(assigned, count, torch.full_like(count, float(max_count)))
    return count

