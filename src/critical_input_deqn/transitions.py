from __future__ import annotations

import torch

from .config import BaselineParams, QMCConfig
from .economics import State
from .qmc import QMCNodes, poisson_icdf


def transition_rule_states(
    st: State,
    A_next: torch.Tensor,
    Delta_next: torch.Tensor,
    nodes: QMCNodes,
    p: BaselineParams,
    qmc_cfg: QMCConfig,
) -> torch.Tensor:
    """Next rule/discretion state for every current state and QMC node.

    Returns shape (B,S,7), where S is number of QMC nodes.
    """

    B = st.D.shape[0]
    S = nodes.n
    D = st.D[:, None]
    X = st.X[:, None]
    ell_D = st.ell_D[:, None]
    ell_X = st.ell_X[:, None]
    log_Z = st.log_Z[:, None]

    lam_D = torch.exp(ell_D)
    lam_X = torch.exp(ell_X)
    n_D = poisson_icdf(nodes.u_N_D[None, :].expand(B, S), lam_D.expand(B, S), qmc_cfg.poisson_max_count)
    n_X = poisson_icdf(nodes.u_N_X[None, :].expand(B, S), lam_X.expand(B, S), qmc_cfg.poisson_max_count)

    D_next = (1.0 - float(p.delta_D)) * D + n_D * float(p.mark_D)
    X_next = (1.0 - float(p.delta_X)) * X + n_X * float(p.mark_X)
    ell_D_next = (
        (1.0 - float(p.rho_lambda_D)) * float(p.log_bar_lambda_D)
        + float(p.rho_lambda_D) * ell_D
        + float(p.kappa_D_lambda) * D
        + float(p.sigma_lambda_D) * nodes.eps_lam_D[None, :]
    )
    ell_X_next = (
        (1.0 - float(p.rho_lambda_X)) * float(p.log_bar_lambda_X)
        + float(p.rho_lambda_X) * ell_X
        + float(p.beta_X) * D
        + float(p.sigma_lambda_X) * nodes.eps_lam_X[None, :]
    )
    log_Z_next = float(p.rho_z) * log_Z + float(p.sigma_z) * nodes.eps_z[None, :]
    A_next_b = A_next[:, None].expand(B, S)
    log_Delta_next = torch.log(torch.clamp(Delta_next[:, None].expand(B, S), min=1e-12))

    return torch.stack(
        [D_next, X_next, ell_D_next, ell_X_next, log_Z_next, A_next_b, log_Delta_next],
        dim=-1,
    )


def transition_natural_states(
    st: State,
    nodes: QMCNodes,
    p: BaselineParams,
    qmc_cfg: QMCConfig,
) -> torch.Tensor:
    """Next flexible-price benchmark state for every current state and QMC node.

    Natural output uses installed A_t as a state and does not choose current
    repair. Hence the benchmark transition carries A forward under the
    no-new-repair law used in the natural-rate Euler residual.
    """

    B = st.D.shape[0]
    S = nodes.n
    D = st.D[:, None]
    X = st.X[:, None]
    ell_D = st.ell_D[:, None]
    ell_X = st.ell_X[:, None]
    log_Z = st.log_Z[:, None]
    lam_D = torch.exp(ell_D)
    lam_X = torch.exp(ell_X)
    n_D = poisson_icdf(nodes.u_N_D[None, :].expand(B, S), lam_D.expand(B, S), qmc_cfg.poisson_max_count)
    n_X = poisson_icdf(nodes.u_N_X[None, :].expand(B, S), lam_X.expand(B, S), qmc_cfg.poisson_max_count)

    D_next = (1.0 - float(p.delta_D)) * D + n_D * float(p.mark_D)
    X_next = (1.0 - float(p.delta_X)) * X + n_X * float(p.mark_X)
    ell_D_next = (
        (1.0 - float(p.rho_lambda_D)) * float(p.log_bar_lambda_D)
        + float(p.rho_lambda_D) * ell_D
        + float(p.kappa_D_lambda) * D
        + float(p.sigma_lambda_D) * nodes.eps_lam_D[None, :]
    )
    ell_X_next = (
        (1.0 - float(p.rho_lambda_X)) * float(p.log_bar_lambda_X)
        + float(p.rho_lambda_X) * ell_X
        + float(p.beta_X) * D
        + float(p.sigma_lambda_X) * nodes.eps_lam_X[None, :]
    )
    log_Z_next = float(p.rho_z) * log_Z + float(p.sigma_z) * nodes.eps_z[None, :]
    A_next = ((1.0 - float(p.delta_A)) * st.A)[:, None].expand(B, S)

    return torch.stack([D_next, X_next, ell_D_next, ell_X_next, log_Z_next, A_next], dim=-1)
