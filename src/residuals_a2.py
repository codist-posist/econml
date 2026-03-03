from __future__ import annotations

from typing import Dict

import torch

from .config import ModelParams
from .model_common import State, identities


def residuals_a2(
    params: ModelParams,
    st: State,
    out: Dict[str, torch.Tensor],
    Et_F_next: torch.Tensor,
    Et_G_next: torch.Tensor,
    Et_dF_dDelta_next: torch.Tensor,
    Et_dG_dDelta_next: torch.Tensor,
    Et_theta_zeta_pi_next: torch.Tensor,
    Et_XiN_next: torch.Tensor,
    Et_XiD_next: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Author-style discretion residuals:
      - eq_1, eq_2, eq_3, eq_4, eq_7.
    """
    ids = identities(params, st, out)

    A = ids["A"]
    g = ids["g"]
    y = ids["y"]
    one_plus_tau = ids["one_plus_tau"]

    c = out["c"]
    pstar = out["pstar"]
    p_star_aux = out["p_star_aux"]
    XiN = out["XiN"]
    XiD = out["XiD"]
    Delta = out["Delta"]
    zeta = out["zeta"]
    eta = out["eta"]

    eps = params.eps
    theta = params.theta
    beta = params.beta
    M = params.M
    gamma = params.gamma
    omega = params.omega
    eps_denom = 1e-12

    # Author Definitions.py (discretion): mu_y
    mu_num = eta * (1.0 - theta) * (1.0 - eps) * p_star_aux - zeta * (1.0 - theta) * eps * p_star_aux.pow(
        (eps + 1.0) / eps
    )
    mu_den = y + c.pow(gamma) * Et_F_next
    mu = -mu_num / torch.clamp(mu_den, min=eps_denom)

    res: Dict[str, torch.Tensor] = {}

    # eq_1
    term1 = c.pow(-gamma) - (c + g).pow(omega) * (Delta / A).pow(1.0 + omega)
    term2 = mu * (
        pstar * (1.0 + gamma * c.pow(gamma - 1.0) * Et_F_next)
        - M
        * (
            ((1.0 + omega) * c + gamma * (c + g))
            * (c + g).pow(omega)
            * c.pow(gamma - 1.0)
            * one_plus_tau
            * A.pow(-1.0)
            * (Delta / A).pow(omega)
            + gamma * c.pow(gamma - 1.0) * Et_G_next
        )
    )
    denom_c = ((c + g).pow(omega) * (Delta / A).pow(1.0 + omega)).abs() + eps_denom
    res["res_c_foc"] = (term1 + term2) / denom_c

    # eq_2
    denom_D = (((c + g) * Delta / A).pow(1.0 + omega) / Delta).abs() + eps_denom
    res["res_Delta_foc"] = (
        -((c + g) * Delta / A).pow(1.0 + omega) / Delta
        + beta * Et_theta_zeta_pi_next
        - zeta
        + mu
        * (
            pstar * c.pow(gamma) * Et_dF_dDelta_next
            - M
            * (
                (c + g).pow(1.0 + omega)
                * c.pow(gamma)
                * (omega / Delta)
                * (Delta / A).pow(omega)
                * one_plus_tau
                * A.pow(-1.0)
                + c.pow(gamma) * Et_dG_dDelta_next
            )
        )
    ) / denom_D

    # eq_3, eq_4
    res["res_XiN_rec"] = (XiN - y * out["w"] * one_plus_tau * A.pow(-1.0) - Et_XiN_next) / (XiN.abs() + eps_denom)
    res["res_XiD_rec"] = (XiD - y - Et_XiD_next) / (XiD.abs() + eps_denom)

    # eq_7
    res["res_pstar_def"] = (p_star_aux - pstar.pow(-eps)) / (p_star_aux.abs() + eps_denom)
    return res
