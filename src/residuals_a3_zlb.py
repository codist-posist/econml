from __future__ import annotations

from typing import Dict

import torch

from .config import ModelParams
from .model_common import State, identities


def _fischer_burmeister(a: torch.Tensor, b: torch.Tensor, eps_cc: float) -> torch.Tensor:
    eps_t = torch.as_tensor(float(eps_cc), device=a.device, dtype=a.dtype)
    return (eps_t + a + b - torch.sqrt(a * a + b * b + 4.0 * eps_t * eps_t)) / 2.0


def residuals_a3_zlb(
    params: ModelParams,
    st: State,
    out: Dict[str, torch.Tensor],
    Et_XiN_next: torch.Tensor,
    Et_XiD_next: torch.Tensor,
    Et_termN_next: torch.Tensor,
    Et_termD_next: torch.Tensor,
    Et_theta_zeta_pi_next: torch.Tensor,
    Et_H_next: torch.Tensor,
    *,
    eps_cc: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Author-style commitment with ZLB residuals (dsge_zlb_commitment):
      - eq_1, eq_2, eq_3, eq_4, eq_7 (Euler), eq_8 (complementarity), eq_9 (pstar_aux).
    """
    if st.vartheta_prev is None or st.varrho_prev is None:
        raise ValueError("commitment_zlb residuals require lagged vartheta_prev and varrho_prev in the state")

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
    vartheta = out["vartheta"]
    varrho = out["varrho"]
    varphi = out["varphi"]
    i_nom = out["i_nom"]
    lam = out["lam"]

    vartheta_old = st.vartheta_prev
    varrho_old = st.varrho_prev
    c_old = st.c_prev if st.c_prev is not None else c
    i_nom_old = st.i_nom_prev if st.i_nom_prev is not None else torch.zeros_like(c)
    varphi_old = st.varphi_prev if st.varphi_prev is not None else torch.zeros_like(c)

    eps = params.eps
    theta = params.theta
    beta = params.beta
    gamma = params.gamma
    omega = params.omega
    eps_denom = 1e-12

    # Definitions.chi_y for zlb commitment
    chi = -beta * varphi * Et_H_next

    res: Dict[str, torch.Tensor] = {}

    # eq_1
    res["res_c_foc"] = (
        c.pow(-gamma)
        - (c + g).pow(omega) * (Delta / A).pow(1.0 + omega)
        + vartheta
        * (
            ((1.0 + omega) * c + gamma * (c + g))
            * (c + g).pow(omega)
            * c.pow(gamma - 1.0)
            * (Delta / A).pow(omega)
            * one_plus_tau
            / A
            + Et_termN_next
        )
        + (1.0 / beta)
        * vartheta_old
        * (-gamma)
        * beta
        * theta
        * c_old.pow(gamma)
        * c.pow(-gamma - 1.0)
        * out["pi_aux"]
        * XiN
        + varrho * (1.0 + Et_termD_next)
        + (1.0 / beta)
        * varrho_old
        * (-gamma)
        * beta
        * theta
        * c_old.pow(gamma)
        * c.pow(-gamma - 1.0)
        * out["pi_aux"].pow((eps - 1.0) / eps)
        * XiD
        + (varphi - varphi_old * ((1.0 + i_nom_old) * out["pi_aux"].pow((eps - 1.0) / eps) / out["pi_aux"]))
        * gamma
        * c.pow(-gamma - 1.0)
    ) / (((c + g).pow(omega) * (Delta / A).pow(1.0 + omega)).abs() + eps_denom)

    # eq_2
    res["res_Delta_foc"] = (
        -((c + g) * Delta / A).pow(1.0 + omega) / Delta
        - zeta
        + beta * Et_theta_zeta_pi_next
        + vartheta * omega * (c + g).pow(1.0 + omega) * c.pow(gamma) * (Delta / A).pow(omega) / Delta * one_plus_tau / A
    ) / ((((c + g) * Delta / A).pow(1.0 + omega) / Delta).abs() + eps_denom)

    # eq_3, eq_4
    res["res_XiN_rec"] = (XiN - y * out["w"] * one_plus_tau * A.pow(-1.0) - Et_XiN_next) / (XiN.abs() + eps_denom)
    res["res_XiD_rec"] = (XiD - y - Et_XiD_next) / (XiD.abs() + eps_denom)

    # eq_7 Euler with i as explicit control
    res["res_euler_i"] = (1.0 + i_nom).pow(-1.0) * (lam / torch.clamp(beta * Et_H_next, min=eps_denom) - (1.0 + i_nom))
    # eq_8 complementarity
    res["res_zlb_comp"] = _fischer_burmeister(chi, i_nom, eps_cc)
    # eq_9
    res["res_pstar_def"] = (p_star_aux - pstar.pow(-eps)) / (p_star_aux.abs() + eps_denom)
    return res

