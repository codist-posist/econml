from __future__ import annotations

from typing import Dict

import torch

from .config import ModelParams
from .model_common import State, identities


def residuals_a1(
    params: ModelParams,
    st: State,
    out: Dict[str, torch.Tensor],
    Et_XiN_next: torch.Tensor,
    Et_XiD_next: torch.Tensor,
    Et_euler: torch.Tensor,
    *,
    i_t_current: torch.Tensor | None = None,
    i_rule_target: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Author-style Taylor / Taylor-para residuals:
      - eq_1, eq_2, eq_3, eq_7 (+ eq_5, eq_8 for para variant).
    """
    ids = identities(params, st, out)
    eps_denom = 1e-12

    XiN = out["XiN"]
    XiD = out["XiD"]
    lam = out["lam"]
    pstar = out["pstar"]
    Delta = out["Delta"]
    pi_aux = out["pi_aux"]
    p_star_aux = out["p_star_aux"]

    y = ids["y"]
    A = ids["A"]
    one_plus_tau = ids["one_plus_tau"]
    w = out["w"]

    res: Dict[str, torch.Tensor] = {}
    res["res_euler"] = 1.0 - Et_euler
    res["res_XiN"] = (XiN - w * one_plus_tau * (y / A) - Et_XiN_next) / (XiN.abs() + eps_denom)
    res["res_XiD"] = (XiD - y - Et_XiD_next) / (XiD.abs() + eps_denom)

    # Author eq_7:
    # p_star_aux^(1/eps) = p_star^(-1)
    res["res_pstar_def"] = pstar * (p_star_aux.pow(1.0 / params.eps) - pstar.pow(-1.0))

    # Para-rule extensions (author dsge_taylor_para):
    if (i_t_current is not None) and (i_rule_target is not None):
        res["res_Delta"] = (
            params.theta * pi_aux * st.Delta_prev + (1.0 - params.theta) * p_star_aux - Delta
        ) / (Delta.abs() + eps_denom)
        res["res_i_rule"] = i_rule_target - i_t_current

    return res
