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
    Appendix A.1 residuals (Taylor / modified Taylor block).

    IMPORTANT:
      Euler/Fisher equation is:  1 = E_t[ beta * (1+i_t)/(1+pi_{t+1}) * (lam_{t+1}/lam_t) ].
      Therefore residual is:     1 - Et_euler  (or Et_euler - 1).
    """
    ids = identities(params, st, out)

    c, lam, w = out["c"], out["lam"], out["w"]
    pi, pstar = out["pi"], out["pstar"]
    XiN, XiD = out["XiN"], out["XiD"]
    Delta = out["Delta"]

    y, h = ids["y"], ids["h"]
    one_plus_tau = ids["one_plus_tau"]
    A = ids["A"]

    # use "safe by construction" object from decode_outputs if present
    one_plus_pi = out.get("one_plus_pi", 1.0 + pi)

    res: Dict[str, torch.Tensor] = {}

    # (A.1) definitions / intratemporal conditions
    res["res_c_lam"] = c.pow(-params.gamma) - lam
    res["res_labor"] = h.pow(params.omega) - w * lam

    # (A.1) Euler/Fisher: 1 = Et_euler
    res["res_euler"] = 1.0 - Et_euler

    # (A.1) Xi recursions
    eps_denom = 1e-12
    res["res_XiN"] = (XiN - (y * w * one_plus_tau / A) - Et_XiN_next) / (XiN.abs() + eps_denom)
    res["res_XiD"] = (XiD - y - Et_XiD_next) / (XiD.abs() + eps_denom)

    # Optimal reset price
    res["res_pstar_def"] = (pstar - (params.M * XiN / XiD)) / (pstar.abs() + eps_denom)

    # Calvo price index / inflation condition
    res["res_calvo"] = 1.0 - (
        params.theta * one_plus_pi.pow(params.eps - 1.0)
        + (1.0 - params.theta) * pstar.pow(1.0 - params.eps)
    )

    # Price dispersion law
    res["res_Delta"] = (Delta - (
        params.theta * one_plus_pi.pow(params.eps) * st.Delta_prev
        + (1.0 - params.theta) * pstar.pow(-params.eps)
    )) / (Delta.abs() + eps_denom)

    # Optional policy-rule residual (author taylor_para-style eq_8 analogue).
    if (i_t_current is not None) and (i_rule_target is not None):
        res["res_i_rule"] = i_t_current - i_rule_target

    return res

