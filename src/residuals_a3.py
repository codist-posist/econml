from __future__ import annotations
from typing import Dict
import torch
from .config import ModelParams
from .model_common import State, identities


def residuals_a3(
    params: ModelParams,
    st: State,
    out: Dict[str, torch.Tensor],
    Et_XiN_next: torch.Tensor,
    Et_XiD_next: torch.Tensor,
    Et_termN_next: torch.Tensor,
    Et_termD_next: torch.Tensor,
    Et_theta_zeta_pi_next: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    assert st.vartheta_prev is not None and st.varrho_prev is not None

    ids = identities(params, st, out)
    A = ids["A"]
    g = ids["g"]
    y = ids["y"]
    h = ids["h"]
    one_plus_tau = ids["one_plus_tau"]

    c = out["c"]
    pi = out["pi"]
    pstar = out["pstar"]
    lam = out["lam"]
    w = out["w"]

    XiN = out["XiN"]
    XiD = out["XiD"]
    Delta = out["Delta"]

    mu = out["mu"]
    nu = out["nu"]
    zeta = out["zeta"]
    vartheta = out["vartheta"]
    varrho = out["varrho"]

    # safe-by-construction inflation object (strictly consistent with transforms.py)
    one_plus_pi = out.get("one_plus_pi", 1.0 + pi)

    eps = params.eps
    theta = params.theta
    beta = params.beta
    M = params.M
    gamma = params.gamma
    omega = params.omega

    # lagged scaled co-states stored in the state:
    # vp_{t-1} = vartheta_{t-1} * c_{t-1}^gamma
    # rp_{t-1} = varrho_{t-1}   * c_{t-1}^gamma
    vp = st.vartheta_prev
    rp = st.varrho_prev

    res: Dict[str, torch.Tensor] = {}

    # ---- FOC w.r.t. c (Appendix A.3) ----
    term_util = c.pow(-gamma) - (c + g).pow(omega) * (Delta / A).pow(1.0 + omega)

    term_vartheta_now = (
        ((1.0 + omega) * c + gamma * (c + g))
        * (c + g).pow(omega) * c.pow(gamma - 1.0)
        * (Delta / A).pow(omega) * one_plus_tau / A
        + Et_termN_next
    )
    term_vartheta_lag = (-gamma) * theta * vp * c.pow(-gamma - 1.0) * one_plus_pi.pow(eps) * XiN

    term_varrho_now = (1.0 + Et_termD_next)
    term_varrho_lag = (-gamma) * theta * rp * c.pow(-gamma - 1.0) * one_plus_pi.pow(eps - 1.0) * XiD

    res["res_c_foc"] = (
        term_util
        + vartheta * term_vartheta_now
        + term_vartheta_lag
        + varrho * term_varrho_now
        + term_varrho_lag
    )

    # ---- FOC w.r.t. Delta ----
    disutil_over_Delta = -((c + g) * Delta / A).pow(1.0 + omega) / Delta
    res["res_Delta_foc"] = (
        disutil_over_Delta
        - zeta
        + beta * Et_theta_zeta_pi_next
        + vartheta
        * omega * (c + g).pow(1.0 + omega) * c.pow(gamma)
        * (Delta / A).pow(omega) * (1.0 / Delta) * one_plus_tau / A
    )

    # ---- FOC w.r.t. pi ----
    res["res_pi_foc"] = (
        nu * theta * (eps - 1.0) * one_plus_pi.pow(eps - 2.0)
        + zeta * theta * eps * one_plus_pi.pow(eps - 1.0) * st.Delta_prev
        + eps * theta * vp * c.pow(-gamma) * one_plus_pi.pow(eps - 1.0) * XiN
        + (eps - 1.0) * theta * rp * c.pow(-gamma) * one_plus_pi.pow(eps - 2.0) * XiD
    )

    # ---- FOC w.r.t. pstar ----
    res["res_pstar_foc"] = (
        -mu * XiD
        + nu * (1.0 - theta) * (1.0 - eps) * pstar.pow(-eps)
        + zeta * (1.0 - theta) * (-eps) * pstar.pow(-eps - 1.0)
    )

    # ---- FOC w.r.t. XiN, XiD ----
    res["res_XiN_foc"] = mu * M - vartheta + theta * vp * c.pow(-gamma) * one_plus_pi.pow(eps)
    res["res_XiD_foc"] = -mu * pstar - varrho + theta * rp * c.pow(-gamma) * one_plus_pi.pow(eps - 1.0)

    # ---- Equilibrium conditions block ----
    res["res_c_lam"] = c.pow(-gamma) - lam
    res["res_labor"] = h.pow(omega) - w * lam

    res["res_XiN_rec"] = XiN - (y * w * one_plus_tau / A) - Et_XiN_next
    res["res_XiD_rec"] = XiD - y - Et_XiD_next

    res["res_pstar_def"] = pstar - (M * XiN / XiD)
    res["res_calvo"] = 1.0 - (theta * one_plus_pi.pow(eps - 1.0) + (1.0 - theta) * pstar.pow(1.0 - eps))
    res["res_Delta_law"] = Delta - (theta * one_plus_pi.pow(eps) * st.Delta_prev + (1.0 - theta) * pstar.pow(-eps))

    return res
