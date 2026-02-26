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
    ids = identities(params, st, out)
    A = ids["A"]
    g = ids["g"]
    y = ids["y"]
    h = ids["h"]
    one_plus_tau = ids["one_plus_tau"]

    c = out["c"]; pi = out["pi"]; pstar = out["pstar"]
    lam = out["lam"]; w = out["w"]
    XiN = out["XiN"]; XiD = out["XiD"]; Delta = out["Delta"]
    mu = out["mu"]; rho = out["rho"]; zeta = out["zeta"]

    # safe-by-construction inflation object
    one_plus_pi = out.get("one_plus_pi", 1.0 + pi)

    eps = params.eps; theta = params.theta; beta = params.beta; M = params.M
    gamma = params.gamma; omega = params.omega

    res: Dict[str, torch.Tensor] = {}

    # ---- FOC w.r.t. c (A.2) ----
    term1 = c.pow(-gamma) - (c + g).pow(omega) * (Delta / A).pow(1.0 + omega)
    term2 = mu * (
        pstar * (1.0 + gamma * c.pow(gamma - 1.0) * Et_F_next)
        - M * (
            ((1.0 + omega) * c + gamma * (c + g))
            * (c + g).pow(omega) * c.pow(gamma - 1.0)
            * (Delta / A).pow(omega) * one_plus_tau / A
            + gamma * c.pow(gamma - 1.0) * Et_G_next
        )
    )
    res["res_c_foc"] = term1 + term2

    # ---- FOC w.r.t. π (eq. (27) in Appendix A.2) ----
    res["res_pi_foc"] = (
        rho * theta * (eps - 1.0) * one_plus_pi.pow(eps - 2.0)
        + zeta * theta * eps * one_plus_pi.pow(eps - 1.0) * st.Delta_prev
    )

    # ---- FOC w.r.t. p* ----
    res["res_pstar_foc"] = (
        mu * XiD
        + rho * (1.0 - theta) * (1.0 - eps) * pstar.pow(-eps)
        - zeta * (1.0 - theta) * eps * pstar.pow(-eps - 1.0)
    )

    # ---- FOC w.r.t. Δ ----
    disutil_over_Delta = -((c + g) * Delta / A).pow(1.0 + omega) / Delta
    expect_term = beta * Et_theta_zeta_pi_next - zeta
    mu_bracket = (
        pstar * c.pow(gamma) * Et_dF_dDelta_next
        - M * (
            (c + g).pow(1.0 + omega) * c.pow(gamma)
            * (omega / Delta) * (Delta / A).pow(omega) * one_plus_tau / A
            + c.pow(gamma) * Et_dG_dDelta_next
        )
    )
    res["res_Delta_foc"] = disutil_over_Delta + expect_term + mu * mu_bracket

    # ---- Equilibrium conditions block (same as A.1) ----
    res["res_c_lam"] = c.pow(-gamma) - lam
    res["res_labor"] = h.pow(omega) - w * lam
    res["res_XiN_rec"] = XiN - (y * w * one_plus_tau / A) - Et_XiN_next
    res["res_XiD_rec"] = XiD - y - Et_XiD_next
    res["res_pstar_def"] = pstar - (M * XiN / XiD)

    res["res_calvo"] = 1.0 - (
        theta * one_plus_pi.pow(eps - 1.0)
        + (1.0 - theta) * pstar.pow(1.0 - eps)
    )
    res["res_Delta_law"] = Delta - (
        theta * one_plus_pi.pow(eps) * st.Delta_prev
        + (1.0 - theta) * pstar.pow(-eps)
    )

    return res
