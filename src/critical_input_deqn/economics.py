from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .config import BaselineParams


@dataclass(frozen=True)
class State:
    D: torch.Tensor
    X: torch.Tensor
    ell_D: torch.Tensor
    ell_X: torch.Tensor
    log_Z: torch.Tensor
    A: torch.Tensor
    log_Delta_prev: torch.Tensor | None = None


def unpack_rule_state(z: torch.Tensor) -> State:
    return State(
        D=z[..., 0],
        X=z[..., 1],
        ell_D=z[..., 2],
        ell_X=z[..., 3],
        log_Z=z[..., 4],
        A=z[..., 5],
        log_Delta_prev=z[..., 6],
    )


def unpack_natural_state(z: torch.Tensor) -> State:
    return State(
        D=z[..., 0],
        X=z[..., 1],
        ell_D=z[..., 2],
        ell_X=z[..., 3],
        log_Z=z[..., 4],
        A=z[..., 5],
        log_Delta_prev=None,
    )


def omega_import(A: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return float(p.omega0) * torch.exp(-float(p.kappa_a) * A)


def external_conditions(st: State, p: BaselineParams) -> Tuple[torch.Tensor, torch.Tensor]:
    pm = float(p.bar_p_m) * torch.exp(float(p.nu_pD) * st.D - float(p.nu_pX) * st.X)
    mbar = float(p.bar_m) * torch.exp(-float(p.nu_qD) * st.D + float(p.nu_qX) * st.X)
    return pm, mbar


def unit_intermediate_price(A: torch.Tensor, p_m_eff: torch.Tensor, p_d: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    rho = float(p.rho)
    omega = omega_import(A, p)
    term = omega.pow(rho) * p_m_eff.pow(1.0 - rho) + (1.0 - omega).pow(rho) * p_d.pow(1.0 - rho)
    return term.pow(1.0 / (1.0 - rho))


def marginal_cost(w: torch.Tensor, p_x: torch.Tensor, Z: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    alpha = float(p.alpha)
    return (1.0 / Z) * (w / (1.0 - alpha)).pow(1.0 - alpha) * (p_x / alpha).pow(alpha)


def p_x_derivative_A(A: torch.Tensor, p_m_eff: torch.Tensor, p_d: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    rho = float(p.rho)
    omega = omega_import(A, p)
    omega_A = -float(p.kappa_a) * omega
    F = omega.pow(rho) * p_m_eff.pow(1.0 - rho) + (1.0 - omega).pow(rho) * p_d.pow(1.0 - rho)
    F_A = rho * omega_A * (
        omega.pow(rho - 1.0) * p_m_eff.pow(1.0 - rho)
        - (1.0 - omega).pow(rho - 1.0) * p_d.pow(1.0 - rho)
    )
    p_x = F.pow(1.0 / (1.0 - rho))
    return p_x * F_A / ((1.0 - rho) * F)


def mc_derivative_A(mc: torch.Tensor, p_x: torch.Tensor, p_x_A: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return mc * float(p.alpha) * p_x_A / p_x


def psi(I: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return I + 0.5 * float(p.phi_A) * I.pow(2)


def psi_prime(I: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return 1.0 + float(p.phi_A) * I


def adaptation_enabled(p: BaselineParams) -> bool:
    return float(p.adaptation_enabled) > 0.5


def effective_repair_investment(I: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    if adaptation_enabled(p):
        return I
    return torch.zeros_like(I)


def omega_A_cost(R: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return (1.0 - float(p.vartheta_A)) + float(p.vartheta_A) * R


def fischer_burmeister(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(a.pow(2) + b.pow(2) + float(eps) ** 2) - a - b


def derive_rule(
    st: State,
    out: Dict[str, torch.Tensor],
    p: BaselineParams,
    *,
    Y_n: torch.Tensor,
    R_n: torch.Tensor | None,
    policy: str,
) -> Dict[str, torch.Tensor]:
    C, Y, N = out["C"], out["Y"], out["N"]
    Pi, chi, I = out["Pi"], out["chi"], effective_repair_investment(out["I_A"], p)
    S_p, F_p = out["S_p"], out["F_p"]
    pm, mbar = external_conditions(st, p)
    p_m_eff = pm + chi
    p_d = torch.full_like(C, float(p.p_d))
    p_a = torch.full_like(C, float(p.p_a))
    Z = torch.exp(st.log_Z)
    Delta_prev = torch.exp(st.log_Delta_prev)

    p_x = unit_intermediate_price(st.A, p_m_eff, p_d, p)
    Lambda = C.pow(-float(p.sigma))
    w = N.pow(float(p.varphi)) / Lambda
    mc = marginal_cost(w, p_x, Z, p)
    p_star = (float(p.epsilon) / (float(p.epsilon) - 1.0)) * S_p / F_p
    Delta = (1.0 - float(p.theta)) * p_star.pow(-float(p.epsilon)) + float(p.theta) * Pi.pow(float(p.epsilon)) * Delta_prev

    X_comp = float(p.alpha) * mc * Delta * Y / p_x
    omega = omega_import(st.A, p)
    M = X_comp * omega.pow(float(p.rho)) * (p_x / p_m_eff).pow(float(p.rho))
    S = X_comp * (1.0 - omega).pow(float(p.rho)) * (p_x / p_d).pow(float(p.rho))
    N_d = (1.0 - float(p.alpha)) * mc * Delta * Y / w
    A_next = (1.0 - float(p.delta_A)) * st.A + I

    if policy.lower() == "fixed":
        intercept = torch.full_like(C, float(p.bar_R))
    elif policy.lower() == "ba":
        if R_n is None:
            raise ValueError("BA policy requires R_n.")
        intercept = float(p.bar_pi) * R_n
    else:
        raise ValueError("policy must be 'fixed' or 'ba'.")
    R = intercept * (Pi / float(p.bar_pi)).pow(float(p.phi_pi)) * (Y / Y_n).pow(float(p.phi_y))

    return {
        "pm": pm,
        "mbar": mbar,
        "p_m_eff": p_m_eff,
        "p_d": p_d,
        "p_a": p_a,
        "Z": Z,
        "Delta_prev": Delta_prev,
        "p_x": p_x,
        "Lambda": Lambda,
        "w": w,
        "mc": mc,
        "p_star": p_star,
        "Delta": Delta,
        "X_comp": X_comp,
        "M": M,
        "S": S,
        "N_d": N_d,
        "I_A_effective": I,
        "A_next": A_next,
        "R": R,
        "Omega_A": omega_A_cost(R, p),
    }


def derive_free(
    st: State,
    out: Dict[str, torch.Tensor],
    p: BaselineParams,
) -> Dict[str, torch.Tensor]:
    """Derived objects when the gross policy rate is an implementability variable."""

    C, Y, N = out["C"], out["Y"], out["N"]
    R, Pi, chi, I = out["R"], out["Pi"], out["chi"], effective_repair_investment(out["I_A"], p)
    S_p, F_p = out["S_p"], out["F_p"]
    pm, mbar = external_conditions(st, p)
    p_m_eff = pm + chi
    p_d = torch.full_like(C, float(p.p_d))
    p_a = torch.full_like(C, float(p.p_a))
    Z = torch.exp(st.log_Z)
    Delta_prev = torch.exp(st.log_Delta_prev)

    p_x = unit_intermediate_price(st.A, p_m_eff, p_d, p)
    Lambda = C.pow(-float(p.sigma))
    w = N.pow(float(p.varphi)) / Lambda
    mc = marginal_cost(w, p_x, Z, p)
    p_star = (float(p.epsilon) / (float(p.epsilon) - 1.0)) * S_p / F_p
    Delta = (1.0 - float(p.theta)) * p_star.pow(-float(p.epsilon)) + float(p.theta) * Pi.pow(float(p.epsilon)) * Delta_prev

    X_comp = float(p.alpha) * mc * Delta * Y / p_x
    omega = omega_import(st.A, p)
    M = X_comp * omega.pow(float(p.rho)) * (p_x / p_m_eff).pow(float(p.rho))
    S = X_comp * (1.0 - omega).pow(float(p.rho)) * (p_x / p_d).pow(float(p.rho))
    N_d = (1.0 - float(p.alpha)) * mc * Delta * Y / w
    A_next = (1.0 - float(p.delta_A)) * st.A + I

    return {
        "pm": pm,
        "mbar": mbar,
        "p_m_eff": p_m_eff,
        "p_d": p_d,
        "p_a": p_a,
        "Z": Z,
        "Delta_prev": Delta_prev,
        "p_x": p_x,
        "Lambda": Lambda,
        "w": w,
        "mc": mc,
        "p_star": p_star,
        "Delta": Delta,
        "X_comp": X_comp,
        "M": M,
        "S": S,
        "N_d": N_d,
        "I_A_effective": I,
        "A_next": A_next,
        "R": R,
        "Omega_A": omega_A_cost(R, p),
    }


def derive_natural(st: State, out: Dict[str, torch.Tensor], p: BaselineParams) -> Dict[str, torch.Tensor]:
    C, Y, N, chi = out["C_n"], out["Y_n"], out["N_n"], out["chi_n"]
    pm, mbar = external_conditions(st, p)
    p_m_eff = pm + chi
    p_d = torch.full_like(C, float(p.p_d))
    Z = torch.exp(st.log_Z)
    p_x = unit_intermediate_price(st.A, p_m_eff, p_d, p)
    Lambda = C.pow(-float(p.sigma))
    w = N.pow(float(p.varphi)) / Lambda
    mc = marginal_cost(w, p_x, Z, p)
    X_comp = float(p.alpha) * mc * Y / p_x
    omega = omega_import(st.A, p)
    M = X_comp * omega.pow(float(p.rho)) * (p_x / p_m_eff).pow(float(p.rho))
    S = X_comp * (1.0 - omega).pow(float(p.rho)) * (p_x / p_d).pow(float(p.rho))
    N_d = (1.0 - float(p.alpha)) * mc * Y / w
    return {
        "pm": pm,
        "mbar": mbar,
        "p_m_eff": p_m_eff,
        "p_d": p_d,
        "Z": Z,
        "p_x": p_x,
        "Lambda": Lambda,
        "w": w,
        "mc": mc,
        "X_comp": X_comp,
        "M": M,
        "S": S,
        "N_d": N_d,
    }
