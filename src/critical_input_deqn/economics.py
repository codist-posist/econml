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
    """Direct equal-price imported-input share, decreasing with adaptation.

    The legacy name is kept because this object is used throughout the model,
    but under the normalized CES specification it is the share mu(A), not the
    transformed CES weight used in earlier drafts.
    """

    omega0 = min(max(float(p.omega0), 1e-8), 1.0 - 1e-8)
    omega_min = min(max(float(p.target_min_import_cost_share), 1e-8), omega0 * (1.0 - 1e-10))
    return omega_min + (omega0 - omega_min) * torch.exp(-float(p.kappa_a) * A)


def external_conditions(st: State, p: BaselineParams) -> Tuple[torch.Tensor, torch.Tensor]:
    pm = float(p.bar_p_m) * torch.exp(float(p.nu_pD) * st.D - float(p.nu_pX) * st.X)
    mbar = float(p.bar_m) * torch.exp(-float(p.nu_qD) * st.D + float(p.nu_qX) * st.X)
    return pm, mbar


def unit_intermediate_price(A: torch.Tensor, p_m_eff: torch.Tensor, p_d: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    rho = float(p.rho)
    mu = omega_import(A, p)
    term = mu * p_m_eff.pow(1.0 - rho) + (1.0 - mu) * p_d.pow(1.0 - rho)
    return term.pow(1.0 / (1.0 - rho))


def marginal_cost(w: torch.Tensor, p_x: torch.Tensor, Z: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    alpha = float(p.alpha)
    return (1.0 / Z) * (w / (1.0 - alpha)).pow(1.0 - alpha) * (p_x / alpha).pow(alpha)


def implied_labor(
    C: torch.Tensor,
    Y: torch.Tensor,
    p_x: torch.Tensor,
    Z: torch.Tensor,
    Delta: torch.Tensor,
    p: BaselineParams,
) -> torch.Tensor:
    """Labor implied by household intratemporal optimality and firm labor demand.

    Combining w=N^varphi C^sigma with Cobb-Douglas labor demand and unit cost gives
    N^{1+alpha varphi} = Delta Y Z^{-1} C^{-alpha sigma}
    [((1-alpha) p_x)/alpha]^alpha.
    """

    alpha = float(p.alpha)
    base = (
        Delta
        * Y
        / Z
        * C.pow(-alpha * float(p.sigma))
        * (((1.0 - alpha) * p_x) / alpha).pow(alpha)
    )
    return torch.clamp(base, min=1e-16).pow(1.0 / (1.0 + alpha * float(p.varphi)))


def p_x_derivative_A(A: torch.Tensor, p_m_eff: torch.Tensor, p_d: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    rho = float(p.rho)
    mu = omega_import(A, p)
    omega0 = min(max(float(p.omega0), 1e-8), 1.0 - 1e-8)
    omega_min = min(max(float(p.target_min_import_cost_share), 1e-8), omega0 * (1.0 - 1e-10))
    mu_A = -float(p.kappa_a) * (mu - omega_min)
    H = mu * p_m_eff.pow(1.0 - rho) + (1.0 - mu) * p_d.pow(1.0 - rho)
    H_A = mu_A * (p_m_eff.pow(1.0 - rho) - p_d.pow(1.0 - rho))
    p_x = H.pow(1.0 / (1.0 - rho))
    return p_x * H_A / ((1.0 - rho) * H)


def mc_derivative_A(mc: torch.Tensor, p_x: torch.Tensor, p_x_A: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return mc * float(p.alpha) * p_x_A / p_x


def desired_import_given_rent(
    st: State,
    C: torch.Tensor,
    Y: torch.Tensor,
    Delta: torch.Tensor,
    pm: torch.Tensor,
    p_d: torch.Tensor,
    chi: torch.Tensor,
    p: BaselineParams,
) -> torch.Tensor:
    """Desired imported-input demand at a candidate scarcity rent."""

    Z = torch.exp(st.log_Z)
    p_m_eff = pm + chi
    p_x0 = unit_intermediate_price(st.A, p_m_eff, p_d, p)
    N0 = implied_labor(C, Y, p_x0, Z, Delta, p)
    Lambda = C.pow(-float(p.sigma))
    w0 = N0.pow(float(p.varphi)) / Lambda
    mc0 = marginal_cost(w0, p_x0, Z, p)
    X0 = float(p.alpha) * mc0 * Delta * Y / p_x0
    mu = omega_import(st.A, p)
    return X0 * mu * (p_x0 / p_m_eff).pow(float(p.rho))


def desired_import_at_zero_rent(
    st: State,
    C: torch.Tensor,
    Y: torch.Tensor,
    Delta: torch.Tensor,
    pm: torch.Tensor,
    p_d: torch.Tensor,
    p: BaselineParams,
) -> torch.Tensor:
    """Desired imported-input demand at chi=0, holding aggregate C,Y,Delta fixed."""

    return desired_import_given_rent(st, C, Y, Delta, pm, p_d, torch.zeros_like(C), p)


def _select_state(st: State, mask: torch.Tensor) -> State:
    return State(
        D=st.D[mask],
        X=st.X[mask],
        ell_D=st.ell_D[mask],
        ell_X=st.ell_X[mask],
        log_Z=st.log_Z[mask],
        A=st.A[mask],
        log_Delta_prev=None if st.log_Delta_prev is None else st.log_Delta_prev[mask],
    )


def solve_import_rent(
    st: State,
    C: torch.Tensor,
    Y: torch.Tensor,
    Delta: torch.Tensor,
    pm: torch.Tensor,
    mbar: torch.Tensor,
    p_d: torch.Tensor,
    p: BaselineParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the one-dimensional imported-input MCP for the scarcity rent.

    If zero-rent desired demand is below the cap, the rent is exactly zero. If
    desired demand exceeds the cap, bisection finds the rent that makes desired
    demand equal available capacity.
    """

    M_zero = desired_import_at_zero_rent(st, C, Y, Delta, pm, p_d, p)
    bind = M_zero > mbar
    chi = torch.zeros_like(C)
    M_at_rent = M_zero.clone()
    if bool(bind.detach().any().cpu()):
        st_b = _select_state(st, bind)
        C_b = C[bind]
        Y_b = Y[bind]
        Delta_b = Delta[bind]
        pm_b = pm[bind]
        mbar_b = mbar[bind]
        p_d_b = p_d[bind]

        lo = torch.zeros_like(C_b)
        hi = torch.clamp(pm_b, min=1.0)
        for _ in range(24):
            M_hi = desired_import_given_rent(st_b, C_b, Y_b, Delta_b, pm_b, p_d_b, hi, p)
            hi = torch.where(M_hi > mbar_b, 2.0 * hi + 1e-8, hi)
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            M_mid = desired_import_given_rent(st_b, C_b, Y_b, Delta_b, pm_b, p_d_b, mid, p)
            tight = M_mid > mbar_b
            lo = torch.where(tight, mid, lo)
            hi = torch.where(tight, hi, mid)
        M_hi = desired_import_given_rent(st_b, C_b, Y_b, Delta_b, pm_b, p_d_b, hi, p)
        chi = chi.index_put((bind,), hi)
        M_at_rent = M_at_rent.index_put((bind,), M_hi)
    return chi, M_zero, M_at_rent


def input_static_quantities(
    st: State,
    C: torch.Tensor,
    Y: torch.Tensor,
    Delta: torch.Tensor,
    pm: torch.Tensor,
    chi: torch.Tensor,
    p_d: torch.Tensor,
    p: BaselineParams,
) -> Dict[str, torch.Tensor]:
    """Static input, labor, and marginal-cost objects implied by C,Y,Delta,chi."""

    p_m_eff = pm + chi
    Z = torch.exp(st.log_Z)
    p_x = unit_intermediate_price(st.A, p_m_eff, p_d, p)
    N = implied_labor(C, Y, p_x, Z, Delta, p)
    Lambda = C.pow(-float(p.sigma))
    w = N.pow(float(p.varphi)) / Lambda
    mc = marginal_cost(w, p_x, Z, p)
    X_comp = float(p.alpha) * mc * Delta * Y / p_x
    mu = omega_import(st.A, p)
    M = X_comp * mu * (p_x / p_m_eff).pow(float(p.rho))
    S = X_comp * (1.0 - mu) * (p_x / p_d).pow(float(p.rho))
    N_d = (1.0 - float(p.alpha)) * mc * Delta * Y / w
    return {
        "p_m_eff": p_m_eff,
        "Z": Z,
        "p_x": p_x,
        "Lambda": Lambda,
        "w": w,
        "mc": mc,
        "N": N,
        "X_comp": X_comp,
        "M": M,
        "S": S,
        "N_d": N_d,
    }


def psi(I: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return float(p.psi_A) * I + 0.5 * float(p.phi_A) * I.pow(2)


def psi_prime(I: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    return float(p.psi_A) + float(p.phi_A) * I


def adaptation_enabled(p: BaselineParams) -> bool:
    return float(p.adaptation_enabled) > 0.5


def bounded_repair_investment(Q_A: torch.Tensor, Omega_A: torch.Tensor, p_a: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    """Repair investment implied by the bounded private repair KKT."""

    if adaptation_enabled(p):
        marginal_cost = torch.clamp(Omega_A * p_a, min=1e-12)
        interior = (Q_A / marginal_cost - float(p.psi_A)) / float(p.phi_A)
        return torch.clamp(interior, min=0.0, max=float(p.repair_capacity))
    return torch.zeros_like(Q_A)


def effective_repair_investment(I: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    if adaptation_enabled(p):
        return torch.clamp(I, min=0.0, max=float(p.repair_capacity))
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
    C, Y = out["C"], out["Y"]
    Pi = out["Pi"]
    S_p, F_p = out["S_p"], out["F_p"]
    pm, mbar = external_conditions(st, p)
    p_d = torch.full_like(C, float(p.p_d))
    p_a = torch.full_like(C, float(p.p_a))
    Delta_prev = torch.exp(st.log_Delta_prev)

    p_star = (float(p.epsilon) / (float(p.epsilon) - 1.0)) * S_p / F_p
    Delta = (1.0 - float(p.theta)) * p_star.pow(-float(p.epsilon)) + float(p.theta) * Pi.pow(float(p.epsilon)) * Delta_prev

    policy_key = policy.lower()
    if policy_key == "fixed":
        intercept = torch.full_like(C, float(p.bar_R))
    elif policy_key == "ba":
        if R_n is None:
            raise ValueError("BA policy requires R_n.")
        intercept = float(p.bar_pi) * R_n
    elif policy_key in {"bottleneck", "repair_aware"}:
        intercept = torch.full_like(C, float(p.bar_R))
    else:
        raise ValueError("policy must be 'fixed', 'ba', 'bottleneck', or 'repair_aware'.")

    R_standard = intercept * (Pi / float(p.bar_pi)).pow(float(p.phi_pi)) * (Y / Y_n).pow(float(p.phi_y))

    bottleneck_scarcity = torch.zeros_like(C)
    bottleneck_adjustment = torch.ones_like(C)
    cap_pressure_policy = torch.zeros_like(C)
    if policy_key == "bottleneck":
        M_zero_policy = desired_import_at_zero_rent(st, C, Y, Delta, pm, p_d, p)
        cap_pressure_policy = M_zero_policy / torch.clamp(mbar, min=1e-12)
        bottleneck_scarcity = torch.relu(torch.log(torch.clamp(cap_pressure_policy, min=1e-12)))
        bottleneck_adjustment = torch.exp(-float(p.phi_bottleneck) * bottleneck_scarcity)

    repair_margin_standard = torch.zeros_like(C)
    repair_margin_support = torch.zeros_like(C)
    repair_cap_support = torch.zeros_like(C)
    repair_support_raw = torch.zeros_like(C)
    repair_support = torch.zeros_like(C)
    repair_adjustment = torch.ones_like(C)
    if policy_key == "repair_aware":
        M_zero_policy = desired_import_at_zero_rent(st, C, Y, Delta, pm, p_d, p)
        cap_pressure_policy = M_zero_policy / torch.clamp(mbar, min=1e-12)
        Omega_standard = omega_A_cost(R_standard, p)
        repair_threshold_standard = torch.clamp(Omega_standard * p_a * float(p.psi_A), min=1e-12)
        repair_margin_standard = out["Q_A"] / repair_threshold_standard
        repair_margin_support = torch.relu(
            torch.log(
                torch.clamp(
                    repair_margin_standard / max(float(p.repair_margin_trigger), 1e-12),
                    min=1e-12,
                )
            )
        )
        repair_cap_support = torch.relu(
            torch.log(
                torch.clamp(
                    cap_pressure_policy / max(float(p.repair_cap_pressure_trigger), 1e-12),
                    min=1e-12,
                )
            )
        )
        repair_support_raw = repair_margin_support * repair_cap_support
        support_max = max(float(p.repair_support_max), 1e-12)
        repair_support = support_max * torch.tanh(repair_support_raw / support_max)
        repair_adjustment = torch.exp(-float(p.phi_repair) * repair_support)

    R = R_standard * bottleneck_adjustment * repair_adjustment
    Omega_A = omega_A_cost(R, p)
    I = bounded_repair_investment(out["Q_A"], Omega_A, p_a, p)
    A_next = (1.0 - float(p.delta_A)) * st.A + I
    chi, M_zero_rent, M_at_rent = solve_import_rent(st, C, Y, Delta, pm, mbar, p_d, p)
    static = input_static_quantities(st, C, Y, Delta, pm, chi, p_d, p)

    return {
        "pm": pm,
        "mbar": mbar,
        "chi": chi,
        "p_m_eff": static["p_m_eff"],
        "p_d": p_d,
        "p_a": p_a,
        "Z": static["Z"],
        "Delta_prev": Delta_prev,
        "p_x": static["p_x"],
        "Lambda": static["Lambda"],
        "w": static["w"],
        "mc": static["mc"],
        "N": static["N"],
        "p_star": p_star,
        "Delta": Delta,
        "X_comp": static["X_comp"],
        "M": static["M"],
        "M_zero_rent": M_zero_rent,
        "M_at_rent": M_at_rent,
        "S": static["S"],
        "N_d": static["N_d"],
        "I_A": I,
        "I_A_effective": I,
        "A_next": A_next,
        "R": R,
        "R_standard": R_standard,
        "Omega_A": Omega_A,
        "cap_pressure_policy": cap_pressure_policy,
        "bottleneck_scarcity": bottleneck_scarcity,
        "bottleneck_adjustment": bottleneck_adjustment,
        "repair_margin_standard": repair_margin_standard,
        "repair_margin_support": repair_margin_support,
        "repair_cap_support": repair_cap_support,
        "repair_support_raw": repair_support_raw,
        "repair_support": repair_support,
        "repair_adjustment": repair_adjustment,
    }


def derive_free(
    st: State,
    out: Dict[str, torch.Tensor],
    p: BaselineParams,
    *,
    R: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Derived objects for optimal policies.

    If ``R`` is omitted, the function falls back to ``out["R"]`` for backward
    compatibility with old checkpoints.  New discretion/commitment policies
    pass the Euler-implied gross policy rate explicitly.
    """

    C, Y = out["C"], out["Y"]
    if R is None:
        R = out["R"]
    Pi = out["Pi"]
    S_p, F_p = out["S_p"], out["F_p"]
    pm, mbar = external_conditions(st, p)
    p_d = torch.full_like(C, float(p.p_d))
    p_a = torch.full_like(C, float(p.p_a))
    Delta_prev = torch.exp(st.log_Delta_prev)

    p_star = (float(p.epsilon) / (float(p.epsilon) - 1.0)) * S_p / F_p
    Delta = (1.0 - float(p.theta)) * p_star.pow(-float(p.epsilon)) + float(p.theta) * Pi.pow(float(p.epsilon)) * Delta_prev
    Omega_A = omega_A_cost(R, p)
    I = bounded_repair_investment(out["Q_A"], Omega_A, p_a, p)
    A_next = (1.0 - float(p.delta_A)) * st.A + I
    chi, M_zero_rent, M_at_rent = solve_import_rent(st, C, Y, Delta, pm, mbar, p_d, p)
    static = input_static_quantities(st, C, Y, Delta, pm, chi, p_d, p)

    return {
        "pm": pm,
        "mbar": mbar,
        "chi": chi,
        "p_m_eff": static["p_m_eff"],
        "p_d": p_d,
        "p_a": p_a,
        "Z": static["Z"],
        "Delta_prev": Delta_prev,
        "p_x": static["p_x"],
        "Lambda": static["Lambda"],
        "w": static["w"],
        "mc": static["mc"],
        "N": static["N"],
        "p_star": p_star,
        "Delta": Delta,
        "X_comp": static["X_comp"],
        "M": static["M"],
        "M_zero_rent": M_zero_rent,
        "M_at_rent": M_at_rent,
        "S": static["S"],
        "N_d": static["N_d"],
        "I_A": I,
        "I_A_effective": I,
        "A_next": A_next,
        "R": R,
        "Omega_A": Omega_A,
    }


def derive_natural(st: State, out: Dict[str, torch.Tensor], p: BaselineParams) -> Dict[str, torch.Tensor]:
    C, Y = out["C_n"], out["Y_n"]
    pm, mbar = external_conditions(st, p)
    p_d = torch.full_like(C, float(p.p_d))
    Delta = torch.ones_like(C)
    chi, M_zero_rent, M_at_rent = solve_import_rent(st, C, Y, Delta, pm, mbar, p_d, p)
    static = input_static_quantities(st, C, Y, Delta, pm, chi, p_d, p)
    return {
        "pm": pm,
        "mbar": mbar,
        "chi_n": chi,
        "chi": chi,
        "p_m_eff": static["p_m_eff"],
        "p_d": p_d,
        "Z": static["Z"],
        "p_x": static["p_x"],
        "Lambda": static["Lambda"],
        "w": static["w"],
        "mc": static["mc"],
        "N": static["N"],
        "X_comp": static["X_comp"],
        "M": static["M"],
        "M_zero_rent": M_zero_rent,
        "M_at_rent": M_at_rent,
        "S": static["S"],
        "N_d": static["N_d"],
    }
