from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Callable
import numpy as np
import torch

from .config import ModelParams
from .sss_from_policy import PolicySSS, switching_policy_sss_by_regime_from_policy
from .policy_rules import i_taylor, i_modified_taylor


def _robust_solve(J: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Robust linear solve for Newton steps (handles near-singular Jacobians).

    We need this to be numerically stable because both commitment and discretion
    switching-SSS solvers rely on implicit-function / Newton updates.

    Strategy:
      1) try direct solve; if it produces non-finite values, treat as failure
      2) try Tikhonov-regularized solves (J + λI) with increasing λ
      3) try normal-equations ridge solve: (JᵀJ + λI) x = Jᵀ b
      4) fall back to least-squares solution
    """
    # 1) direct solve
    try:
        x = torch.linalg.solve(J, b)
        if torch.isfinite(x).all():
            return x
    except Exception:
        pass

    n = J.shape[0]
    I = torch.eye(n, dtype=J.dtype, device=J.device)

    # 2) (J + λI) x = b
    lam = torch.tensor(1e-12, dtype=J.dtype, device=J.device)
    for _ in range(12):
        try:
            x = torch.linalg.solve(J + lam * I, b)
            if torch.isfinite(x).all():
                return x
        except Exception:
            pass
        lam = lam * 10.0

    # 3) normal equations ridge
    JTJ = J.T @ J
    JTb = J.T @ b
    lam = torch.tensor(1e-12, dtype=J.dtype, device=J.device)
    for _ in range(12):
        try:
            x = torch.linalg.solve(JTJ + lam * I, JTb)
            if torch.isfinite(x).all():
                return x
        except Exception:
            pass
        lam = lam * 10.0

    # 4) least squares fallback
    return torch.linalg.lstsq(J, b).solution


@dataclass
class FlexSSS:
    by_regime: Dict[int, Dict[str, float]]



def solve_efficient_sss(params: ModelParams, max_iter: int = 20000, tol: float = 1e-14) -> Dict[str, float]:
    """Efficient (planner) non-stochastic steady state, per paper eq. (??) around page 12.

    With A=1 and g=g_bar, efficient consumption ĉ solves:
        ((ĉ + g_bar)/A)^omega - A * ĉ^{-gamma} = 0
    which at A=1 simplifies to:
        (ĉ + g_bar)^omega - ĉ^{-gamma} = 0

    Returns a dict with:
      - c_hat: ĉ
      - r_hat: 1/beta - 1  (non-stochastic efficient real rate)
    """
    g = float(params.g_bar)
    omega = float(params.omega)
    gamma = float(params.gamma)

    # Newton on scalar ĉ
    c = 0.9  # good generic starting point for this calibration
    for _ in range(max_iter):
        val = (c + g) ** omega - (c ** (-gamma))
        # derivative: omega*(c+g)^(omega-1) + gamma*c^(-gamma-1)
        dval = omega * (c + g) ** (omega - 1.0) + gamma * (c ** (-gamma - 1.0))
        step = val / dval
        c_new = max(1e-12, c - step)
        if abs(c_new - c) < float(tol):
            c = c_new
            break
        c = c_new

    return {"c_hat": float(c), "r_hat": float(1.0 / params.beta - 1.0)}


def solve_flexprice_sss(params: ModelParams, max_iter: int = 20000, tol: float = 1e-14) -> FlexSSS:
    """
    Flexible-price (star) allocation per regime, consistent with eq. (14) in the paper.

    In the (A=1, g=g_bar, xi=0) regime-conditional stochastic steady state, the flexible-price allocation solves:
        (c + g_bar)^omega - 1 / ((1+tau_s) * M * c^gamma) = 0
    where:
        (1+tau_s) = 1 - tau_bar + eta_s
        eta_s = 0 in normal (s=0), eta_bar in bad (s=1)
        M = eps/(eps-1)

    IMPORTANT: The second term is the reciprocal, exactly as written in the paper.
    """
    M = float(params.M)
    gbar = float(params.g_bar)
    gamma = float(params.gamma)
    omega = float(params.omega)

    out: Dict[int, Dict[str, float]] = {}
    for s in [0, 1]:
        eta_s = 0.0 if s == 0 else float(params.eta_bar)
        one_plus_tau = (1.0 - float(params.tau_bar)) + eta_s  # xi=0 in SSS

        def f(c: float) -> float:
            return (c + gbar) ** omega - 1.0 / (one_plus_tau * M * (c ** gamma))

        def df(c: float) -> float:
            return omega * (c + gbar) ** (omega - 1.0) + gamma / (one_plus_tau * M * (c ** (gamma + 1.0)))

        c = 1.0
        for _ in range(int(max_iter)):
            val = f(c)
            dval = df(c)
            if abs(dval) < 1e-18:
                break
            c_new = max(1e-12, c - val / dval)
            if abs(c_new - c) < float(tol):
                c = c_new
                break
            c = c_new

        # Under flex prices (p* = 1, pi = 0, Delta = 1), the remaining objects are pinned down by
        # the paper's static equilibrium conditions:
        #   y = c + g_bar,
        #   h = y * Delta / A  (with A=1, Delta=1)  => h = c + g_bar,
        #   w = A / (M (1+tau_s)),
        #   lambda = c^{-gamma}.
        h = float(c + gbar)
        w = float(1.0 / (one_plus_tau * M))
        lam = float(c ** (-gamma))

        out[s] = {
            "c": float(c),
            "h": float(h),
            "w": float(w),
            "lam": float(lam),
            "pi": 0.0,
            "Delta": 1.0,
            "pstar": 1.0,
        }

# Natural rate in each regime (flex-price SSS): r*_s = 1/(beta * E_s[m_{t+1}]) - 1,
    # where m_{t+1} = (c_{t+1}/c_t)^(-gamma) and expectation is over s_{t+1} only (Markov).
    P = params.P.detach().cpu().numpy()
    c0, c1 = out[0]["c"], out[1]["c"]

    def r_star(s: int) -> float:
        c_s = c0 if s == 0 else c1
        pi0 = float(P[s, 0])
        pi1 = float(P[s, 1])
        # E[(c_{t+1}/c_t)^(-gamma)] over next regime only
        E_m = pi0 * ((c0 / c_s) ** (-gamma)) + pi1 * ((c1 / c_s) ** (-gamma))
        return 1.0 / (float(params.beta) * E_m) - 1.0

    out[0]["r_star"] = float(r_star(0))
    out[1]["r_star"] = float(r_star(1))
    return FlexSSS(by_regime=out)


@dataclass
class TaylorSSS:
    by_regime: Dict[int, Dict[str, float]]


def solve_taylor_sss(params: ModelParams, flex: FlexSSS) -> TaylorSSS:
    """
    SSS under the Taylor rule (paper eq. (25)), regime by regime.

    Uses r_star(s) from the flex-price allocation as the regime-dependent natural rate r_{s,ss}.
    The Taylor-rule intercept targets:
        rbar = 1/beta - 1    (efficient non-stochastic SS real rate, i.e. hat{r})
    and:
        pi_ss = pi_bar + (r_ss - rbar)/(psi - 1)
    """
    psi = float(params.psi)
    if abs(psi - 1.0) < 1e-12:
        raise ValueError("Taylor-rule slope psi must differ from 1 for eq. (25).")

    # FIX (2): rbar is real-rate target, not plus pi_bar.
    rbar = (1.0 / float(params.beta)) - 1.0

    out: Dict[int, Dict[str, float]] = {}
    for s in [0, 1]:
        r_ss = float(flex.by_regime[s]["r_star"])
        pi_ss = float(params.pi_bar + (r_ss - rbar) / (psi - 1.0))

        # i_ss computed with the exact same Taylor rule function used elsewhere.
        i_ss = float(
            i_taylor(
                params,
                torch.tensor(pi_ss, dtype=params.dtype, device=params.device),
            ).detach().cpu()
        )

        # Calvo SS objects implied by pi_ss
        theta = float(params.theta)
        eps = float(params.eps)
        one_plus_pi = 1.0 + pi_ss

        # price index: 1 = theta(1+pi)^{eps-1} + (1-theta)pstar^{1-eps}
        base = (1.0 - theta * (one_plus_pi ** (eps - 1.0))) / (1.0 - theta)
        if base <= 0:
            raise ValueError(f"Taylor SSS infeasible in regime {s}: implied base<=0 (pi_ss out of feasible region).")
        pstar = base ** (1.0 / (1.0 - eps))

        # Delta law in SS: Delta = (1-theta)pstar^{-eps} / (1 - theta(1+pi)^eps)
        denom = 1.0 - theta * (one_plus_pi ** eps)
        if denom <= 0:
            raise ValueError(f"Taylor SSS infeasible in regime {s}: denom<=0 in Delta law.")
        Delta = (1.0 - theta) * (pstar ** (-eps)) / denom

        out[s] = {"pi": pi_ss, "i": i_ss, "r": r_ss, "pstar": float(pstar), "Delta": float(Delta)}

    return TaylorSSS(by_regime=out)


def export_rbar_tensor(params: ModelParams, flex: FlexSSS) -> torch.Tensor:
    return torch.tensor(
        [flex.by_regime[0]["r_star"], flex.by_regime[1]["r_star"]],
        device=params.device,
        dtype=params.dtype,
    )


def _newton_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    max_iter: int = 200,
    tol: float = 1e-12,
    damping: float = 1.0,
) -> torch.Tensor:
    """Small damped Newton solver using autograd Jacobian (exact, no approximations)."""
    x = x0.clone()
    for _ in range(max_iter):
        x = x.detach().requires_grad_(True)
        r = f(x)
        norm = torch.max(torch.abs(r)).item()
        if norm < tol:
            return x.detach()
        J = torch.autograd.functional.jacobian(f, x, create_graph=False)
        dx = torch.linalg.solve(J, -r)
        x = (x + damping * dx).detach()
    return x.detach()


def commitment_local_ss(params: ModelParams, regime: int) -> Dict[str, float]:
    """
    Local (no-regime-switching) steady state for commitment (Appendix A.3 equations),
    used ONLY for principled initialization of lagged co-states (vartheta_prev, varrho_prev).

    Regime enters only through eta (0 or eta_bar), hence (1+tau).
    """
    assert regime in (0, 1)
    dev, dt = params.device, params.dtype

    g = torch.tensor(params.g_bar, device=dev, dtype=dt)
    eta = torch.tensor(0.0 if regime == 0 else params.eta_bar, device=dev, dtype=dt)
    one_plus_tau = torch.tensor(1.0 - params.tau_bar, device=dev, dtype=dt) + eta  # xi=0 in local SS
    A = torch.tensor(1.0, device=dev, dtype=dt)

    beta = params.beta
    gamma = params.gamma
    omega = params.omega
    eps = params.eps
    theta = params.theta
    M = params.M

    def unpack(u: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(u[i] for i in range(13))

    def residual(u: torch.Tensor) -> torch.Tensor:
        c, pi, pstar, lam, w, XiN, XiD, Delta, mu, nu, zeta, vartheta, varrho = unpack(u)

        # positivity by squaring (reparameterization for solver stability)
        c = c**2 + 1e-12
        Delta = Delta**2 + 1e-12
        pstar = pstar**2 + 1e-12
        XiN = XiN**2 + 1e-14
        XiD = XiD**2 + 1e-14

        y = c + g
        h = y * Delta / A

        # Recursions in SS (no switching): Lambda = beta * lam_next/lam = beta
        Et_XiN = theta * beta * (1.0 + pi) ** eps * XiN
        Et_XiD = theta * beta * (1.0 + pi) ** (eps - 1.0) * XiD

        # Terms used in A.3 (SS specialization)
        Et_termN = beta * theta * gamma * c.pow(gamma - 1.0) * c.pow(-gamma) * (1.0 + pi) ** eps * XiN
        Et_termD = beta * theta * gamma * c.pow(gamma - 1.0) * c.pow(-gamma) * (1.0 + pi) ** (eps - 1.0) * XiD
        Et_theta_zeta_pi = theta * (1.0 + pi) ** eps * zeta

        # scaled lag co-states in our DEQN state convention: vp = vartheta * c^gamma ; rp similarly
        vp = vartheta * c.pow(gamma)
        rp = varrho * c.pow(gamma)

        res = []

        # (1) c FOC
        term_util = c.pow(-gamma) - (y * Delta / A).pow(1.0 + omega)
        term_vartheta_now = (
            (((1.0 + omega) * c + gamma * y) * y.pow(omega) * c.pow(gamma - 1.0) * (Delta / A).pow(omega) * one_plus_tau / A)
            + Et_termN
        )
        term_vartheta_lag = (-gamma) * theta * vp * c.pow(-gamma - 1.0) * (1.0 + pi) ** eps * XiN
        term_varrho_now = 1.0 + Et_termD
        term_varrho_lag = (-gamma) * theta * rp * c.pow(-gamma - 1.0) * (1.0 + pi) ** (eps - 1.0) * XiD
        res.append(term_util + vartheta * term_vartheta_now + term_vartheta_lag + varrho * term_varrho_now + term_varrho_lag)

        # (2) Delta FOC
        disutil_over_D = -(y * Delta / A).pow(1.0 + omega) / Delta
        res.append(disutil_over_D - zeta + beta * Et_theta_zeta_pi + vartheta * omega * y.pow(1.0 + omega) * c.pow(gamma) * (Delta / A).pow(omega) * (1.0 / Delta) * one_plus_tau / A)

        Delta_prev_exp = Delta  # no-switching local SS
        # (3) pi FOC
        res.append(
            nu * theta * (eps - 1.0) * (1.0 + pi).pow(eps - 2.0)
            + zeta * theta * eps * (1.0 + pi).pow(eps - 1.0) * Delta_prev_exp
            + eps * theta * vp * c.pow(-gamma) * (1.0 + pi).pow(eps - 1.0) * XiN
            + (eps - 1.0) * theta * rp * c.pow(-gamma) * (1.0 + pi).pow(eps - 2.0) * XiD
        )

        # (4) pstar FOC
        res.append(-mu * XiD + nu * (1.0 - theta) * (1.0 - eps) * pstar.pow(-eps) + zeta * (1.0 - theta) * (-eps) * pstar.pow(-eps - 1.0))

        # (5) XiN FOC
        res.append(mu * M - vartheta + theta * vp * c.pow(-gamma) * (1.0 + pi).pow(eps))

        # (6) XiD FOC
        res.append(-mu * pstar - varrho + theta * rp * c.pow(-gamma) * (1.0 + pi).pow(eps - 1.0))

        # (7) c^{-gamma} = lam
        res.append(c.pow(-gamma) - lam)

        # (8) labor
        res.append(h.pow(omega) - w * lam)

        # (9) XiN recursion
        res.append(XiN - (y * w * one_plus_tau / A) - Et_XiN)

        # (10) XiD recursion
        res.append(XiD - y - Et_XiD)

        # (11) pstar def
        res.append(pstar - (M * XiN / XiD))

        # (12) Calvo price index
        res.append(1.0 - (theta * (1.0 + pi).pow(eps - 1.0) + (1.0 - theta) * pstar.pow(1.0 - eps)))

        # (13) Delta law (Delta_prev=Delta)
        res.append(Delta - (theta * (1.0 + pi).pow(eps) * Delta + (1.0 - theta) * pstar.pow(-eps)))

        return torch.stack(res)

    # Initial guess
    c0 = torch.tensor(1.0, device=dev, dtype=dt)
    u0 = torch.zeros(13, device=dev, dtype=dt)
    u0[0] = torch.sqrt(c0)  # c via square
    u0[1] = 0.0
    u0[2] = 1.0
    u0[3] = 1.0
    u0[4] = 1.0
    u0[5] = 1.0
    u0[6] = 1.0
    u0[7] = 1.0
    u0[8] = 0.1
    u0[9] = 0.1
    u0[10] = 0.1
    u0[11] = 0.1
    u0[12] = -0.1

    sol = _newton_solve(residual, u0, max_iter=200, tol=1e-10, damping=0.5)

    c, pi, pstar, lam, w, XiN, XiD, Delta, mu, nu, zeta, vartheta, varrho = [sol[i].item() for i in range(13)]
    c = c * c + 1e-12
    Delta = Delta * Delta + 1e-12
    pstar = pstar * pstar + 1e-12
    XiN = XiN * XiN + 1e-14
    XiD = XiD * XiD + 1e-14

    vartheta_prev = float(vartheta * (c ** float(gamma)))
    varrho_prev = float(varrho * (c ** float(gamma)))

    return {
        "c": float(c),
        "pi": float(pi),
        "pstar": float(pstar),
        "lam": float(lam),
        "w": float(w),
        "XiN": float(XiN),
        "XiD": float(XiD),
        "Delta": float(Delta),
        "mu": float(mu),
        "nu": float(nu),
        "zeta": float(zeta),
        "vartheta": float(vartheta),
        "varrho": float(varrho),
        "vartheta_prev": vartheta_prev,
        "varrho_prev": varrho_prev,
        "regime": int(regime),
    }


def commitment_init_by_regime(params: ModelParams) -> Dict[int, Dict[str, float]]:
    """Return principled (no-heuristic) commitment init multipliers for both regimes."""
    return {0: commitment_local_ss(params, 0), 1: commitment_local_ss(params, 1)}


@dataclass
class CommitmentSSS:
    by_regime: Dict[int, Dict[str, float]]


def solve_commitment_sss_switching(params: ModelParams, max_iter: int = 200, tol: float = 1e-12, damping: float = 1.0) -> CommitmentSSS:
    """
    LEGACY helper: regime-conditional point solution for the A.3 system using a Bayes-closure
    for lagged co-states.

    IMPORTANT (paper-faithful usage): The paper's "timeless" commitment results are computed
    from the optimal policy itself (a fixed point / center of the ergodic distribution), not by
    imposing a single deterministic value for lagged co-states conditional on s_t.

    This routine is kept for backward compatibility and as a warm-start diagnostic, but the
    recommended, "strictly per paper" SSS for commitment is:

        solve_commitment_sss_from_policy(params, net)

    where net is the trained commitment policy network.
    """
    dev, dt = params.device, params.dtype
    beta, gamma, omega, eps, theta, M = params.beta, params.gamma, params.omega, params.eps, params.theta, params.M

    P = params.P  # [s_next, s]

    g = torch.tensor(params.g_bar, device=dev, dtype=dt)
    A = torch.tensor(1.0, device=dev, dtype=dt)
    xi = torch.tensor(0.0, device=dev, dtype=dt)
    eta0 = torch.tensor(0.0, device=dev, dtype=dt)
    eta1 = torch.tensor(params.eta_bar, device=dev, dtype=dt)

    def one_plus_tau(s: int) -> torch.Tensor:
        eta = eta0 if s == 0 else eta1
        return torch.tensor(1.0 - params.tau_bar, device=dev, dtype=dt) + xi + eta

    def unpack(u: torch.Tensor):
        return tuple(u[i] for i in range(13))

    def pack(U0: torch.Tensor, U1: torch.Tensor) -> torch.Tensor:
        return torch.cat([U0, U1], dim=0)

    def split(Z: torch.Tensor):
        return Z[:13], Z[13:]

    def residual_regime(u: torch.Tensor, s: int, u0_next: torch.Tensor, u1_next: torch.Tensor) -> torch.Tensor:
        c, pi, pstar, lam, w, XiN, XiD, Delta, mu, nu, zeta, vartheta, varrho = unpack(u)

        c = c**2 + 1e-12
        Delta = Delta**2 + 1e-12
        pstar = pstar**2 + 1e-12
        XiN = XiN**2 + 1e-14
        XiD = XiD**2 + 1e-14

        y = c + g
        h = y * Delta / A
        optau = one_plus_tau(s)

        def decode_next(u_next):
            cN, piN, pstarN, lamN, wN, XiNN, XiDN, DeltaN, muN, nuN, zetaN, varthetaN, varrhoN = unpack(u_next)
            cN = cN**2 + 1e-12
            DeltaN = DeltaN**2 + 1e-12
            pstarN = pstarN**2 + 1e-12
            XiNN = XiNN**2 + 1e-14
            XiDN = XiDN**2 + 1e-14
            return cN, piN, pstarN, lamN, wN, XiNN, XiDN, DeltaN, muN, nuN, zetaN, varthetaN, varrhoN

        c0, pi0, pstar0, lam0, w0, XiN0, XiD0, Delta0, mu0, nu0, zeta0, vartheta0, varrho0 = decode_next(u0_next)
        c1, pi1, pstar1, lam1, w1, XiN1, XiD1, Delta1, mu1, nu1, zeta1, vartheta1, varrho1 = decode_next(u1_next)

        Lam0 = beta * (lam0 / lam)
        Lam1 = beta * (lam1 / lam)

        pi0w = P[s, 0]
        pi1w = P[s, 1]

        Et_XiN = theta * (pi0w * (Lam0 * (1.0 + pi0).pow(eps) * XiN0) + pi1w * (Lam1 * (1.0 + pi1).pow(eps) * XiN1))
        Et_XiD = theta * (pi0w * (Lam0 * (1.0 + pi0).pow(eps - 1.0) * XiD0) + pi1w * (Lam1 * (1.0 + pi1).pow(eps - 1.0) * XiD1))

        Et_theta_zeta_pi = (pi0w * (theta * (1.0 + pi0).pow(eps) * zeta0) + pi1w * (theta * (1.0 + pi1).pow(eps) * zeta1))

        Et_termN = beta * theta * gamma * c.pow(gamma - 1.0) * (
            pi0w * (c0.pow(-gamma) * (1.0 + pi0).pow(eps) * XiN0) + pi1w * (c1.pow(-gamma) * (1.0 + pi1).pow(eps) * XiN1)
        )
        Et_termD = beta * theta * gamma * c.pow(gamma - 1.0) * (
            pi0w * (c0.pow(-gamma) * (1.0 + pi0).pow(eps - 1.0) * XiD0) + pi1w * (c1.pow(-gamma) * (1.0 + pi1).pow(eps - 1.0) * XiD1)
        )

        # Lagged objects in regime-conditional SSS equal same-regime steady values
        # Lagged objects in the state are realized from t-1. Conditioning on s_t does not pin s_{t-1},
        # so in a regime-switching SSS we use backward (Bayes) weights based on the stationary distribution.
        p12 = float(params.p12); p21 = float(params.p21)
        pi0_stat = p21 / (p12 + p21)
        pi1_stat = p12 / (p12 + p21)
        pi_s = pi0_stat if s == 0 else pi1_stat
        B0 = (pi0_stat * float(P[0, s])) / pi_s
        B1 = (pi1_stat * float(P[1, s])) / pi_s

        Delta_prev_ss = B0 * Delta0 + B1 * Delta1

        vp0_ss = vartheta0 * c0.pow(gamma)
        vp1_ss = vartheta1 * c1.pow(gamma)
        rp0_ss = varrho0 * c0.pow(gamma)
        rp1_ss = varrho1 * c1.pow(gamma)

        vp_prev_ss = B0 * vp0_ss + B1 * vp1_ss
        rp_prev_ss = B0 * rp0_ss + B1 * rp1_ss

        res = []

        term_util = c.pow(-gamma) - (y * Delta / A).pow(1.0 + omega)
        term_vartheta_now = (((1.0 + omega) * c + gamma * y) * y.pow(omega) * c.pow(gamma - 1.0) * (Delta / A).pow(omega) * optau / A) + Et_termN
        term_vartheta_lag = (-gamma) * theta * vp_prev_ss * c.pow(-gamma - 1.0) * (1.0 + pi).pow(eps) * XiN
        term_varrho_now = 1.0 + Et_termD
        term_varrho_lag = (-gamma) * theta * rp_prev_ss * c.pow(-gamma - 1.0) * (1.0 + pi).pow(eps - 1.0) * XiD
        res.append(term_util + vartheta * term_vartheta_now + term_vartheta_lag + varrho * term_varrho_now + term_varrho_lag)

        disutil_over_D = -(y * Delta / A).pow(1.0 + omega) / Delta
        res.append(disutil_over_D - zeta + beta * Et_theta_zeta_pi + vartheta * omega * y.pow(1.0 + omega) * c.pow(gamma) * (Delta / A).pow(omega) * (1.0 / Delta) * optau / A)

        res.append(
            nu * theta * (eps - 1.0) * (1.0 + pi).pow(eps - 2.0)
            + zeta * theta * eps * (1.0 + pi).pow(eps - 1.0) * Delta_prev_ss
            + eps * theta * vp_prev_ss * c.pow(-gamma) * (1.0 + pi).pow(eps - 1.0) * XiN
            + (eps - 1.0) * theta * rp_prev_ss * c.pow(-gamma) * (1.0 + pi).pow(eps - 2.0) * XiD
        )

        res.append(-mu * XiD + nu * (1.0 - theta) * (1.0 - eps) * pstar.pow(-eps) + zeta * (1.0 - theta) * (-eps) * pstar.pow(-eps - 1.0))

        res.append(mu * M - vartheta + theta * vp_prev_ss * c.pow(-gamma) * (1.0 + pi).pow(eps))
        res.append(-mu * pstar - varrho + theta * rp_prev_ss * c.pow(-gamma) * (1.0 + pi).pow(eps - 1.0))

        res.append(c.pow(-gamma) - lam)
        res.append(h.pow(omega) - w * lam)

        res.append(XiN - (y * w * optau / A) - Et_XiN)
        res.append(XiD - y - Et_XiD)

        res.append(pstar - (M * XiN / XiD))
        res.append(1.0 - (theta * (1.0 + pi).pow(eps - 1.0) + (1.0 - theta) * pstar.pow(1.0 - eps)))

        res.append(Delta - (theta * (1.0 + pi).pow(eps) * Delta_prev_ss + (1.0 - theta) * pstar.pow(-eps)))

        return torch.stack(res)

    def F(Z: torch.Tensor) -> torch.Tensor:
        U0, U1 = split(Z)
        r0 = residual_regime(U0, 0, U0, U1)
        r1 = residual_regime(U1, 1, U0, U1)
        return torch.cat([r0, r1], dim=0)

    def u_from_dict(d):
        return torch.tensor(
            [d["c"], d["pi"], d["pstar"], d["lam"], d["w"], d["XiN"], d["XiD"], d["Delta"], d["mu"], d["nu"], d["zeta"], d["vartheta"], d["varrho"]],
            device=dev,
            dtype=dt,
        )

    U0 = u_from_dict(commitment_local_ss(params, 0))
    U1 = u_from_dict(commitment_local_ss(params, 1))

    def inv_param(u):
        u = u.clone()
        for i in [0, 2, 5, 6, 7]:
            u[i] = torch.sqrt(torch.clamp(u[i] - (1e-12 if i in [0, 2, 7] else 1e-14), min=1e-16))
        return u

    Z0 = pack(inv_param(U0), inv_param(U1))

    Z = _newton_solve(F, Z0, max_iter=max_iter, tol=tol, damping=damping)
    U0s, U1s = split(Z)

    def decode(U, vp_prev: float, rp_prev: float):
        c, pi, pstar, lam, w, XiN, XiD, Delta, mu, nu, zeta, vartheta, varrho = unpack(U)
        c = float((c**2 + 1e-12).cpu())
        pstar = float((pstar**2 + 1e-12).cpu())
        XiN = float((XiN**2 + 1e-14).cpu())
        XiD = float((XiD**2 + 1e-14).cpu())
        Delta = float((Delta**2 + 1e-12).cpu())
        return {
            "c": c,
            "pi": float(pi.cpu()),
            "pstar": pstar,
            "lam": float(lam.cpu()),
            "w": float(w.cpu()),
            "XiN": XiN,
            "XiD": XiD,
            "Delta": Delta,
            "mu": float(mu.cpu()),
            "nu": float(nu.cpu()),
            "zeta": float(zeta.cpu()),
            "vartheta": float(vartheta.cpu()),
            "varrho": float(varrho.cpu()),
            "vartheta_prev": float(vp_prev),
            "varrho_prev": float(rp_prev),
        }

    c0 = float((U0s[0] ** 2 + 1e-12).cpu())
    c1 = float((U1s[0] ** 2 + 1e-12).cpu())
    vartheta0 = float(U0s[11].cpu())
    vartheta1 = float(U1s[11].cpu())
    varrho0 = float(U0s[12].cpu())
    varrho1 = float(U1s[12].cpu())
    vp0 = vartheta0 * (c0 ** float(gamma))
    vp1 = vartheta1 * (c1 ** float(gamma))
    rp0 = varrho0 * (c0 ** float(gamma))
    rp1 = varrho1 * (c1 ** float(gamma))

        # backward-conditional lags for storage (same logic as inside residual_regime)
    p12 = float(params.p12); p21 = float(params.p21)
    pi0_stat = p21 / (p12 + p21)
    pi1_stat = p12 / (p12 + p21)

    def B(current_s: int):
        pi_s = pi0_stat if current_s == 0 else pi1_stat
        B0 = (pi0_stat * float(params.P[current_s, 0].detach().cpu())) / pi_s
        B1 = (pi1_stat * float(params.P[current_s, 1].detach().cpu())) / pi_s
        return B0, B1

    B00, B10 = B(0)
    B01, B11 = B(1)

    vp_prev_0 = B00 * vp0 + B10 * vp1
    vp_prev_1 = B01 * vp0 + B11 * vp1
    rp_prev_0 = B00 * rp0 + B10 * rp1
    rp_prev_1 = B01 * rp0 + B11 * rp1

    return CommitmentSSS(by_regime={
        0: decode(U0s, vp_prev=vp_prev_0, rp_prev=rp_prev_0),
        1: decode(U1s, vp_prev=vp_prev_1, rp_prev=rp_prev_1),
    })


def solve_commitment_sss_from_policy(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-12,
    damping: float = 0.5,
    floors: Dict[str, float] | None = None,
) -> PolicySSS:
    """Switching-consistent (Table-2) commitment SSS computed as a fixed point of the trained policy.

    Note: Frozen-regime SSS routines were removed; this function is kept as a stable public alias.
    """
    return solve_commitment_sss_from_policy_switching(
        params,
        net,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
        floors=floors,
    )


def solve_discretion_sss_from_policy(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-12,
    damping: float = 0.5,
    floors: Dict[str, float] | None = None,
) -> PolicySSS:
    """Switching-consistent (Table-2) discretion SSS computed as a fixed point of the trained policy.

    Note: Frozen-regime SSS routines were removed; this function is kept as a stable public alias.
    """
    return solve_discretion_sss_from_policy_switching(
        params,
        net,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
        floors=floors,
    )


def solve_commitment_sss_from_policy_switching(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-12,
    damping: float = 0.5,
    floors: Dict[str, float] | None = None,
) -> PolicySSS:
    """Switching-consistent (Table-2) commitment SSS computed as a fixed point of the trained policy."""
    from .sss_from_policy import switching_policy_sss_by_regime_from_policy
    return switching_policy_sss_by_regime_from_policy(
        params,
        net,
        policy="commitment",
        max_iter=max_iter,
        tol=tol,
        damping=damping,
        floors=floors,
    )


def solve_discretion_sss_from_policy_switching(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-12,
    damping: float = 0.5,
    floors: Dict[str, float] | None = None,
) -> PolicySSS:
    """Switching-consistent (Table-2) discretion SSS computed as a fixed point of the trained policy."""
    from .sss_from_policy import switching_policy_sss_by_regime_from_policy
    return switching_policy_sss_by_regime_from_policy(
        params,
        net,
        policy="discretion",
        max_iter=max_iter,
        tol=tol,
        damping=damping,
        floors=floors,
    )

@dataclass
class DiscretionSSS:
    by_regime: Dict[int, Dict[str, float]]


def solve_discretion_sss_switching(
    params: ModelParams,
    max_iter: int = 60,
    tol: float = 1e-10,
    damping: float = 1.0,
) -> DiscretionSSS:
    """
    Stochastic steady state (SSS) under discretion with regime switching.

    **Strictly follows the paper's final system**: 11 equations per regime (22 total).
    We do NOT treat derivatives as additional unknowns or impose extra "consistency residuals".
    Instead, the derivative terms needed inside the Delta FOC are computed as *derived objects*
    using a (small) fixed-point iteration on the implied-derivative mapping.

    A practical staged solve is used to keep runtime reasonable while still converging tightly:
      - Stage 1: solve core-22 with frozen derivatives (set to 0).
      - Stage 2: compute exact implied derivatives at the stage-1 solution.
      - Stage 3: a few Newton steps on core-22 while recomputing derivatives each step.

    Unknowns per regime (11):
      [c, pi, pstar, lam, w, XiN, XiD, Delta, mu, rho, zeta]
    """
    from .model_common import State
    from .residuals_a2 import residuals_a2

    dev = params.device
    dt = torch.float64

    import dataclasses as _dc
    _d = _dc.asdict(params)
    _d["dtype"] = dt
    params64 = ModelParams(**_d).to_torch()

    logA = torch.tensor(0.0, device=dev, dtype=dt)
    logg = torch.tensor(0.0, device=dev, dtype=dt)
    xi = torch.tensor(0.0, device=dev, dtype=dt)

    P = params.P.to(device=dev, dtype=dt)
    theta = torch.tensor(params.theta, device=dev, dtype=dt)
    beta = torch.tensor(params.beta, device=dev, dtype=dt)
    eps = torch.tensor(params.eps, device=dev, dtype=dt)

    # --- helpers -------------------------------------------------------------
    def pack(u: Dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [u["c"], u["pi"], u["pstar"], u["lam"], u["w"], u["XiN"], u["XiD"],
             u["Delta"], u["mu"], u["rho"], u["zeta"]],
            device=dev, dtype=dt,
        )

    def unpack(u: torch.Tensor):
        return u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10]

    def encode_u(u: torch.Tensor) -> torch.Tensor:
        # positivity constraints
        c, pi, pstar, lam, w, XiN, XiD, D, mu, rho, zeta = unpack(u)
        return torch.stack([
            torch.log(c),
            pi,
            torch.log(pstar),
            torch.log(lam),
            w,
            XiN,
            XiD,
            torch.log(D),
            mu,
            rho,
            zeta,
        ])

    def decode_u(z: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            torch.exp(z[0]),
            z[1],
            torch.exp(z[2]),
            torch.exp(z[3]),
            z[4],
            z[5],
            z[6],
            torch.exp(z[7]),
            z[8],
            z[9],
            z[10],
        ])

    def decode_Z22(Z: torch.Tensor):
        z0 = Z[:11]
        z1 = Z[11:]
        return decode_u(z0), decode_u(z1)

    # stationary distribution and backward weights for Delta_prev
    def stationary_dist(Pm: torch.Tensor) -> torch.Tensor:
        # Solve (I - P.T) * pi = 0 with sum(pi)=1. Here P is 2x2 row-stochastic: P[current,next].
        A = (I - Pm.T)  # 2x2
        A = torch.vstack([A, torch.ones((1, 2), device=dev, dtype=dt)])  # 3x2
        b = torch.tensor([[0.0], [0.0], [1.0]], device=dev, dtype=dt)    # 3x1
        pi = torch.linalg.lstsq(A, b).solution.squeeze(-1)               # 2,
        return pi

    pi_stat = stationary_dist(P)

    def backward_weights(curr_s: int) -> torch.Tensor:
        # w_prev[j] = Pr(s_{t-1}=j | s_t=curr_s) ∝ pi_stat[j] * P[j, curr_s]
        w = torch.stack([pi_stat[0] * P[0, curr_s], pi_stat[1] * P[1, curr_s]])
        return w / torch.sum(w)

    bw0 = backward_weights(0)
    bw1 = backward_weights(1)

    def compute_Delta_prev(u0: torch.Tensor, u1: torch.Tensor):
        D0 = unpack(u0)[7]
        D1 = unpack(u1)[7]
        DP0 = bw0[0] * D0 + bw0[1] * D1
        DP1 = bw1[0] * D0 + bw1[1] * D1
        return DP0, DP1

    # expectations needed by residuals_a2
    def F_val(c, pi, XiD):
        return theta * beta * c.pow(-params.gamma) * (1.0 + pi).pow(eps - 1.0) * XiD

    def G_val(c, pi, XiN):
        return theta * beta * c.pow(-params.gamma) * (1.0 + pi).pow(eps) * XiN

    def expectations(u0: torch.Tensor, u1: torch.Tensor):
        c0, pi0, pstar0, lam0, w0, XiN0, XiD0, D0, mu0, rho0, zeta0 = unpack(u0)
        c1, pi1, pstar1, lam1, w1, XiN1, XiD1, D1, mu1, rho1, zeta1 = unpack(u1)

        F0 = F_val(c0, pi0, XiD0); F1 = F_val(c1, pi1, XiD1)
        G0 = G_val(c0, pi0, XiN0); G1 = G_val(c1, pi1, XiN1)

        TH0 = theta * (1.0 + pi0).pow(eps) * zeta0
        TH1 = theta * (1.0 + pi1).pow(eps) * zeta1

        # Et Xi expectations in paper (see your existing implementation)
        Et_XiN_0 = P[0, 0] * (theta * (beta * lam0 / lam0) * (1.0 + pi0).pow(eps) * XiN0) + \
                  P[0, 1] * (theta * (beta * lam1 / lam0) * (1.0 + pi1).pow(eps) * XiN1)
        Et_XiD_0 = P[0, 0] * (theta * (beta * lam0 / lam0) * (1.0 + pi0).pow(eps - 1.0) * XiD0) + \
                  P[0, 1] * (theta * (beta * lam1 / lam0) * (1.0 + pi1).pow(eps - 1.0) * XiD1)

        Et_XiN_1 = P[1, 0] * (theta * (beta * lam0 / lam1) * (1.0 + pi0).pow(eps) * XiN0) + \
                  P[1, 1] * (theta * (beta * lam1 / lam1) * (1.0 + pi1).pow(eps) * XiN1)
        Et_XiD_1 = P[1, 0] * (theta * (beta * lam0 / lam1) * (1.0 + pi0).pow(eps - 1.0) * XiD0) + \
                  P[1, 1] * (theta * (beta * lam1 / lam1) * (1.0 + pi1).pow(eps - 1.0) * XiD1)

        Et_F_0 = P[0, 0] * F0 + P[0, 1] * F1
        Et_G_0 = P[0, 0] * G0 + P[0, 1] * G1
        Et_TH_0 = P[0, 0] * TH0 + P[0, 1] * TH1

        Et_F_1 = P[1, 0] * F0 + P[1, 1] * F1
        Et_G_1 = P[1, 0] * G0 + P[1, 1] * G1
        Et_TH_1 = P[1, 0] * TH0 + P[1, 1] * TH1

        return {
            0: {"Et_F": Et_F_0, "Et_G": Et_G_0, "Et_TH": Et_TH_0, "Et_XiN": Et_XiN_0, "Et_XiD": Et_XiD_0},
            1: {"Et_F": Et_F_1, "Et_G": Et_G_1, "Et_TH": Et_TH_1, "Et_XiN": Et_XiN_1, "Et_XiD": Et_XiD_1},
        }

    def Et_from_regime_values(x0: torch.Tensor, x1: torch.Tensor):
        return P[0, 0] * x0 + P[0, 1] * x1, P[1, 0] * x0 + P[1, 1] * x1

    def implied_dF_dG_given_derivative_expectations(
        s: int,
        us: torch.Tensor,
        DP: torch.Tensor,
        *,
        exps: Dict[int, Dict[str, torch.Tensor]],
        Et_dF: torch.Tensor,
        Et_dG: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute implied dF/dDP and dG/dDP using IFT via jacobian of residuals_a2
        us = us.clone().detach().to(device=dev, dtype=dt).requires_grad_(True)
        DP = DP.clone().detach().to(device=dev, dtype=dt).requires_grad_(True)

        c, pi, pstar, lam, w, XiN, XiD, Delta, mu, rho, zeta = unpack(us)
        out = {"c": c, "pi": pi, "pstar": pstar, "lam": lam, "w": w,
               "XiN": XiN, "XiD": XiD, "Delta": Delta, "mu": mu, "rho": rho, "zeta": zeta}

        st = State(Delta_prev=DP, logA=logA, loggtilde=logg, xi=xi, s=torch.tensor(s, device=dev, dtype=torch.long))

        res = residuals_a2(
            params=params64,
            st=st,
            out=out,
            Et_F_next=exps[s]["Et_F"],
            Et_G_next=exps[s]["Et_G"],
            Et_dF_dDelta_next=Et_dF,
            Et_dG_dDelta_next=Et_dG,
            Et_theta_zeta_pi_next=exps[s]["Et_TH"],
            Et_XiN_next=exps[s]["Et_XiN"],
            Et_XiD_next=exps[s]["Et_XiD"],
        )

        keys = ["res_c_foc", "res_pi_foc", "res_pstar_foc", "res_Delta_foc",
                "res_c_lam", "res_labor", "res_XiN_rec", "res_XiD_rec",
                "res_pstar_def", "res_calvo", "res_Delta_law"]

        def f(u_vec: torch.Tensor, dp: torch.Tensor):
            c, pi, pstar, lam, w, XiN, XiD, Delta, mu, rho, zeta = unpack(u_vec)
            out2 = {"c": c, "pi": pi, "pstar": pstar, "lam": lam, "w": w,
                    "XiN": XiN, "XiD": XiD, "Delta": Delta, "mu": mu, "rho": rho, "zeta": zeta}
            st2 = State(Delta_prev=dp, logA=logA, loggtilde=logg, xi=xi, s=torch.tensor(s, device=dev, dtype=torch.long))
            r = residuals_a2(
                params=params64,
                st=st2,
                out=out2,
                Et_F_next=exps[s]["Et_F"],
                Et_G_next=exps[s]["Et_G"],
                Et_dF_dDelta_next=Et_dF,
                Et_dG_dDelta_next=Et_dG,
                Et_theta_zeta_pi_next=exps[s]["Et_TH"],
                Et_XiN_next=exps[s]["Et_XiN"],
                Et_XiD_next=exps[s]["Et_XiD"],
            )
            return torch.stack([r[k] for k in keys])

        Ju = torch.autograd.functional.jacobian(lambda uu: f(uu, DP), us, create_graph=False)
        Jdp = torch.autograd.functional.jacobian(lambda dd: f(us, dd), DP, create_graph=False).reshape(-1, 1)

        # d u / d DP from IFT: Ju * du + Jdp = 0
        du = -torch.linalg.solve(Ju, Jdp).reshape(-1)  # (11,)
        # implied dF/dDP and dG/dDP = (d/dDP) Et_F_next, Et_G_next? Here we want the objects used in training:
        # We follow your existing convention: treat implied derivatives as derivative of Et_F_next/Et_G_next wrt current Delta_prev.
        # Since Et_F_next depends on (u0,u1) only, we approximate by derivative of F_s wrt DP via du chain rule.
        # Compute dF = dF_val/du * du, similarly for G with XiD/XiN components.
        # Simpler: use your stored dF/dG as IFT target for Et_F and Et_G in residuals_a2 by differentiating their definitions:
        c, pi, pstar, lam, w, XiN, XiD, Delta, mu, rho, zeta = unpack(us)
        # partials for F and G wrt u:
        # F = theta*beta*c^{-gamma}*(1+pi)^{eps-1}*XiD
        dF = (theta*beta) * (
            (-params.gamma) * c.pow(-params.gamma - 1.0) * (1.0 + pi).pow(eps - 1.0) * XiD * du[0]
            + c.pow(-params.gamma) * (eps - 1.0) * (1.0 + pi).pow(eps - 2.0) * XiD * du[1]
            + c.pow(-params.gamma) * (1.0 + pi).pow(eps - 1.0) * du[6]
        )
        # G = theta*beta*c^{-gamma}*(1+pi)^{eps}*XiN
        dG = (theta*beta) * (
            (-params.gamma) * c.pow(-params.gamma - 1.0) * (1.0 + pi).pow(eps) * XiN * du[0]
            + c.pow(-params.gamma) * eps * (1.0 + pi).pow(eps - 1.0) * XiN * du[1]
            + c.pow(-params.gamma) * (1.0 + pi).pow(eps) * du[5]
        )
        return dF.detach(), dG.detach()

    def implied_derivs_fixed_point(u0: torch.Tensor, u1: torch.Tensor, dF0, dF1, dG0, dG1, max_fp_iter: int, fp_tol: float):
        for _ in range(max_fp_iter):
            exps = expectations(u0, u1)
            DP0, DP1 = compute_Delta_prev(u0, u1)
            Et_dF_0, Et_dF_1 = Et_from_regime_values(dF0, dF1)
            Et_dG_0, Et_dG_1 = Et_from_regime_values(dG0, dG1)

            ndF0, ndG0 = implied_dF_dG_given_derivative_expectations(0, u0, DP0, exps=exps, Et_dF=Et_dF_0, Et_dG=Et_dG_0)
            ndF1, ndG1 = implied_dF_dG_given_derivative_expectations(1, u1, DP1, exps=exps, Et_dF=Et_dF_1, Et_dG=Et_dG_1)

            diff = torch.max(torch.stack([torch.abs(ndF0-dF0), torch.abs(ndF1-dF1), torch.abs(ndG0-dG0), torch.abs(ndG1-dG1)]))
            dF0, dF1, dG0, dG1 = ndF0, ndF1, ndG0, ndG1
            if diff.item() < fp_tol:
                break
        return dF0, dF1, dG0, dG1

    # core residuals (22)
    def residuals_core(Z22: torch.Tensor, dF0, dF1, dG0, dG1) -> torch.Tensor:
        u0, u1 = decode_Z22(Z22)
        exps = expectations(u0, u1)
        DP0, DP1 = compute_Delta_prev(u0, u1)
        Et_dF_0, Et_dF_1 = Et_from_regime_values(dF0, dF1)
        Et_dG_0, Et_dG_1 = Et_from_regime_values(dG0, dG1)

        def res_regime(s: int, u: torch.Tensor, DP: torch.Tensor, Et_dF: torch.Tensor, Et_dG: torch.Tensor):
            c, pi, pstar, lam, w, XiN, XiD, Delta, mu, rho, zeta = unpack(u)
            out = {"c": c, "pi": pi, "pstar": pstar, "lam": lam, "w": w,
                   "XiN": XiN, "XiD": XiD, "Delta": Delta, "mu": mu, "rho": rho, "zeta": zeta}
            st = State(Delta_prev=DP, logA=logA, loggtilde=logg, xi=xi, s=torch.tensor(s, device=dev, dtype=torch.long))
            res = residuals_a2(
                params=params64,
                st=st,
                out=out,
                Et_F_next=exps[s]["Et_F"],
                Et_G_next=exps[s]["Et_G"],
                Et_dF_dDelta_next=Et_dF,
                Et_dG_dDelta_next=Et_dG,
                Et_theta_zeta_pi_next=exps[s]["Et_TH"],
                Et_XiN_next=exps[s]["Et_XiN"],
                Et_XiD_next=exps[s]["Et_XiD"],
            )
            keys = ["res_c_foc", "res_pi_foc", "res_pstar_foc", "res_Delta_foc",
                    "res_c_lam", "res_labor", "res_XiN_rec", "res_XiD_rec",
                    "res_pstar_def", "res_calvo", "res_Delta_law"]
            return torch.stack([res[k] for k in keys])

        r0 = res_regime(0, u0, DP0, Et_dF_0, Et_dG_0)
        r1 = res_regime(1, u1, DP1, Et_dF_1, Et_dG_1)
        return torch.cat([r0, r1], dim=0)

    def newton_core(Z_init: torch.Tensor, dF0, dF1, dG0, dG1, max_it: int, tol_core: float, damping_local: float):
        Z = Z_init.clone().detach().to(device=dev, dtype=dt)
        for _ in range(max_it):
            r = residuals_core(Z, dF0, dF1, dG0, dG1)
            m = torch.max(torch.abs(r)).item()
            if m < tol_core:
                return Z, m
            J = torch.autograd.functional.jacobian(lambda zz: residuals_core(zz, dF0, dF1, dG0, dG1), Z, create_graph=False)
            step = torch.linalg.solve(J, r)
            Z = Z - damping_local * step
            if not torch.isfinite(Z).all():
                break
        r = residuals_core(Z, dF0, dF1, dG0, dG1)
        return Z, torch.max(torch.abs(r)).item()

    # --- initialization (reuse your existing seeds) --------------------------
    init_pairs = [
        (0.0, 0.0),
        (0.0, 0.003),
        (0.0, 0.006),
        (0.0, 0.009),
        (0.0, 0.012),
        (0.0, 0.018),
        (0.0, 0.024),
        (0.0, 0.030),
        (0.0, -0.003),
        (0.0, -0.006),
    ]

    best = None  # (score, Z, u0, u1, dF0,dF1,dG0,dG1)
    for pi0_init, pi1_init in init_pairs:
        # start from your previous near-flex guess
        # (use existing helper in file if present; otherwise simple)
        u0 = pack({"c": 0.94, "pi": pi0_init, "pstar": 1.0, "lam": 1.0, "w": 1.0,
                   "XiN": 4.0, "XiD": 4.0, "Delta": 1.0, "mu": 0.0, "rho": 4.0, "zeta": -3.0})
        u1 = pack({"c": 0.90, "pi": pi1_init, "pstar": 1.0, "lam": 1.0, "w": 1.0,
                   "XiN": 4.0, "XiD": 4.0, "Delta": 1.0, "mu": 0.0, "rho": 4.0, "zeta": -3.0})

        Z0 = torch.cat([encode_u(u0), encode_u(u1)], dim=0)

        # Stage 1: freeze derivatives to zero
        Z1, s1 = newton_core(Z0, torch.zeros((),device=dev,dtype=dt), torch.zeros((),device=dev,dtype=dt),
                             torch.zeros((),device=dev,dtype=dt), torch.zeros((),device=dev,dtype=dt),
                             max_it=max_iter, tol_core=1e-6, damping_local=damping)
        if not torch.isfinite(Z1).all():
            continue

        u0_1, u1_1 = decode_Z22(Z1)

        # Stage 2: compute implied derivatives at stage1 point
        dF0 = torch.zeros((), device=dev, dtype=dt); dF1 = torch.zeros((), device=dev, dtype=dt)
        dG0 = torch.zeros((), device=dev, dtype=dt); dG1 = torch.zeros((), device=dev, dtype=dt)
        dF0, dF1, dG0, dG1 = implied_derivs_fixed_point(u0_1, u1_1, dF0, dF1, dG0, dG1, max_fp_iter=15, fp_tol=1e-8)

        # Stage 3: a few Newton steps with refreshed derivatives
        Z = Z1
        for _ in range(8):
            u0c, u1c = decode_Z22(Z)
            dF0, dF1, dG0, dG1 = implied_derivs_fixed_point(u0c, u1c, dF0, dF1, dG0, dG1, max_fp_iter=6, fp_tol=1e-10)
            Z, score = newton_core(Z, dF0, dF1, dG0, dG1, max_it=1, tol_core=tol, damping_local=damping)
            if score < tol:
                break

        u0f, u1f = decode_Z22(Z)
        final_r = residuals_core(Z, dF0, dF1, dG0, dG1)
        final_score = torch.max(torch.abs(final_r)).item()

        if (best is None) or (final_score < best[0]):
            best = (final_score, u0f, u1f, dF0, dF1, dG0, dG1)

    if best is None:
        raise RuntimeError("Discretion SSS: no candidate converged to a finite solution.")

    score, u0f, u1f, dF0, dF1, dG0, dG1 = best

    # build output dicts
    def to_dict(u: torch.Tensor, dF: torch.Tensor, dG: torch.Tensor):
        c, pi, pstar, lam, w, XiN, XiD, Delta, mu, rho, zeta = unpack(u)
        return {
            "c": float(c.item()),
            "pi": float(pi.item()),
            "pstar": float(pstar.item()),
            "lam": float(lam.item()),
            "w": float(w.item()),
            "XiN": float(XiN.item()),
            "XiD": float(XiD.item()),
            "Delta": float(Delta.item()),
            "mu": float(mu.item()),
            "rho": float(rho.item()),
            "zeta": float(zeta.item()),
            "dF_dDeltaPrev": float(dF.item()),
            "dG_dDeltaPrev": float(dG.item()),
            "core_max_residual": float(score),
        }

    by_regime = {
        0: to_dict(u0f, dF0, dG0),
        1: to_dict(u1f, dF1, dG1),
    }
    return DiscretionSSS(by_regime=by_regime)
def commitment_init_from_sss(params: ModelParams) -> Dict[int, Dict[str, float]]:
    """
    Return regime-specific initial lagged co-states for commitment.
    Keys 0 and 1 correspond to the current regime s_t used in the state vector.
    Each value contains:
        vartheta_prev:  ϑ_{t-1} * c_{t-1}^γ  (scaled lag co-state)
        varrho_prev:    ϱ_{t-1} * c_{t-1}^γ  (scaled lag co-state)
    """
    sss = solve_commitment_sss_switching(params).by_regime
    return {
        0: {"vartheta_prev": sss[0]["vartheta_prev"], "varrho_prev": sss[0]["varrho_prev"]},
        1: {"vartheta_prev": sss[1]["vartheta_prev"], "varrho_prev": sss[1]["varrho_prev"]},
    }