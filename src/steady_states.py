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
        eta_s = regime-specific cost-push level eta_s
        M = eps/(eps-1)

    IMPORTANT: The second term is the reciprocal, exactly as written in the paper.
    """
    M = float(params.M)
    gbar = float(params.g_bar)
    gamma = float(params.gamma)
    omega = float(params.omega)

    out: Dict[int, Dict[str, float]] = {}
    eta_levels = params.eta_by_regime
    n_regimes = int(params.n_regimes)
    for s in range(n_regimes):
        eta_s = float(eta_levels[s])
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

    # Natural rate in each regime (flex-price SSS):
    # r*_s = 1/(beta * E_s[m_{t+1}]) - 1, where
    # m_{t+1} = (c_{t+1}/c_t)^(-gamma) and expectation is over s_{t+1}.
    P = params.P.detach().cpu().numpy()
    c_by_regime = {k: out[k]["c"] for k in out.keys()}

    def r_star(s: int) -> float:
        c_s = float(c_by_regime[s])
        E_m = 0.0
        for sp in range(n_regimes):
            p_sp = float(P[s, sp])
            c_sp = float(c_by_regime[sp])
            E_m += p_sp * ((c_sp / c_s) ** (-gamma))
        return 1.0 / (float(params.beta) * E_m) - 1.0

    for s in range(n_regimes):
        out[s]["r_star"] = float(r_star(s))
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
    for s in range(int(params.n_regimes)):
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
        [flex.by_regime[s]["r_star"] for s in range(int(params.n_regimes))],
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

    Regime enters only through eta_s, hence (1+tau).
    """
    if int(regime) < 0 or int(regime) >= int(params.n_regimes):
        raise ValueError(f"regime must be in [0, {int(params.n_regimes)-1}], got {regime!r}")
    dev, dt = params.device, params.dtype

    g = torch.tensor(params.g_bar, device=dev, dtype=dt)
    eta = torch.tensor(float(params.eta_by_regime[int(regime)]), device=dev, dtype=dt)
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
    """Return principled (no-heuristic) commitment init multipliers for all regimes."""
    return {s: commitment_local_ss(params, s) for s in range(int(params.n_regimes))}


@dataclass
class CommitmentSSS:
    by_regime: Dict[int, Dict[str, float]]


def commitment_author_like_sss(
    params: ModelParams,
    *,
    vartheta_old: float = -0.019182,
    varrho_old: float = 0.016500,
    c_old: float = 0.921336,
    delta_prev: float = 1.0,
) -> CommitmentSSS:
    """
    Author-style timeless initialization used in the public Keras code (Hooks.py).

    Their state stores lagged multipliers and lagged consumption separately.
    In this PyTorch codebase the commitment state keeps scaled lagged multipliers:

        vartheta_prev = vartheta_old * c_old^gamma
        varrho_prev   = varrho_old   * c_old^gamma

    We assign the same warm-start values to all regimes, matching the author setup
    and extending it to multi-regime runs.
    """
    scale = float(c_old) ** float(params.gamma)
    vp = float(vartheta_old) * scale
    rp = float(varrho_old) * scale
    d = float(delta_prev)
    by = {
        s: {"Delta_prev": d, "vartheta_prev": vp, "varrho_prev": rp}
        for s in range(int(params.n_regimes))
    }
    return CommitmentSSS(by_regime=by)


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
    if int(params.n_regimes) < 2 or int(params.n_regimes) > 2:
        # Multi-regime fallback for legacy callers: return local per-regime SS guesses.
        # For paper-consistent switching SSS use solve_commitment_sss_from_policy(...).
        by = {s: commitment_local_ss(params, s) for s in range(int(params.n_regimes))}
        return CommitmentSSS(by_regime=by)

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
        # Backward weights use Pr(s_t=current_s | s_{t-1}=j) = P[j, current_s].
        B0 = (pi0_stat * float(params.P[0, current_s].detach().cpu())) / pi_s
        B1 = (pi1_stat * float(params.P[1, current_s].detach().cpu())) / pi_s
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
    max_iter: int = 80,
    tol: float = 1e-10,
    damping: float = 1.0,
) -> DiscretionSSS:
    """
    Legacy discretion switching-SSS solver for the author-style A.2 residual stack.
    This routine is intentionally lightweight and regime-generic (R >= 1):
      - solves 5 equations per regime (A.2 residuals used in training),
      - unknowns per regime are [c, XiN, XiD, p_star_aux, eta],
      - handles Markov switching via the full transition matrix P,
      - computes Delta_prev with backward (Bayes) weights from stationary P.
    As a legacy diagnostic helper, derivative terms Et[dF/dDelta], Et[dG/dDelta]
    are held at zero (same convention as stage-1 warm starts).
    """
    from .model_common import State
    from .residuals_a2 import residuals_a2
    dev = params.device
    dt = torch.float64
    R = int(params.n_regimes)
    if R <= 0:
        raise ValueError("params.n_regimes must be >= 1")
    import dataclasses as _dc
    _d = _dc.asdict(params)
    _d["dtype"] = dt
    params64 = ModelParams(**_d).to_torch()
    logA0 = torch.tensor(0.0, device=dev, dtype=dt)
    logg0 = torch.tensor(0.0, device=dev, dtype=dt)
    xi0 = torch.tensor(0.0, device=dev, dtype=dt)
    P = params64.P.to(device=dev, dtype=dt)
    theta_f = float(params64.theta)
    beta_f = float(params64.beta)
    eps_f = float(params64.eps)
    gamma_f = float(params64.gamma)
    omega_f = float(params64.omega)
    M_f = float(params64.M)
    gbar = torch.tensor(float(params64.g_bar), device=dev, dtype=dt)
    eta_levels = torch.tensor(params64.eta_by_regime, device=dev, dtype=dt)
    def _stationary_dist(Pm: torch.Tensor) -> torch.Tensor:
        I = torch.eye(R, device=dev, dtype=dt)
        A = torch.cat([I - Pm.T, torch.ones((1, R), device=dev, dtype=dt)], dim=0)
        b = torch.cat([torch.zeros((R,), device=dev, dtype=dt), torch.ones((1,), device=dev, dtype=dt)], dim=0)
        pi = torch.linalg.lstsq(A, b).solution
        pi = torch.clamp(pi, min=1e-16)
        return pi / torch.clamp(pi.sum(), min=1e-16)
    pi_stat = _stationary_dist(P)
    # B[s_current, s_prev] = Pr(s_{t-1}=s_prev | s_t=s_current)
    B = torch.zeros((R, R), device=dev, dtype=dt)
    for s in range(R):
        numer = pi_stat * P[:, s]
        denom = torch.clamp(torch.sum(numer), min=1e-16)
        B[s, :] = numer / denom
    def _decode(Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        U = Z.view(R, 5)
        c = torch.exp(torch.clamp(U[:, 0], min=-40.0, max=40.0)) + 1e-12
        XiN = torch.exp(torch.clamp(U[:, 1], min=-40.0, max=40.0)) + 1e-14
        XiD = torch.exp(torch.clamp(U[:, 2], min=-40.0, max=40.0)) + 1e-14
        p_star_aux = torch.exp(torch.clamp(U[:, 3], min=-40.0, max=40.0)) + 1e-12
        eta = U[:, 4]
        a = torch.clamp(p_star_aux, min=1e-12).pow((eps_f - 1.0) / eps_f)
        inside = (1.0 - (1.0 - theta_f) * a) / theta_f
        inside = torch.clamp(inside, min=1e-12)
        pi_aux = inside.pow(eps_f / (eps_f - 1.0))
        one_plus_pi = torch.clamp(pi_aux, min=1e-12).pow(1.0 / eps_f)
        pi = one_plus_pi - 1.0
        A_delta = torch.eye(R, device=dev, dtype=dt) - theta_f * (torch.diag(pi_aux) @ B)
        rhs_delta = (1.0 - theta_f) * p_star_aux
        Delta = _robust_solve(A_delta, rhs_delta)
        Delta = torch.clamp(Delta, min=1e-12)
        Delta_prev = B @ Delta
        lam = c.pow(-gamma_f)
        y = c + gbar
        h = y * Delta
        w = h.pow(omega_f) / torch.clamp(lam, min=1e-12)
        pstar = M_f * XiN / torch.clamp(XiD, min=1e-12)
        zeta = -(eta * (eps_f - 1.0)) / (eps_f * torch.clamp(one_plus_pi * Delta_prev, min=1e-12))
        F = theta_f * beta_f * c.pow(-gamma_f) * one_plus_pi.pow(eps_f - 1.0) * XiD
        G = theta_f * beta_f * c.pow(-gamma_f) * one_plus_pi.pow(eps_f) * XiN
        TH = theta_f * one_plus_pi.pow(eps_f) * zeta
        ratio_next_over_cur = lam.view(1, R) / torch.clamp(lam.view(R, 1), min=1e-12)
        termN = theta_f * beta_f * one_plus_pi.pow(eps_f) * XiN
        termD = theta_f * beta_f * one_plus_pi.pow(eps_f - 1.0) * XiD
        Et_XiN = torch.sum(P * ratio_next_over_cur * termN.view(1, R), dim=1)
        Et_XiD = torch.sum(P * ratio_next_over_cur * termD.view(1, R), dim=1)
        Et_F = P @ F
        Et_G = P @ G
        Et_TH = P @ TH
        Et_dF = torch.zeros((R,), device=dev, dtype=dt)
        Et_dG = torch.zeros((R,), device=dev, dtype=dt)
        mu_num = (
            eta * (1.0 - theta_f) * (1.0 - eps_f) * p_star_aux
            - zeta * (1.0 - theta_f) * eps_f * p_star_aux.pow((eps_f + 1.0) / eps_f)
        )
        mu_den = y + c.pow(gamma_f) * Et_F
        mu = -mu_num / torch.clamp(mu_den, min=1e-12)
        return {
            "c": c,
            "XiN": XiN,
            "XiD": XiD,
            "p_star_aux": p_star_aux,
            "eta": eta,
            "pi_aux": pi_aux,
            "one_plus_pi": one_plus_pi,
            "pi": pi,
            "Delta": Delta,
            "Delta_prev": Delta_prev,
            "lam": lam,
            "w": w,
            "pstar": pstar,
            "zeta": zeta,
            "mu": mu,
            "rho": torch.zeros_like(mu),
            "Et_F": Et_F,
            "Et_G": Et_G,
            "Et_TH": Et_TH,
            "Et_XiN": Et_XiN,
            "Et_XiD": Et_XiD,
            "Et_dF": Et_dF,
            "Et_dG": Et_dG,
        }
    def _residuals(Z: torch.Tensor) -> torch.Tensor:
        d = _decode(Z)
        parts = []
        for s in range(R):
            out_s = {
                "c": d["c"][s],
                "XiN": d["XiN"][s],
                "XiD": d["XiD"][s],
                "p_star_aux": d["p_star_aux"][s],
                "pstar": d["pstar"][s],
                "Delta": d["Delta"][s],
                "w": d["w"][s],
                "zeta": d["zeta"][s],
                "eta": d["eta"][s],
            }
            st = State(
                Delta_prev=d["Delta_prev"][s],
                logA=logA0,
                loggtilde=logg0,
                xi=xi0,
                s=torch.tensor(int(s), device=dev, dtype=torch.long),
            )
            res = residuals_a2(
                params=params64,
                st=st,
                out=out_s,
                Et_F_next=d["Et_F"][s],
                Et_G_next=d["Et_G"][s],
                Et_dF_dDelta_next=d["Et_dF"][s],
                Et_dG_dDelta_next=d["Et_dG"][s],
                Et_theta_zeta_pi_next=d["Et_TH"][s],
                Et_XiN_next=d["Et_XiN"][s],
                Et_XiD_next=d["Et_XiD"][s],
            )
            parts.append(
                torch.stack(
                    [
                        res["res_c_foc"],
                        res["res_Delta_foc"],
                        res["res_XiN_rec"],
                        res["res_XiD_rec"],
                        res["res_pstar_def"],
                    ]
                )
            )
        return torch.cat(parts, dim=0)
    def _initial_guess() -> torch.Tensor:
        flex = solve_flexprice_sss(params64)
        c0 = torch.tensor([float(flex.by_regime[s]["c"]) for s in range(R)], device=dev, dtype=dt)
        lam0 = c0.pow(-gamma_f)
        paux0 = torch.ones((R,), device=dev, dtype=dt)
        pi_aux0 = torch.ones((R,), device=dev, dtype=dt)
        A_delta0 = torch.eye(R, device=dev, dtype=dt) - theta_f * (torch.diag(pi_aux0) @ B)
        Delta0 = _robust_solve(A_delta0, (1.0 - theta_f) * paux0)
        Delta0 = torch.clamp(Delta0, min=1e-12)
        y0 = c0 + gbar
        w0 = (y0 * Delta0).pow(omega_f) / torch.clamp(lam0, min=1e-12)
        ratio_next_over_cur = lam0.view(1, R) / torch.clamp(lam0.view(R, 1), min=1e-12)
        A_xid = torch.eye(R, device=dev, dtype=dt) - (P * ratio_next_over_cur * (theta_f * beta_f))
        XiD0 = _robust_solve(A_xid, y0)
        XiD0 = torch.clamp(XiD0, min=1e-10)
        one_plus_tau0 = 1.0 - float(params64.tau_bar) + eta_levels
        XiN0 = _robust_solve(A_xid, y0 * w0 * one_plus_tau0)
        XiN0 = torch.clamp(XiN0, min=1e-10)
        Delta_prev0 = B @ Delta0
        k = (eps_f - 1.0) / (eps_f * torch.clamp(Delta_prev0, min=1e-12))
        a = theta_f * (eps_f - 1.0) / (eps_f * torch.clamp(Delta_prev0, min=1e-12))
        A_eta = torch.diag(k) - beta_f * (P @ torch.diag(a))
        rhs_eta = (y0 * Delta0).pow(1.0 + omega_f) / torch.clamp(Delta0, min=1e-12)
        eta0 = _robust_solve(A_eta, rhs_eta)
        z0 = torch.zeros((R, 5), device=dev, dtype=dt)
        z0[:, 0] = torch.log(torch.clamp(c0, min=1e-12))
        z0[:, 1] = torch.log(torch.clamp(XiN0, min=1e-12))
        z0[:, 2] = torch.log(torch.clamp(XiD0, min=1e-12))
        z0[:, 3] = 0.0
        z0[:, 4] = eta0
        return z0.reshape(-1)
    Z = _initial_guess()
    best_Z = Z.detach().clone()
    best_score = float("inf")
    for _ in range(int(max_iter)):
        Z_req = Z.detach().requires_grad_(True)
        r = _residuals(Z_req)
        score = float(torch.max(torch.abs(r)).detach().cpu())
        if score < best_score:
            best_score = score
            best_Z = Z_req.detach().clone()
        if score < float(tol):
            break
        J = torch.autograd.functional.jacobian(_residuals, Z_req, create_graph=False)
        step = _robust_solve(J, -r.detach())
        alpha = float(max(1e-6, min(1.0, damping)))
        improved = False
        for _ in range(12):
            Z_try = (Z + alpha * step).detach()
            r_try = _residuals(Z_try)
            score_try = float(torch.max(torch.abs(r_try)).detach().cpu())
            if np.isfinite(score_try) and score_try < score:
                Z = Z_try
                improved = True
                break
            alpha *= 0.5
        if not improved:
            Z = (Z + 0.1 * alpha * step).detach()
    final = _decode(best_Z)
    res_final = _residuals(best_Z).reshape(R, 5)
    by_regime: Dict[int, Dict[str, float]] = {}
    for s in range(R):
        by_regime[s] = {
            "c": float(final["c"][s].item()),
            "pi": float(final["pi"][s].item()),
            "pstar": float(final["pstar"][s].item()),
            "p_star_aux": float(final["p_star_aux"][s].item()),
            "lam": float(final["lam"][s].item()),
            "w": float(final["w"][s].item()),
            "XiN": float(final["XiN"][s].item()),
            "XiD": float(final["XiD"][s].item()),
            "Delta": float(final["Delta"][s].item()),
            "Delta_prev": float(final["Delta_prev"][s].item()),
            "eta": float(final["eta"][s].item()),
            "mu": float(final["mu"][s].item()),
            "rho": float(final["rho"][s].item()),
            "zeta": float(final["zeta"][s].item()),
            "dF_dDeltaPrev": 0.0,
            "dG_dDeltaPrev": 0.0,
            "core_max_residual": float(torch.max(torch.abs(res_final[s])).item()),
        }
    return DiscretionSSS(by_regime=by_regime)


def commitment_init_from_sss(params: ModelParams) -> Dict[int, Dict[str, float]]:
    """
    Return regime-specific initial lagged co-states for commitment.
    Keys correspond to the current regime s_t used in the state vector.
    Each value contains:
        vartheta_prev:  vartheta_{t-1} * c_{t-1}^gamma  (scaled lag co-state)
        varrho_prev:    varrho_{t-1} * c_{t-1}^gamma    (scaled lag co-state)
    """
    sss = solve_commitment_sss_switching(params).by_regime
    out: Dict[int, Dict[str, float]] = {}
    for s in sorted(int(k) for k in sss.keys()):
        out[s] = {
            "vartheta_prev": float(sss[s].get("vartheta_prev", 0.0)),
            "varrho_prev": float(sss[s].get("varrho_prev", 0.0)),
        }
    return out
