from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Callable
import numpy as np
import torch

from .config import ModelParams
from .sss_from_policy import PolicySSS, switching_policy_sss_by_regime_from_policy
from .policy_rules import i_taylor, i_modified_taylor
from .model_common import transition_probs_to_next_regimes


def _robust_solve(J: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Robust linear solve for Newton steps (handles near-singular Jacobians).

    We need this to be numerically stable because both commitment and discretion
    switching-SSS solvers rely on implicit-function / Newton updates.

    Strategy:
      1) try direct solve; if it produces non-finite values, treat as failure
      2) try Tikhonov-regularized solves (J + О»I) with increasing О»
      3) try normal-equations ridge solve: (JбµЂJ + О»I) x = JбµЂ b
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

    # 2) (J + О»I) x = b
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

    With A=1 and g=g_bar, efficient consumption Д‰ solves:
        ((Д‰ + g_bar)/A)^omega - A * Д‰^{-gamma} = 0
    which at A=1 simplifies to:
        (Д‰ + g_bar)^omega - Д‰^{-gamma} = 0

    Returns a dict with:
      - c_hat: Д‰
      - r_hat: 1/beta - 1  (non-stochastic efficient real rate)
    """
    g = float(params.g_bar)
    omega = float(params.omega)
    gamma = float(params.gamma)

    # Newton on scalar Д‰
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
        eta_s follows params.eta_by_regime (normal/bad/severe/...).
        M = eps/(eps-1)

    IMPORTANT: The second term is the reciprocal, exactly as written in the paper.
    """
    M = float(params.M)
    gbar = float(params.g_bar)
    gamma = float(params.gamma)
    omega = float(params.omega)

    n_reg = int(params.n_regimes)
    eta_levels = tuple(float(v) for v in params.eta_by_regime)
    out: Dict[int, Dict[str, float]] = {}
    for s in range(n_reg):
        if s < len(eta_levels):
            eta_s = float(eta_levels[s])
        else:
            eta_s = float(eta_levels[-1])
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
    # where m_{t+1} = (c_{t+1}/c_t)^(-gamma) and expectation is over s_{t+1} only.
    # Transition probabilities can be state-dependent (via xi), so each row is
    # evaluated at regime-specific xi stationary means.
    c_by_regime = {int(k): float(v["c"]) for k, v in out.items()}
    sigma_by_regime = tuple(float(v) for v in params.sigma_tau_by_regime)
    xi_ss_by_regime: Dict[int, float] = {}
    for s in range(n_reg):
        sig = sigma_by_regime[s] if s < len(sigma_by_regime) else sigma_by_regime[-1]
        xi_ss_by_regime[s] = float(-(sig**2) / 2.0)

    def r_star(s: int) -> float:
        c_s = c_by_regime[int(s)]
        s_t = torch.tensor([int(s)], device=params.device, dtype=torch.long)
        xi_t = torch.tensor([float(xi_ss_by_regime[int(s)])], device=params.device, dtype=params.dtype)
        p_row = transition_probs_to_next_regimes(params, s_t, xi=xi_t).view(-1).detach().cpu().numpy()
        E_m = 0.0
        for sp in range(n_reg):
            p_sp = float(p_row[sp] if sp < p_row.shape[0] else 0.0)
            c_sp = c_by_regime[int(sp)]
            E_m += p_sp * ((c_sp / c_s) ** (-gamma))
        return 1.0 / (float(params.beta) * E_m) - 1.0

    for s in range(n_reg):
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

    regime_ids = sorted(int(k) for k in flex.by_regime.keys())
    out: Dict[int, Dict[str, float]] = {}
    for s in regime_ids:
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
    vals = [float(flex.by_regime[s]["r_star"]) for s in range(int(params.n_regimes))]
    return torch.tensor(vals, device=params.device, dtype=params.dtype)


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

    Regime enters through eta_by_regime[regime], hence (1+tau).
    """
    if int(regime) < 0 or int(regime) >= int(params.n_regimes):
        raise ValueError(
            f"regime must be in [0, {int(params.n_regimes)-1}], got {regime}"
        )
    dev, dt = params.device, params.dtype

    g = torch.tensor(params.g_bar, device=dev, dtype=dt)
    eta = torch.tensor(params.eta_by_regime[int(regime)], device=dev, dtype=dt)
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

    We assign the same warm-start values to all regimes, matching the author setup.
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
    LEGACY helper: regime-conditional commitment initialization under switching.

    IMPORTANT (paper-faithful usage): The paper's "timeless" commitment results are computed
    from the optimal policy itself (a fixed point / center of the ergodic distribution), not by
    imposing a single deterministic value for lagged co-states conditional on s_t.

    This routine is kept for backward compatibility and as a warm-start diagnostic, but the
    recommended, "strictly per paper" SSS for commitment is:

        solve_commitment_sss_from_policy(params, net)

    where net is the trained commitment policy network.

    This legacy routine returns a regime-generic warm-start object:
      1) solve local commitment SS in each regime separately;
      2) form backward weights Pr(s_{t-1}=j | s_t) from the full Markov matrix;
      3) set lagged state objects as backward-weighted averages.
    """
    _ = (max_iter, tol, damping)  # kept for backward-compatible signature
    n_reg = int(params.n_regimes)
    if n_reg <= 0:
        raise ValueError("params.n_regimes must be >= 1")

    local = {s: commitment_local_ss(params, s) for s in range(n_reg)}
    P = params.P.detach().cpu().numpy().astype(np.float64)
    gamma = float(params.gamma)

    # Stationary distribution for row-stochastic P[current, next].
    pi = np.full((n_reg,), 1.0 / float(n_reg), dtype=np.float64)
    for _ in range(4096):
        nxt = pi @ P
        if np.max(np.abs(nxt - pi)) < 1e-14:
            pi = nxt
            break
        pi = nxt
    pi = np.clip(pi, 1e-16, None)
    pi = pi / float(np.sum(pi))

    Delta = np.array([float(local[s]["Delta"]) for s in range(n_reg)], dtype=np.float64)
    vp = np.array(
        [
            float(local[s]["vartheta"]) * (float(local[s]["c"]) ** gamma)
            for s in range(n_reg)
        ],
        dtype=np.float64,
    )
    rp = np.array(
        [
            float(local[s]["varrho"]) * (float(local[s]["c"]) ** gamma)
            for s in range(n_reg)
        ],
        dtype=np.float64,
    )

    by_regime: Dict[int, Dict[str, float]] = {}
    for cur in range(n_reg):
        # w_prev[j] = Pr(s_{t-1}=j | s_t=cur)
        w = pi * P[:, int(cur)]
        denom = float(np.sum(w))
        if denom <= 0.0:
            w = np.zeros((n_reg,), dtype=np.float64)
            w[int(cur)] = 1.0
        else:
            w = w / denom

        d = dict(local[cur])
        d["Delta_prev"] = float(np.dot(w, Delta))
        d["vartheta_prev"] = float(np.dot(w, vp))
        d["varrho_prev"] = float(np.dot(w, rp))
        by_regime[int(cur)] = d

    return CommitmentSSS(by_regime=by_regime)


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
    Legacy warm-start builder for discretion under switching.

    This helper is kept for compatibility and diagnostics. It does not solve the
    full switching nonlinear system; instead it constructs regime-wise initial
    values from flex-price and Taylor SSS objects in a fully n-regime-safe way.
    """
    _ = (max_iter, tol, damping)  # kept for backward-compatible signature

    n_reg = int(params.n_regimes)
    if n_reg <= 0:
        raise ValueError("params.n_regimes must be >= 1")

    flex = solve_flexprice_sss(params)
    taylor = solve_taylor_sss(params, flex)

    gamma = float(params.gamma)
    theta = float(params.theta)
    eps = float(params.eps)
    gbar = float(params.g_bar)
    M = float(params.M)
    eta_levels = tuple(float(v) for v in params.eta_by_regime)

    by_regime: Dict[int, Dict[str, float]] = {}
    for s in range(n_reg):
        f = flex.by_regime[int(s)]
        t = taylor.by_regime[int(s)]
        c = float(f["c"])
        pi = float(t["pi"])
        Delta = float(t["Delta"])
        pstar = float(t["pstar"])

        eta_s = float(eta_levels[s] if s < len(eta_levels) else eta_levels[-1])
        one_plus_tau = (1.0 - float(params.tau_bar)) + eta_s

        y = c + gbar
        lam = c ** (-gamma)
        w = 1.0 / (M * one_plus_tau)

        # Regime-local recursion closures used only as numerical warm starts.
        denom_n = max(1e-12, 1.0 - theta * beta * ((1.0 + pi) ** eps))
        denom_d = max(1e-12, 1.0 - theta * beta * ((1.0 + pi) ** (eps - 1.0)))
        XiN = y * w * one_plus_tau / denom_n
        XiD = y / denom_d

        by_regime[int(s)] = {
            "c": float(c),
            "pi": float(pi),
            "pstar": float(pstar),
            "lam": float(lam),
            "w": float(w),
            "XiN": float(XiN),
            "XiD": float(XiD),
            "Delta": float(Delta),
            "mu": 0.0,
            "rho": 0.0,
            "zeta": 0.0,
            "dF_dDeltaPrev": 0.0,
            "dG_dDeltaPrev": 0.0,
            "core_max_residual": float("nan"),
        }

    return DiscretionSSS(by_regime=by_regime)


def commitment_init_from_sss(params: ModelParams) -> Dict[int, Dict[str, float]]:
    """
    Return regime-specific initial lagged co-states for commitment.
    Keys correspond to current regime s_t used in the state vector.
    Each value contains:
        vartheta_prev:  П‘_{t-1} * c_{t-1}^Оі  (scaled lag co-state)
        varrho_prev:    П±_{t-1} * c_{t-1}^Оі  (scaled lag co-state)
    """
    sss = solve_commitment_sss_switching(params).by_regime
    out: Dict[int, Dict[str, float]] = {}
    for s in range(int(params.n_regimes)):
        out[int(s)] = {
            "vartheta_prev": float(sss[int(s)]["vartheta_prev"]),
            "varrho_prev": float(sss[int(s)]["varrho_prev"]),
        }
    return out
