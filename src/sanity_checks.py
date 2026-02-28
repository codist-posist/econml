
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import time



def _broadcast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape x to broadcast over ref (keep batch dim, add singleton dims)."""
    if ref.dim() <= 1:
        return x.view(-1)
    return x.view(-1, *([1] * (ref.dim() - 1)))

from .config import ModelParams, PolicyName
from .model_common import State, unpack_state, shock_laws_of_motion
from .transforms import decode_outputs
from .residuals_a1 import residuals_a1
from .residuals_a2 import residuals_a2
from .residuals_a3 import residuals_a3

# For trajectory (ergodic) residual checks we reuse the same residual evaluator as training.
from .deqn import Trainer, TrainConfig, residual_metrics, residual_metrics_by_regime


def _params_with_net_dtype(params: ModelParams, net: torch.nn.Module) -> ModelParams:
    """Return params cast to network dtype when needed."""
    try:
        net_dt = next(net.parameters()).dtype
    except StopIteration:
        return params
    if net_dt == params.dtype:
        return params
    return ModelParams(
        beta=params.beta, gamma=params.gamma, omega=params.omega,
        theta=params.theta, eps=params.eps, tau_bar=params.tau_bar,
        rho_A=params.rho_A, rho_tau=params.rho_tau, rho_g=params.rho_g,
        sigma_A=params.sigma_A, sigma_tau=params.sigma_tau, sigma_g=params.sigma_g,
        g_bar=params.g_bar, eta_bar=params.eta_bar,
        bad_state=params.bad_state,
        p12=params.p12, p21=params.p21,
        pi_bar=params.pi_bar, psi=params.psi,
        device=params.device, dtype=net_dt,
    ).to_torch()


@dataclass
class TrajectoryResidualCheckResult:
    """Residual diagnostics evaluated on a (possibly long) simulated path."""

    n_states_evaluated: int
    tol: float
    metrics: Dict[str, float]


def trajectory_residuals_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sim_paths: Dict[str, "np.ndarray"],
    rbar_by_regime: Optional[torch.Tensor] = None,
    gh_n: int = 3,
    tol: float = 1e-3,
    max_states: int = 50_000,
    batch_size: int = 4096,
    seed: int = 0,
) -> TrajectoryResidualCheckResult:
    """Evaluate equilibrium residuals on states reconstructed from a stored simulation.

    This is a *post-training* quality check. It answers: does the trained policy satisfy the
    model equations (Appendix A / B) on the ergodic region actually visited in simulation?

    Notes:
      - We reconstruct x_t = (Delta_{t-1}, logA_t, loggtilde_t, xi_t, s_t[, vp_prev, rp_prev]).
      - Delta_{t-1} is reconstructed by lagging the stored Delta series. The first stored period
        is dropped to avoid an undefined lag.
      - Residuals are computed using the same expectation operator as training (Appendix B).
    """
    import numpy as np  # local import

    required = {"Delta", "logA", "loggtilde", "xi", "s"}
    missing = required.difference(sim_paths.keys())
    if missing:
        raise KeyError(
            "trajectory_residuals_check requires sim_paths with keys %s; missing %s. "
            "Run simulate_paths(..., store_states=True)." % (sorted(required), sorted(missing))
        )

    params = _params_with_net_dtype(params, net)
    dev, dt = params.device, params.dtype

    Delta = np.asarray(sim_paths["Delta"], dtype=np.float64)
    logA = np.asarray(sim_paths["logA"], dtype=np.float64)
    logg = np.asarray(sim_paths["loggtilde"], dtype=np.float64)
    xi = np.asarray(sim_paths["xi"], dtype=np.float64)
    s = np.asarray(sim_paths["s"], dtype=np.int64)

    if Delta.shape != logA.shape or Delta.shape != logg.shape or Delta.shape != xi.shape or Delta.shape != s.shape:
        raise ValueError("sim_paths arrays must have identical shapes for Delta/logA/loggtilde/xi/s")

    K, B = Delta.shape
    if K < 2:
        raise ValueError("Need at least 2 stored periods to build Delta_prev by lagging Delta")

    Delta_prev = Delta[:-1]
    logA_c = logA[1:]
    logg_c = logg[1:]
    xi_c = xi[1:]
    s_c = s[1:]
    K2 = K - 1

    if policy == "commitment":
        for kreq in ["vartheta_prev", "varrho_prev"]:
            if kreq not in sim_paths:
                raise KeyError(
                    f"policy='commitment' requires sim_paths['{kreq}'] to reconstruct x. "
                    "Run simulate_paths(..., store_states=True)."
                )
        vp = np.asarray(sim_paths["vartheta_prev"], dtype=np.float64)[1:]
        rp = np.asarray(sim_paths["varrho_prev"], dtype=np.float64)[1:]
        if vp.shape != Delta_prev.shape or rp.shape != Delta_prev.shape:
            raise ValueError("vartheta_prev/varrho_prev must have same shape as Delta")

    rng = np.random.default_rng(int(seed))
    pool_n = K2 * B
    take = int(min(max_states, pool_n))
    flat_idx = rng.choice(pool_n, size=take, replace=False) if take < pool_n else np.arange(pool_n)
    t_idx = flat_idx // B
    b_idx = flat_idx % B

    if policy == "commitment":
        X = np.stack(
            [
                Delta_prev[t_idx, b_idx],
                logA_c[t_idx, b_idx],
                logg_c[t_idx, b_idx],
                xi_c[t_idx, b_idx],
                s_c[t_idx, b_idx].astype(np.float64),
                vp[t_idx, b_idx],
                rp[t_idx, b_idx],
            ],
            axis=1,
        )
    else:
        X = np.stack(
            [
                Delta_prev[t_idx, b_idx],
                logA_c[t_idx, b_idx],
                logg_c[t_idx, b_idx],
                xi_c[t_idx, b_idx],
                s_c[t_idx, b_idx].astype(np.float64),
            ],
            axis=1,
        )

    x = torch.tensor(X, device=dev, dtype=dt)

    if params.device == "cpu":
        cfg_sim = TrainConfig.dev(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    else:
        cfg_sim = TrainConfig.full(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, gh_n=int(gh_n), rbar_by_regime=rbar_by_regime)

    resids = []
    ctx = torch.enable_grad() if policy == "discretion" else torch.inference_mode()
    with ctx:
        for i in range(0, x.shape[0], int(batch_size)):
            xb = x[i : i + int(batch_size)]
            rb = trainer._residuals(xb)
            resids.append(rb.detach())
    resid = torch.cat(resids, dim=0)

    keys = trainer.res_keys
    m = residual_metrics(resid, keys, tol=float(tol))
    m_reg = residual_metrics_by_regime(x, resid, keys, tol=float(tol), policy=policy)
    m.update(m_reg)

    return TrajectoryResidualCheckResult(n_states_evaluated=int(x.shape[0]), tol=float(tol), metrics=m)


@dataclass
class FixedPointCheckResult:
    regime: int
    max_abs_state_diff: float


@dataclass
class ResidualCheckResult:
    regime: int
    max_abs_residual: float
    residuals: Dict[str, float]


def _state_from_policy_sss(params: ModelParams, policy: PolicyName, sss: Dict[str, float], regime: int) -> torch.Tensor:
    """Build a 1xN torch state vector x consistent with the project's state ordering."""
    dev, dt = params.device, params.dtype
    s = float(int(regime))

    if policy == "commitment":
        # x = (Delta_prev, logA, logg, xi, s, vp_prev, rp_prev)
        # where vp_prev = vartheta_prev * c_prev^gamma, rp_prev = varrho_prev * c_prev^gamma.
        # sss_from_policy stores vartheta_prev/varrho_prev already in that representation.
        vp_prev = float(sss.get("vartheta_prev", 0.0))
        rp_prev = float(sss.get("varrho_prev", 0.0))
        x = torch.tensor(
            [float(sss.get("Delta_prev", sss["Delta"])), float(sss["logA"]), float(sss["loggtilde"]), float(sss["xi"]), s, vp_prev, rp_prev],
            device=dev,
            dtype=dt,
        ).view(1, -1)
        return x

    # taylor, mod_taylor, discretion share x=(Delta_prev, logA, logg, xi, s)
    x = torch.tensor(
        [float(sss.get("Delta_prev", sss["Delta"])), float(sss["logA"]), float(sss["loggtilde"]), float(sss["xi"]), s],
        device=dev,
        dtype=dt,
    ).view(1, -1)
    return x


def _deterministic_next_state(
    params: ModelParams,
    policy: PolicyName,
    st,
    out: Dict[str, torch.Tensor],
    *,
    regime: int,
) -> torch.Tensor:
    """Compute x_{t+1} under zero innovations and fixed regime (as in the paper's SSS definition)."""
    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)

    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)

    if policy == "commitment":
        gamma = params.gamma
        vp_n = out["vartheta"] * out["c"].pow(gamma)
        rp_n = out["varrho"] * out["c"].pow(gamma)
        x_next = torch.stack(
            [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt), vp_n.view(-1), rp_n.view(-1)],
            dim=-1,
        )
        return x_next

    x_next = torch.stack(
        [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt)],
        dim=-1,
    )
    return x_next


def fixed_point_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, FixedPointCheckResult]:
    """
    Fixed point check at the SSS computed from the policy:
      - hold regime fixed
      - set innovations to zero
      - one-step deterministic transition should satisfy x_{t+1} ≈ x_t
    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    params = _params_with_net_dtype(params, net)
    out_by_regime: Dict[int, FixedPointCheckResult] = {}
    for r, sss in sss_by_regime.items():
        x = _state_from_policy_sss(params, policy, sss, r)
        out = decode_outputs(policy, net(x), floors=floors)
        st = unpack_state(x, policy)
        x_next = _deterministic_next_state(params, policy, st, out, regime=r)
        out_by_regime[int(r)] = FixedPointCheckResult(regime=int(r), max_abs_state_diff=float((x_next - x).abs().max().item()))
    return out_by_regime


def _deterministic_terms_discretion(params: ModelParams, net: torch.nn.Module, x: torch.Tensor, *, regime: int, floors: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deterministic (innovations=0, fixed regime) Et_* terms needed for Appendix A.2 residuals.
    Returns (out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD).
    """
    x = x.clone().detach().requires_grad_(True)
    out = decode_outputs("discretion", net(x), floors=floors)
    st = unpack_state(x, "discretion")

    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)

    def f_all():
        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
        Delta_cur = _broadcast_like(out["Delta"], logA_n).expand_as(logA_n)
        xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1).view(1, -1)
        on = decode_outputs("discretion", net(xn), floors=floors)

        Lambda = params.beta * (on["lam"] / out["lam"])
        one_plus_pi = on["one_plus_pi"]

        # match deqn.py (Trainer._residuals): F, G, theta_term, XiN_rec, XiD_rec
        F = params.theta * params.beta * on["c"].pow(-params.gamma) * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]
        G = params.theta * params.beta * on["c"].pow(-params.gamma) * one_plus_pi.pow(params.eps) * on["XiN"]
        theta_term = params.theta * one_plus_pi.pow(params.eps) * on["zeta"]
        XiN_rec = params.theta * Lambda * one_plus_pi.pow(params.eps) * on["XiN"]
        XiD_rec = params.theta * Lambda * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]

        return F, G, theta_term, XiN_rec, XiD_rec

    Et_F, Et_G, Et_theta, Et_XiN, Et_XiD = f_all()

    Et_dF = torch.autograd.grad(Et_F.sum(), out["Delta"], create_graph=False, retain_graph=True)[0]
    Et_dG = torch.autograd.grad(Et_G.sum(), out["Delta"], create_graph=False, retain_graph=True)[0]

    return out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD


def _deterministic_terms_commitment(params: ModelParams, net: torch.nn.Module, x: torch.Tensor, *, regime: int, floors: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deterministic (innovations=0, fixed regime) Et_* terms needed for Appendix A.3 residuals.
    Returns (out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta_zeta_pi).
    """
    x = x.clone().detach()
    out = decode_outputs("commitment", net(x), floors=floors)
    st = unpack_state(x, "commitment")

    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)
    gamma = params.gamma

    # build next state like in deqn.py
    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
    Delta_cur = _broadcast_like(out["Delta"], logA_n).expand_as(logA_n)
    vp_cur = _broadcast_like(out["vartheta"] * out["c"].pow(gamma), logA_n).expand_as(logA_n)
    rp_cur = _broadcast_like(out["varrho"] * out["c"].pow(gamma), logA_n).expand_as(logA_n)
    xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vp_cur, rp_cur], dim=-1).view(1, -1)

    on = decode_outputs("commitment", net(xn), floors=floors)

    Lambda = params.beta * (on["lam"] / out["lam"])
    one_plus_pi = on["one_plus_pi"]

    # match deqn.py (Trainer._residuals): XiN_rec, XiD_rec, termN, termD, theta_term
    Et_XiN = params.theta * Lambda * one_plus_pi.pow(params.eps) * on["XiN"]
    Et_XiD = params.theta * Lambda * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]

    c_tg = out["c"].view(-1, 1, 1)
    termN = params.beta * params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * one_plus_pi.pow(params.eps) * on["XiN"]
    termD = params.beta * params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]
    theta_term = params.theta * one_plus_pi.pow(params.eps) * on["zeta"]

    return out, Et_XiN, Et_XiD, termN, termD, theta_term


def residuals_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, ResidualCheckResult]:
    """
    Residuals check evaluated at the SSS fixed point under:
      - innovations=0
      - fixed regime
    For discretion: 11 residuals (Appendix A.2).
    For commitment: 13 residuals (Appendix A.3).
    For Taylor variants: 8 residuals (Appendix A.1).
    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    params = _params_with_net_dtype(params, net)
    results: Dict[int, ResidualCheckResult] = {}

    for r, sss in sss_by_regime.items():
        r = int(r)
        x = _state_from_policy_sss(params, policy, sss, r)

        if policy in ("taylor", "mod_taylor"):
            out = decode_outputs(policy, net(x), floors=floors)
            st = unpack_state(x, policy)

            # deterministic next (needed for Euler + Xi recursions)
            dev, dt = params.device, params.dtype
            eps0 = torch.zeros(1, device=dev, dtype=dt)
            s_fixed = torch.full((1,), r, device=dev, dtype=torch.long)

            logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
            xn = torch.stack([out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt)], dim=-1)

            on = decode_outputs(policy, net(xn), floors=floors)
            Lambda = params.beta * (on["lam"] / out["lam"])
            one_plus_pi = on["one_plus_pi"]
            Et_XiN = params.theta * Lambda * one_plus_pi.pow(params.eps) * on["XiN"]
            Et_XiD = params.theta * Lambda * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]
            # Euler term uses the policy rule i_t. In some runs we don't store i in `out`
            # (it's a derived diagnostic), so compute it directly from the rule here.
            if "i" in out:
                i_t = out["i"]
            else:
                from .policy_rules import i_taylor, i_modified_taylor
                if policy == "taylor":
                    i_t = i_taylor(params, out["pi"])
                else:
                    # mod_taylor needs rbar_by_regime (natural-rate steady states)
                    from .steady_states import solve_flexprice_sss, export_rbar_tensor
                    rbar_by_regime = export_rbar_tensor(params, solve_flexprice_sss(params))
                    i_t = i_modified_taylor(params, out["pi"], rbar_by_regime, st.s)
            Et_eul = params.beta * ((1.0 + i_t) / one_plus_pi) * (on["lam"] / out["lam"])

            res = residuals_a1(params, st, out, Et_XiN, Et_XiD, Et_eul)

        elif policy == "discretion":
            out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD = _deterministic_terms_discretion(params, net, x, regime=r, floors=floors)
            st = unpack_state(x, "discretion")
            res = residuals_a2(params, st, out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD)

        elif policy == "commitment":
            out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta = _deterministic_terms_commitment(params, net, x, regime=r, floors=floors)
            st = unpack_state(x, "commitment")
            res = residuals_a3(params, st, out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta)

        else:
            raise ValueError(f"Unsupported policy for residual check: {policy}")

        # to python floats
        res_f = {k: float(v.detach().abs().max().item()) for k, v in res.items()}
        max_abs = max(res_f.values()) if res_f else float("nan")
        results[r] = ResidualCheckResult(regime=r, max_abs_residual=max_abs, residuals=res_f)

    return results


# ---------------- Switching-consistent residual checks (paper Table-2 SSS) ----------------

def residuals_check_switching_consistent(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    floors: Optional[Dict[str, float]] = None,
    implied_deriv_max_fp_iter: int = 200,
    implied_deriv_tol: float = 1e-10,
) -> Dict[int, ResidualCheckResult]:
    """Residuals check at a *switching-consistent* SSS-by-regime.

    Paper-faithful (Table 2) notion:
      - expectations over next regime use params.P
      - lagged objects use backward weights Pr(s_{t-1}|s_t) implied by params.P and the stationary dist.
    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    params = _params_with_net_dtype(params, net)
    dev = params.device
    dt = params.dtype
    # Keep backward-compatible variable name used below.
    params64 = params

    P = params.P.to(device=dev, dtype=dt)  # shape [s_current, s_next] (row-stochastic)
    beta = params.beta
    theta = params.theta
    eps = params.eps
    # stationary distribution of Markov chain
    I = torch.eye(2, device=dev, dtype=dt)
    A = I - P.T
    A = torch.vstack([A, torch.ones((1, 2), device=dev, dtype=dt)])  # (3,2)
    b = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt).unsqueeze(-1)  # (3,1)
    pi_stat = torch.linalg.lstsq(A, b).solution.squeeze(-1)  # (2,)

    def backward_weights(curr_s: int) -> torch.Tensor:
        # Pr(s_{t-1}=j | s_t=curr_s) ∝ pi_stat[j] * P[j, curr_s]
        w = torch.stack([pi_stat[0] * P[0, curr_s], pi_stat[1] * P[1, curr_s]])
        return w / w.sum()

    bw0 = backward_weights(0)
    bw1 = backward_weights(1)

    # Build policy states and decode outputs at the two regime points (from provided SSS dicts)
    b0 = sss_by_regime[0]
    b1 = sss_by_regime[1]

    x0 = _state_from_policy_sss(params64, policy, b0, 0)
    x1 = _state_from_policy_sss(params64, policy, b1, 1)

    out0 = decode_outputs(policy, net(x0), floors=floors)
    out1 = decode_outputs(policy, net(x1), floors=floors)

    # Cast to float64
    out0 = {k: v.to(device=dev, dtype=dt) for k, v in out0.items()}
    out1 = {k: v.to(device=dev, dtype=dt) for k, v in out1.items()}

    def scal(out: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return out[k].view(())

    # regime 0 values
    c0, pi0, pstar0, lam0 = scal(out0,"c"), scal(out0,"pi"), scal(out0,"pstar"), scal(out0,"lam")
    w0, XiN0, XiD0, D0 = scal(out0,"w"), scal(out0,"XiN"), scal(out0,"XiD"), scal(out0,"Delta")
    zeta0 = scal(out0,"zeta")
    # regime 1 values
    c1, pi1, pstar1, lam1 = scal(out1,"c"), scal(out1,"pi"), scal(out1,"pstar"), scal(out1,"lam")
    w1, XiN1, XiD1, D1 = scal(out1,"w"), scal(out1,"XiN"), scal(out1,"XiD"), scal(out1,"Delta")
    zeta1 = scal(out1,"zeta")

    def Et_from_regime_values(x0v: torch.Tensor, x1v: torch.Tensor):
        # Row-stochastic: P[s_current, s_next]
        return (
            P[0,0] * x0v + P[0,1] * x1v,
            P[1,0] * x0v + P[1,1] * x1v,
        )

    # Backward-weighted lag for Delta
    DP0 = bw0[0]*D0 + bw0[1]*D1
    DP1 = bw1[0]*D0 + bw1[1]*D1

    results: Dict[int, ResidualCheckResult] = {}

    if policy == "discretion":
        # Definitions used by the discretion SSS solver (Appendix A.2)
        def F_val(c, pi, XiD):
            return theta * beta * c.pow(-params64.gamma) * (1.0 + pi).pow(eps - 1.0) * XiD

        def G_val(c, pi, XiN):
            return theta * beta * c.pow(-params64.gamma) * (1.0 + pi).pow(eps) * XiN

        F0 = F_val(c0, pi0, XiD0); F1 = F_val(c1, pi1, XiD1)
        G0 = G_val(c0, pi0, XiN0); G1 = G_val(c1, pi1, XiN1)
        TH0 = theta * (1.0 + pi0).pow(eps) * zeta0
        TH1 = theta * (1.0 + pi1).pow(eps) * zeta1

        Et_F_0, Et_F_1 = Et_from_regime_values(F0, F1)
        Et_G_0, Et_G_1 = Et_from_regime_values(G0, G1)
        Et_TH_0, Et_TH_1 = Et_from_regime_values(TH0, TH1)

        # Xi expectations (matches steady_states.solve_discretion_sss_switching)
        Et_XiN_0 = P[0,0] * (theta * (beta * lam0 / lam0) * (1.0 + pi0).pow(eps) * XiN0) + \
                  P[0,1] * (theta * (beta * lam1 / lam0) * (1.0 + pi1).pow(eps) * XiN1)
        Et_XiD_0 = P[0,0] * (theta * (beta * lam0 / lam0) * (1.0 + pi0).pow(eps - 1.0) * XiD0) + \
                  P[0,1] * (theta * (beta * lam1 / lam0) * (1.0 + pi1).pow(eps - 1.0) * XiD1)

        Et_XiN_1 = P[1,0] * (theta * (beta * lam0 / lam1) * (1.0 + pi0).pow(eps) * XiN0) + \
                  P[1,1] * (theta * (beta * lam1 / lam1) * (1.0 + pi1).pow(eps) * XiN1)
        Et_XiD_1 = P[1,0] * (theta * (beta * lam0 / lam1) * (1.0 + pi0).pow(eps - 1.0) * XiD0) + \
                  P[1,1] * (theta * (beta * lam1 / lam1) * (1.0 + pi1).pow(eps - 1.0) * XiD1)

        # pack u vectors
        def pack(out: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.stack([scal(out,"c"), scal(out,"pi"), scal(out,"pstar"), scal(out,"lam"), scal(out,"w"),
                                scal(out,"XiN"), scal(out,"XiD"), scal(out,"Delta"), scal(out,"mu"), scal(out,"rho"), scal(out,"zeta")])

        u0 = pack(out0)
        u1 = pack(out1)

        def unpack_u(u_vec: torch.Tensor):
            return u_vec[0],u_vec[1],u_vec[2],u_vec[3],u_vec[4],u_vec[5],u_vec[6],u_vec[7],u_vec[8],u_vec[9],u_vec[10]

        # exps dict passed into residuals_a2
        exps = {
            0: {"Et_F": Et_F_0, "Et_G": Et_G_0, "Et_TH": Et_TH_0, "Et_XiN": Et_XiN_0, "Et_XiD": Et_XiD_0},
            1: {"Et_F": Et_F_1, "Et_G": Et_G_1, "Et_TH": Et_TH_1, "Et_XiN": Et_XiN_1, "Et_XiD": Et_XiD_1},
        }

        keys = ["res_c_foc", "res_pi_foc", "res_pstar_foc", "res_Delta_foc",
                "res_c_lam", "res_labor", "res_XiN_rec", "res_XiD_rec",
                "res_pstar_def", "res_calvo", "res_Delta_law"]

        # read common stationary continuous states from dicts (should match across regimes)
        logA = torch.tensor(float(b0.get("logA", 0.0)), device=dev, dtype=dt)
        logg = torch.tensor(float(b0.get("loggtilde", 0.0)), device=dev, dtype=dt)
        xi   = torch.tensor(float(b0.get("xi", 0.0)), device=dev, dtype=dt)

        def implied_dF_dG(s: int, u_vec: torch.Tensor, DP: torch.Tensor, Et_dF: torch.Tensor, Et_dG: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            u_req = u_vec.clone().detach().requires_grad_(True)
            dp_req = DP.clone().detach().requires_grad_(True)

            def f(uu: torch.Tensor, dd: torch.Tensor):
                c,pi,pstar,lam,w,XiN,XiD,Delta,mu,rho,zeta = unpack_u(uu)
                out2={"c":c,"pi":pi,"pstar":pstar,"lam":lam,"w":w,"XiN":XiN,"XiD":XiD,"Delta":Delta,"mu":mu,"rho":rho,"zeta":zeta}
                st2=State(Delta_prev=dd, logA=logA, loggtilde=logg, xi=xi, s=torch.tensor(s, device=dev, dtype=torch.long))
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

            Ju = torch.autograd.functional.jacobian(lambda uu: f(uu, dp_req), u_req, create_graph=False)
            Jdp = torch.autograd.functional.jacobian(lambda dd: f(u_req, dd), dp_req, create_graph=False).reshape(-1,1)
            du = -torch.linalg.solve(Ju, Jdp).reshape(-1)

            c,pi,pstar,lam,w,XiN,XiD,Delta,mu,rho,zeta = unpack_u(u_req)

            dF = (theta*beta) * (
                (-params64.gamma) * c.pow(-params64.gamma - 1.0) * (1.0 + pi).pow(eps - 1.0) * XiD * du[0]
                + c.pow(-params64.gamma) * (eps - 1.0) * (1.0 + pi).pow(eps - 2.0) * XiD * du[1]
                + c.pow(-params64.gamma) * (1.0 + pi).pow(eps - 1.0) * du[6]
            )
            dG = (theta*beta) * (
                (-params64.gamma) * c.pow(-params64.gamma - 1.0) * (1.0 + pi).pow(eps) * XiN * du[0]
                + c.pow(-params64.gamma) * eps * (1.0 + pi).pow(eps - 1.0) * XiN * du[1]
                + c.pow(-params64.gamma) * (1.0 + pi).pow(eps) * du[5]
            )
            return dF.detach(), dG.detach()

        # implied derivative fixed point
        dF0 = torch.zeros((), device=dev, dtype=dt)
        dF1 = torch.zeros((), device=dev, dtype=dt)
        dG0 = torch.zeros((), device=dev, dtype=dt)
        dG1 = torch.zeros((), device=dev, dtype=dt)

        t0 = time.time()
        n_fp_iter = 0
        last_diff = None
        for _ in range(implied_deriv_max_fp_iter):
            n_fp_iter += 1
            Et_dF_0, Et_dF_1 = Et_from_regime_values(dF0, dF1)
            Et_dG_0, Et_dG_1 = Et_from_regime_values(dG0, dG1)
            ndF0, ndG0 = implied_dF_dG(0, u0, DP0, Et_dF_0, Et_dG_0)
            ndF1, ndG1 = implied_dF_dG(1, u1, DP1, Et_dF_1, Et_dG_1)
            diff = torch.max(torch.stack([
                torch.abs(ndF0 - dF0), torch.abs(ndF1 - dF1),
                torch.abs(ndG0 - dG0), torch.abs(ndG1 - dG1)
            ]))
            last_diff = float(diff.item())
            dF0, dF1, dG0, dG1 = ndF0, ndF1, ndG0, ndG1
            if last_diff < implied_deriv_tol:
                break
        t1 = time.time()
        if last_diff is None:
            last_diff = float('nan')
        print(f"[switching residual check] implied dF/dG fixed point: iters={n_fp_iter}, last_diff={last_diff:.3e}, time={t1-t0:.2f}s")

        Et_dF_0, Et_dF_1 = Et_from_regime_values(dF0, dF1)
        Et_dG_0, Et_dG_1 = Et_from_regime_values(dG0, dG1)

        for s in [0,1]:
            DP = DP0 if s==0 else DP1
            u = u0 if s==0 else u1
            c,pi,pstar,lam,w,XiN,XiD,Delta,mu,rho,zeta = unpack_u(u)
            out_small={"c":c,"pi":pi,"pstar":pstar,"lam":lam,"w":w,"XiN":XiN,"XiD":XiD,"Delta":Delta,"mu":mu,"rho":rho,"zeta":zeta}
            st = State(Delta_prev=DP, logA=logA, loggtilde=logg, xi=xi, s=torch.tensor(s, device=dev, dtype=torch.long))
            res = residuals_a2(
                params=params64,
                st=st,
                out=out_small,
                Et_F_next=exps[s]["Et_F"],
                Et_G_next=exps[s]["Et_G"],
                Et_dF_dDelta_next=Et_dF_0 if s==0 else Et_dF_1,
                Et_dG_dDelta_next=Et_dG_0 if s==0 else Et_dG_1,
                Et_theta_zeta_pi_next=exps[s]["Et_TH"],
                Et_XiN_next=exps[s]["Et_XiN"],
                Et_XiD_next=exps[s]["Et_XiD"],
            )
            res_cpu = {k: float(v.detach().cpu().item()) for k,v in res.items()}
            max_abs = max(abs(v) for v in res_cpu.values())
            results[s] = ResidualCheckResult(regime=s, max_abs_residual=float(max_abs), residuals=res_cpu)

        return results

    if policy == "commitment":
        gamma = params64.gamma

        # backward-weighted lagged co-states (vp_prev, rp_prev) with the code's convention:
        # vp = vartheta * c^gamma, rp = varrho * c^gamma
        vp0 = scal(out0, "vartheta") * c0.pow(gamma)
        vp1 = scal(out1, "vartheta") * c1.pow(gamma)
        rp0 = scal(out0, "varrho")   * c0.pow(gamma)
        rp1 = scal(out1, "varrho")   * c1.pow(gamma)

        vp_prev0 = bw0[0]*vp0 + bw0[1]*vp1
        rp_prev0 = bw0[0]*rp0 + bw0[1]*rp1
        vp_prev1 = bw1[0]*vp0 + bw1[1]*vp1
        rp_prev1 = bw1[0]*rp0 + bw1[1]*rp1

        def Et_terms(curr_s: int, c_curr: torch.Tensor, lam_curr: torch.Tensor):
            Lam0 = beta * (lam0 / lam_curr)
            Lam1 = beta * (lam1 / lam_curr)
            # Forward transition probabilities for E_t[·_{t+1} | s_t=curr_s].
            p0w = P[curr_s, 0]
            p1w = P[curr_s, 1]

            Et_XiN = theta * (p0w * (Lam0 * (1.0 + pi0).pow(eps)     * XiN0) +
                              p1w * (Lam1 * (1.0 + pi1).pow(eps)     * XiN1))
            Et_XiD = theta * (p0w * (Lam0 * (1.0 + pi0).pow(eps-1.0) * XiD0) +
                              p1w * (Lam1 * (1.0 + pi1).pow(eps-1.0) * XiD1))
            Et_TH  = (p0w * (theta * (1.0 + pi0).pow(eps) * zeta0) +
                      p1w * (theta * (1.0 + pi1).pow(eps) * zeta1))

            Et_termN = beta * theta * gamma * c_curr.pow(gamma - 1.0) * (
                p0w * (c0.pow(-gamma) * (1.0 + pi0).pow(eps)       * XiN0) +
                p1w * (c1.pow(-gamma) * (1.0 + pi1).pow(eps)       * XiN1)
            )
            Et_termD = beta * theta * gamma * c_curr.pow(gamma - 1.0) * (
                p0w * (c0.pow(-gamma) * (1.0 + pi0).pow(eps - 1.0) * XiD0) +
                p1w * (c1.pow(-gamma) * (1.0 + pi1).pow(eps - 1.0) * XiD1)
            )
            return Et_XiN, Et_XiD, Et_termN, Et_termD, Et_TH

        logA = torch.tensor(float(b0.get("logA", 0.0)), device=dev, dtype=dt)
        logg = torch.tensor(float(b0.get("loggtilde", 0.0)), device=dev, dtype=dt)
        xi   = torch.tensor(float(b0.get("xi", 0.0)), device=dev, dtype=dt)

        for s in [0,1]:
            DP = DP0 if s==0 else DP1
            out = out0 if s==0 else out1
            c = c0 if s==0 else c1
            lam = lam0 if s==0 else lam1
            vp_prev = vp_prev0 if s==0 else vp_prev1
            rp_prev = rp_prev0 if s==0 else rp_prev1

            Et_XiN, Et_XiD, Et_termN, Et_termD, Et_TH = Et_terms(s, c, lam)

            st = State(
                Delta_prev=DP,
                logA=logA,
                loggtilde=logg,
                xi=xi,
                s=torch.tensor(s, device=dev, dtype=torch.long),
                vartheta_prev=vp_prev,
                varrho_prev=rp_prev,
            )

            out_small = {k: v.view(()) for k,v in out.items() if k in ["c","pi","pstar","lam","w","XiN","XiD","Delta","mu","nu","zeta","vartheta","varrho"]}
            res = residuals_a3(params64, st, out_small, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_TH)
            res_cpu = {k: float(v.detach().cpu().item()) for k,v in res.items()}
            max_abs = max(abs(v) for v in res_cpu.values())
            results[s] = ResidualCheckResult(regime=s, max_abs_residual=float(max_abs), residuals=res_cpu)

        return results

    # Other policies: use the existing fixed-regime residual check.
    return residuals_check(params64, net, policy=policy, sss_by_regime=sss_by_regime, floors=floors)
