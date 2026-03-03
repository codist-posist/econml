
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


def _infer_policy_input_dim(net: torch.nn.Module) -> int | None:
    try:
        for p in net.parameters():
            if p.ndim == 2:
                return int(p.shape[1])
    except Exception:
        return None
    return None


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
        pi_bar=params.pi_bar, psi=params.psi, rho_i=params.rho_i,
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
      - We reconstruct x_t = (Delta_{t-1}, logA_t, loggtilde_t, xi_t, s_t[, vartheta_prev, varrho_prev]).
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
        d_in = _infer_policy_input_dim(net) or 7
        c_prev = None
        if d_in >= 8:
            if "c" not in sim_paths:
                raise KeyError(
                    "policy='commitment' with 8D state requires sim_paths['c'] to reconstruct c_prev. "
                    "Run simulate_paths(..., store_states=True)."
                )
            c_prev = np.asarray(sim_paths["c"], dtype=np.float64)[:-1]

    rng = np.random.default_rng(int(seed))
    pool_n = K2 * B
    take = int(min(max_states, pool_n))
    flat_idx = rng.choice(pool_n, size=take, replace=False) if take < pool_n else np.arange(pool_n)
    t_idx = flat_idx // B
    b_idx = flat_idx % B

    if policy == "commitment":
        if c_prev is not None:
            X = np.stack(
                [
                    Delta_prev[t_idx, b_idx],
                    logA_c[t_idx, b_idx],
                    logg_c[t_idx, b_idx],
                    xi_c[t_idx, b_idx],
                    s_c[t_idx, b_idx].astype(np.float64),
                    vp[t_idx, b_idx],
                    rp[t_idx, b_idx],
                    c_prev[t_idx, b_idx],
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


def _state_from_policy_sss(
    params: ModelParams,
    policy: PolicyName,
    sss: Dict[str, float],
    regime: int,
    *,
    commitment_state_dim: int | None = None,
) -> torch.Tensor:
    """Build a 1xN torch state vector x consistent with the project's state ordering."""
    dev, dt = params.device, params.dtype
    s = float(int(regime))

    if policy == "commitment":
        # x = (Delta_prev, logA, logg, xi, s, vartheta_prev, varrho_prev[, c_prev])
        vartheta_prev = float(sss.get("vartheta_prev", 0.0))
        varrho_prev = float(sss.get("varrho_prev", 0.0))
        base = [float(sss.get("Delta_prev", sss["Delta"])), float(sss["logA"]), float(sss["loggtilde"]), float(sss["xi"]), s, vartheta_prev, varrho_prev]
        if (commitment_state_dim is not None and int(commitment_state_dim) >= 8) or ("c_prev" in sss):
            base.append(float(sss.get("c_prev", sss.get("c", 1.0))))
        x = torch.tensor(base, device=dev, dtype=dt).view(1, -1)
        return x

    if policy == "mod_taylor":
        # Optional author taylor-para state:
        # x=(Delta_prev, i_prev, logA, xi, logg, s, p21)
        i_prev = float(sss.get("i_prev", sss.get("i_nom", (1.0 + float(params.pi_bar)) / float(params.beta) - 1.0)))
        p21 = float(sss.get("p21", params.p21))
        if "i_prev" in sss or "p21" in sss:
            x = torch.tensor(
                [float(sss.get("Delta_prev", sss["Delta"])), i_prev, float(sss["logA"]), float(sss["xi"]), float(sss["loggtilde"]), s, p21],
                device=dev,
                dtype=dt,
            ).view(1, -1)
            return x

    # taylor, discretion (and legacy mod_taylor) share x=(Delta_prev, logA, logg, xi, s)
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
        if st.c_prev is not None:
            x_next = torch.stack(
                [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt), out["vartheta"].view(-1), out["varrho"].view(-1), out["c"].view(-1)],
                dim=-1,
            )
        else:
            x_next = torch.stack(
                [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt), out["vartheta"].view(-1), out["varrho"].view(-1)],
                dim=-1,
            )
        return x_next

    if policy == "mod_taylor" and st.i_prev is not None and st.p21 is not None:
        return torch.stack(
            [out["Delta"], out["i_nom"].view(-1), logA_n.view(-1), xi_n.view(-1), logg_n.view(-1), s_n.to(dt), st.p21.view(-1)],
            dim=-1,
        )

    return torch.stack(
        [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt)],
        dim=-1,
    )


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
    commit_dim = _infer_policy_input_dim(net) if policy == "commitment" else None
    out_by_regime: Dict[int, FixedPointCheckResult] = {}
    for r, sss in sss_by_regime.items():
        x = _state_from_policy_sss(params, policy, sss, r, commitment_state_dim=commit_dim)
        st = unpack_state(x, policy)
        out = decode_outputs(policy, net(x), floors=floors, params=params, st=st)
        x_next = _deterministic_next_state(params, policy, st, out, regime=r)
        out_by_regime[int(r)] = FixedPointCheckResult(regime=int(r), max_abs_state_diff=float((x_next - x).abs().max().item()))
    return out_by_regime


def _deterministic_terms_discretion(params: ModelParams, net: torch.nn.Module, x: torch.Tensor, *, regime: int, floors: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deterministic (innovations=0, fixed regime) Et_* terms needed for Appendix A.2 residuals.
    Returns (out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD).
    """
    x = x.clone().detach().requires_grad_(True)
    st = unpack_state(x, "discretion")
    out = decode_outputs("discretion", net(x), floors=floors, params=params, st=st)

    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)

    def f_all():
        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
        Delta_cur = _broadcast_like(out["Delta"], logA_n).expand_as(logA_n)
        xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1).view(1, -1)
        on = decode_outputs("discretion", net(xn), floors=floors, params=params, st=unpack_state(xn, "discretion"))

        lam_ratio = on["lam"] / out["lam"]
        pi_aux_n = on["pi_aux"]

        # match deqn.py (Trainer._residuals): F, G, theta_term, XiN_rec, XiD_rec
        F = params.theta * params.beta * on["c"].pow(-params.gamma) * pi_aux_n.pow((params.eps - 1.0) / params.eps) * on["XiD"]
        G = params.theta * params.beta * on["c"].pow(-params.gamma) * pi_aux_n * on["XiN"]
        theta_term = params.theta * pi_aux_n * on["zeta"]
        XiN_rec = params.beta * params.theta * lam_ratio * pi_aux_n * on["XiN"]
        XiD_rec = params.beta * params.theta * lam_ratio * pi_aux_n.pow((params.eps - 1.0) / params.eps) * on["XiD"]

        return F, G, theta_term, XiN_rec, XiD_rec

    Et_F, Et_G, Et_theta, Et_XiN, Et_XiD = f_all()

    Et_dF = torch.autograd.grad(Et_F.sum(), x, create_graph=False, retain_graph=True)[0][..., 0]
    Et_dG = torch.autograd.grad(Et_G.sum(), x, create_graph=False, retain_graph=True)[0][..., 0]

    return out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD


def _deterministic_terms_commitment(params: ModelParams, net: torch.nn.Module, x: torch.Tensor, *, regime: int, floors: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deterministic (innovations=0, fixed regime) Et_* terms needed for Appendix A.3 residuals.
    Returns (out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta_zeta_pi).
    """
    x = x.clone().detach()
    st = unpack_state(x, "commitment")
    out = decode_outputs("commitment", net(x), floors=floors, params=params, st=st)

    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)
    gamma = params.gamma

    # build next state like in deqn.py
    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
    Delta_cur = _broadcast_like(out["Delta"], logA_n).expand_as(logA_n)
    vp_cur = _broadcast_like(out["vartheta"], logA_n).expand_as(logA_n)
    rp_cur = _broadcast_like(out["varrho"], logA_n).expand_as(logA_n)
    if st.c_prev is not None:
        c_prev_cur = _broadcast_like(out["c"], logA_n).expand_as(logA_n)
        xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vp_cur, rp_cur, c_prev_cur], dim=-1).view(1, -1)
    else:
        xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vp_cur, rp_cur], dim=-1).view(1, -1)

    on = decode_outputs("commitment", net(xn), floors=floors, params=params, st=unpack_state(xn, "commitment"))

    Lambda = params.beta * (on["lam"] / out["lam"])
    one_plus_pi = on["one_plus_pi"]

    # match deqn.py (Trainer._residuals): XiN_rec, XiD_rec, termN, termD, theta_term
    Et_XiN = params.theta * Lambda * on["pi_aux"] * on["XiN"]
    Et_XiD = params.theta * Lambda * on["pi_aux"].pow((params.eps - 1.0) / params.eps) * on["XiD"]

    c_tg = out["c"].view(-1, 1, 1)
    termN = params.beta * params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * on["pi_aux"] * on["XiN"]
    termD = params.beta * params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * on["pi_aux"].pow((params.eps - 1.0) / params.eps) * on["XiD"]
    theta_term = params.theta * on["pi_aux"] * on["zeta"]

    return out, Et_XiN, Et_XiD, termN, termD, theta_term


def residuals_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, ResidualCheckResult]:
    """Compatibility wrapper: use switching-consistent residual checks."""
    return residuals_check_switching_consistent(
        params,
        net,
        policy=policy,
        sss_by_regime=sss_by_regime,
        floors=floors,
    )


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
    """Residuals check at a switching-consistent SSS-by-regime.

    Uses the same residual evaluator as training (``Trainer._residuals``),
    so diagnostics stay aligned with the active model equations.
    """
    _ = implied_deriv_max_fp_iter
    _ = implied_deriv_tol
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    params = _params_with_net_dtype(params, net)
    commit_dim = _infer_policy_input_dim(net) if policy == "commitment" else None

    cfg_sim = TrainConfig.author_like(policy=policy, seed=0)
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net)

    results: Dict[int, ResidualCheckResult] = {}
    for r, sss in sss_by_regime.items():
        rr = int(r)
        x = _state_from_policy_sss(params, policy, sss, rr, commitment_state_dim=commit_dim)
        ctx = torch.enable_grad() if policy == "discretion" else torch.inference_mode()
        with ctx:
            resid_vec = trainer._residuals(x).view(-1)

        res_cpu = {k: float(v.detach().cpu().item()) for k, v in zip(trainer.res_keys, resid_vec)}
        max_abs = max(abs(v) for v in res_cpu.values()) if res_cpu else float("nan")
        results[rr] = ResidualCheckResult(regime=rr, max_abs_residual=float(max_abs), residuals=res_cpu)

    return results
