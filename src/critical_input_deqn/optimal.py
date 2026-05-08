from __future__ import annotations

from typing import Dict, Tuple

import torch

from .config import (
    BaselineParams,
    COMMITMENT_OUTPUT_NAMES,
    COMMITMENT_PROMISE_NAMES,
    DISCRETION_OUTPUT_NAMES,
    OPT_CONTROL_NAMES,
    OPT_MULTIPLIER_NAMES,
    PRIVATE_RESIDUAL_NAMES,
    QMCConfig,
)
from .economics import (
    State,
    derive_free,
    fischer_burmeister,
    mc_derivative_A,
    p_x_derivative_A,
    psi,
    psi_prime,
    unpack_rule_state,
)
from .qmc import QMCNodes, poisson_icdf
from .residuals import _mean_over_nodes
from .transforms import decode_optimal_outputs


TensorDict = Dict[str, torch.Tensor]


def period_utility(C: torch.Tensor, N: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    sigma = float(p.sigma)
    varphi = float(p.varphi)
    if abs(sigma - 1.0) < 1e-10:
        u_c = torch.log(C)
    else:
        u_c = C.pow(1.0 - sigma) / (1.0 - sigma)
    return u_c - N.pow(1.0 + varphi) / (1.0 + varphi)


def decode_discretion(raw: torch.Tensor) -> TensorDict:
    return decode_optimal_outputs(raw, DISCRETION_OUTPUT_NAMES)


def decode_commitment(raw: torch.Tensor) -> TensorDict:
    return decode_optimal_outputs(raw, COMMITMENT_OUTPUT_NAMES)


def multipliers(out: TensorDict) -> torch.Tensor:
    return torch.stack([out[name] for name in OPT_MULTIPLIER_NAMES], dim=-1)


def controls(out: TensorDict) -> tuple[torch.Tensor, ...]:
    return tuple(out[name] for name in OPT_CONTROL_NAMES)


def private_residuals_free(
    z: torch.Tensor,
    out: TensorDict,
    policy_net,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    fb_epsilon: float,
    commitment: bool = False,
) -> Tuple[TensorDict, TensorDict]:
    """Private implementability residuals with the policy rate as a control."""

    z_phys = z[..., :7]
    st = unpack_rule_state(z_phys)
    drv = derive_free(st, out, params)

    z_next_phys = transition_physical_states(st, drv["A_next"], drv["Delta"], nodes, params, qmc_cfg)
    B, S, K = z_next_phys.shape
    if commitment:
        p_next = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
        p_next = p_next[:, None, :].expand(B, S, len(COMMITMENT_PROMISE_NAMES))
        z_next = torch.cat([z_next_phys, p_next], dim=-1)
        raw_next = policy_net(z_next.reshape(B * S, K + len(COMMITMENT_PROMISE_NAMES)))
        out_next = decode_commitment(raw_next)
    else:
        z_next = z_next_phys
        raw_next = policy_net(z_next.reshape(B * S, K))
        out_next = decode_discretion(raw_next)

    st_next = unpack_rule_state(z_next_phys.reshape(B * S, K))
    drv_next = derive_free(st_next, out_next, params)

    C = out["C"]
    Lambda = drv["Lambda"]
    Lambda_next = out_next["C"].pow(-float(params.sigma)).reshape(B, S)
    Pi_next = out_next["Pi"].reshape(B, S)
    S_p_next = out_next["S_p"].reshape(B, S)
    F_p_next = out_next["F_p"].reshape(B, S)
    Q_next = out_next["Q_A"].reshape(B, S)
    Mdisc = float(params.beta) * Lambda_next / Lambda[:, None]

    p_x_A_next = p_x_derivative_A(st_next.A, drv_next["p_m_eff"], drv_next["p_d"], params)
    mc_A_next = mc_derivative_A(drv_next["mc"], drv_next["p_x"], p_x_A_next, params)
    benefit_A_next = -(mc_A_next * drv_next["Delta"] * out_next["Y"]).reshape(B, S)

    res: TensorDict = {}
    res["hh_euler"] = torch.log(
        torch.clamp(
            _mean_over_nodes(float(params.beta) * out["R"][:, None] * Lambda_next / Lambda[:, None] / Pi_next),
            min=1e-12,
        )
    )
    res["labor"] = (out["N"] - drv["N_d"]) / out["N"]
    res["resource"] = (
        out["Y"]
        - out["C"]
        - drv["pm"] * drv["M"]
        - float(params.p_d) * drv["S"]
        - float(params.p_a) * psi(out["I_A"], params)
    ) / out["Y"]
    res["price_index"] = (
        1.0
        - (1.0 - float(params.theta)) * drv["p_star"].pow(1.0 - float(params.epsilon))
        - float(params.theta) * out["Pi"].pow(float(params.epsilon) - 1.0)
    )
    res["calvo_S"] = (
        out["S_p"]
        - drv["mc"] * out["Y"]
        - float(params.theta) * _mean_over_nodes(Mdisc * Pi_next.pow(float(params.epsilon)) * S_p_next)
    ) / out["S_p"]
    res["calvo_F"] = (
        out["F_p"]
        - out["Y"]
        - float(params.theta) * _mean_over_nodes(Mdisc * Pi_next.pow(float(params.epsilon) - 1.0) * F_p_next)
    ) / out["F_p"]
    cap_slack = (drv["mbar"] - drv["M"]) / torch.clamp(drv["mbar"], min=1e-12)
    res["cap_fb"] = fischer_burmeister(out["chi"], cap_slack, fb_epsilon)
    repair_gap = drv["Omega_A"] * float(params.p_a) * psi_prime(out["I_A"], params) - out["Q_A"]
    res["repair_fb"] = fischer_burmeister(out["I_A"], repair_gap, fb_epsilon)
    res["Q"] = (
        out["Q_A"]
        - _mean_over_nodes(Mdisc * (benefit_A_next + (1.0 - float(params.delta_A)) * Q_next))
    ) / out["Q_A"]
    return res, {**out, **drv, "z_next": z_next, "out_next": out_next}


def transition_physical_states(
    st: State,
    A_next: torch.Tensor,
    Delta_next: torch.Tensor,
    nodes: QMCNodes,
    p: BaselineParams,
    qmc_cfg: QMCConfig,
) -> torch.Tensor:
    B = st.D.shape[0]
    S = nodes.n
    D = st.D[:, None]
    X = st.X[:, None]
    ell_D = st.ell_D[:, None]
    ell_X = st.ell_X[:, None]
    log_Z = st.log_Z[:, None]

    lam_D = torch.exp(ell_D)
    lam_X = torch.exp(ell_X)
    n_D = poisson_icdf(nodes.u_N_D[None, :].expand(B, S), lam_D.expand(B, S), qmc_cfg.poisson_max_count)
    n_X = poisson_icdf(nodes.u_N_X[None, :].expand(B, S), lam_X.expand(B, S), qmc_cfg.poisson_max_count)

    D_next = (1.0 - float(p.delta_D)) * D + n_D * float(p.mark_D)
    X_next = (1.0 - float(p.delta_X)) * X + n_X * float(p.mark_X)
    ell_D_next = (
        (1.0 - float(p.rho_lambda_D)) * float(p.log_bar_lambda_D)
        + float(p.rho_lambda_D) * ell_D
        + float(p.kappa_D_lambda) * D
        + float(p.sigma_lambda_D) * nodes.eps_lam_D[None, :]
    )
    ell_X_next = (
        (1.0 - float(p.rho_lambda_X)) * float(p.log_bar_lambda_X)
        + float(p.rho_lambda_X) * ell_X
        + float(p.beta_X) * D
        + float(p.sigma_lambda_X) * nodes.eps_lam_X[None, :]
    )
    log_Z_next = float(p.rho_z) * log_Z + float(p.sigma_z) * nodes.eps_z[None, :]
    A_next_b = A_next[:, None].expand(B, S)
    log_Delta_next = torch.log(torch.clamp(Delta_next[:, None].expand(B, S), min=1e-12))
    return torch.stack([D_next, X_next, ell_D_next, ell_X_next, log_Z_next, A_next_b, log_Delta_next], dim=-1)


def private_residual_matrix(res: TensorDict) -> torch.Tensor:
    return torch.stack([res[name] for name in PRIVATE_RESIDUAL_NAMES], dim=-1)


def stationarity_from_lagrangian(
    lagrangian: torch.Tensor,
    out: TensorDict,
) -> TensorDict:
    grads = torch.autograd.grad(
        lagrangian.sum(),
        controls(out),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    stat: TensorDict = {}
    for name, grad in zip(OPT_CONTROL_NAMES, grads):
        if grad is None:
            grad = torch.zeros_like(out[name])
        stat[f"stat_{name}"] = grad
    return stat


def discretion_residuals(
    z: torch.Tensor,
    raw: torch.Tensor,
    policy_net,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    fb_epsilon: float,
) -> Tuple[TensorDict, TensorDict]:
    out = decode_discretion(raw)
    priv, drv = private_residuals_free(
        z,
        out,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=False,
    )
    z_next = drv["z_next"]
    B, S, K = z_next.shape
    out_next = decode_discretion(policy_net(z_next.reshape(B * S, K)))
    V_next = out_next["V"].reshape(B, S)
    U = period_utility(out["C"], out["N"], params)
    H = private_residual_matrix(priv)
    mu = multipliers(out)
    bellman = out["V"] - U - float(params.beta) * _mean_over_nodes(V_next)
    lagrangian = U + (mu * H).sum(dim=-1) + float(params.beta) * _mean_over_nodes(V_next)
    stat = stationarity_from_lagrangian(lagrangian, out)
    res: TensorDict = {f"priv_{k}": v for k, v in priv.items()}
    res["bellman"] = bellman
    res.update(stat)
    return res, drv


def commitment_promise_term(z: torch.Tensor, out: TensorDict, drv: TensorDict, params: BaselineParams) -> torch.Tensor:
    """Inherited scaled-promise term in the Ramsey stationarity conditions.

    The promise states already contain the previous-period inverse marginal
    utility normalizer.  Multiplying by current Lambda here gives the same
    current-period scaling as a raw-promise implementation that separately
    carried lagged consumption.
    """

    pE, pS, pF, pQ = [z[..., 7 + i] for i in range(4)]
    st = unpack_rule_state(z[..., :7])
    Lambda = drv["Lambda"]
    p_x_A = p_x_derivative_A(st.A, drv["p_m_eff"], drv["p_d"], params)
    mc_A = mc_derivative_A(drv["mc"], drv["p_x"], p_x_A, params)
    benefit_A = -(mc_A * drv["Delta"] * out["Y"])
    return (
        pE * Lambda / out["Pi"]
        + pS * Lambda * out["Pi"].pow(float(params.epsilon)) * out["S_p"]
        + pF * Lambda * out["Pi"].pow(float(params.epsilon) - 1.0) * out["F_p"]
        + pQ * Lambda * (benefit_A + (1.0 - float(params.delta_A)) * out["Q_A"])
    )


def commitment_promise_map(out: TensorDict, drv: TensorDict, params: BaselineParams) -> torch.Tensor:
    """Map current Ramsey multipliers into next-period scaled promise states."""

    Lambda = drv["Lambda"]
    inv_lambda = 1.0 / torch.clamp(Lambda, min=1e-12)
    return torch.stack(
        [
            out["mu_hh_euler"] * out["R"] * inv_lambda,
            out["mu_calvo_S"] * inv_lambda,
            out["mu_calvo_F"] * inv_lambda,
            out["mu_Q"] * inv_lambda,
        ],
        dim=-1,
    )


def commitment_residuals(
    zc: torch.Tensor,
    raw: torch.Tensor,
    policy_net,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    fb_epsilon: float,
) -> Tuple[TensorDict, TensorDict]:
    out = decode_commitment(raw)
    priv, drv = private_residuals_free(
        zc,
        out,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=True,
    )
    H = private_residual_matrix(priv)
    mu = multipliers(out)
    U = period_utility(out["C"], out["N"], params)
    promise_term = commitment_promise_term(zc, out, drv, params)

    z_next = drv["z_next"]
    B, S, K = z_next.shape
    raw_next = policy_net(z_next.reshape(B * S, K))
    out_next = decode_commitment(raw_next)
    priv_next, _ = private_residuals_free(
        z_next.reshape(B * S, K),
        out_next,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=True,
    )
    H_next = private_residual_matrix(priv_next).reshape(B, S, len(PRIVATE_RESIDUAL_NAMES))
    mu_next = multipliers(out_next).reshape(B, S, len(PRIVATE_RESIDUAL_NAMES))
    future_term = _mean_over_nodes((mu_next * H_next).sum(dim=-1))
    lagrangian = U + promise_term + (mu * H).sum(dim=-1) + float(params.beta) * future_term
    stat = stationarity_from_lagrangian(lagrangian, out)

    selected_mu = commitment_promise_map(out, drv, params)
    promised = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
    promise_resid = promised - selected_mu

    res: TensorDict = {f"priv_{k}": v for k, v in priv.items()}
    res.update(stat)
    for i, name in enumerate(COMMITMENT_PROMISE_NAMES):
        res[name] = promise_resid[..., i]
    return res, drv


def random_physical_step(z_phys: torch.Tensor, out: TensorDict, *, params: BaselineParams) -> torch.Tensor:
    st = unpack_rule_state(z_phys)
    drv = derive_free(st, out, params)
    lam_D = torch.exp(st.ell_D)
    lam_X = torch.exp(st.ell_X)
    n_D = torch.poisson(torch.clamp(lam_D, min=1e-12))
    n_X = torch.poisson(torch.clamp(lam_X, min=1e-12))
    D_next = (1.0 - float(params.delta_D)) * st.D + n_D * float(params.mark_D)
    X_next = (1.0 - float(params.delta_X)) * st.X + n_X * float(params.mark_X)
    ell_D_next = (
        (1.0 - float(params.rho_lambda_D)) * float(params.log_bar_lambda_D)
        + float(params.rho_lambda_D) * st.ell_D
        + float(params.kappa_D_lambda) * st.D
        + float(params.sigma_lambda_D) * torch.randn_like(st.ell_D)
    )
    ell_X_next = (
        (1.0 - float(params.rho_lambda_X)) * float(params.log_bar_lambda_X)
        + float(params.rho_lambda_X) * st.ell_X
        + float(params.beta_X) * st.D
        + float(params.sigma_lambda_X) * torch.randn_like(st.ell_X)
    )
    log_Z_next = float(params.rho_z) * st.log_Z + float(params.sigma_z) * torch.randn_like(st.log_Z)
    log_Delta_next = torch.log(torch.clamp(drv["Delta"], min=1e-12))
    return torch.stack(
        [D_next, X_next, ell_D_next, ell_X_next, log_Z_next, drv["A_next"], log_Delta_next],
        dim=-1,
    )


def simulate_optimal_episode(
    initial_z: torch.Tensor,
    policy_net,
    *,
    kind: str,
    params: BaselineParams,
    length: int,
) -> torch.Tensor:
    states = [initial_z]
    z = initial_z
    with torch.no_grad():
        for _ in range(1, int(length)):
            if kind == "discretion":
                out = decode_discretion(policy_net(z))
                z = random_physical_step(z, out, params=params)
            elif kind == "commitment":
                out = decode_commitment(policy_net(z))
                z_phys = random_physical_step(z[..., :7], out, params=params)
                p_next = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
                z = torch.cat([z_phys, p_next], dim=-1)
            else:
                raise ValueError("kind must be 'discretion' or 'commitment'.")
            states.append(z)
    return torch.stack(states, dim=0)
