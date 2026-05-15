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
    OPT_PRIVATE_RESIDUAL_NAMES,
    QMCConfig,
)
from .economics import (
    State,
    derive_free,
    adaptation_enabled,
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
EULER_RATE_FIXED_POINT_ITERS = 12
EULER_RATE_FIXED_POINT_TOL = 1e-10


def period_utility(C: torch.Tensor, N: torch.Tensor, p: BaselineParams) -> torch.Tensor:
    sigma = float(p.sigma)
    varphi = float(p.varphi)
    if abs(sigma - 1.0) < 1e-10:
        u_c = torch.log(C)
    else:
        u_c = C.pow(1.0 - sigma) / (1.0 - sigma)
    return u_c - N.pow(1.0 + varphi) / (1.0 + varphi)


def decode_discretion(raw: torch.Tensor, *, params: BaselineParams | None = None) -> TensorDict:
    return decode_optimal_outputs(raw, DISCRETION_OUTPUT_NAMES, params=params)


def decode_commitment(raw: torch.Tensor, *, params: BaselineParams | None = None) -> TensorDict:
    return decode_optimal_outputs(raw, COMMITMENT_OUTPUT_NAMES, params=params)


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
    """Private implementability residuals with the policy rate Euler-implied.

    This follows the author DEQN convention: discretion/commitment networks do
    not output the nominal policy rate.  Instead, the gross rate is recovered
    from the household Euler equation.  Because in this model R also affects
    repair costs and therefore A_{t+1}, we close the scalar feedback with a
    short differentiable fixed-point iteration.
    """

    z_phys = z[..., :7]
    st = unpack_rule_state(z_phys)
    R = torch.full_like(out["C"], float(params.bar_R))

    drv: TensorDict
    z_next: torch.Tensor
    z_next_phys: torch.Tensor
    out_next: TensorDict
    B = S = K = 0
    for _ in range(EULER_RATE_FIXED_POINT_ITERS):
        drv = derive_free(st, out, params, R=R)
        z_next, z_next_phys, out_next, B, S, K = _optimal_next_outputs(
            st,
            out,
            drv,
            policy_net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            commitment=commitment,
        )
        R_next = _euler_implied_gross_rate(out, drv, out_next, B, S, params)
        step_resid = torch.log(torch.clamp(R_next / torch.clamp(R, min=1e-12), min=1e-12))
        R = R_next
        if bool((step_resid.detach().abs().max() < EULER_RATE_FIXED_POINT_TOL).cpu()):
            break

    drv = derive_free(st, out, params, R=R)
    z_next, z_next_phys, out_next, B, S, K = _optimal_next_outputs(
        st,
        out,
        drv,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        commitment=commitment,
    )
    R_check = _euler_implied_gross_rate(out, drv, out_next, B, S, params)
    euler_rate_residual = torch.log(torch.clamp(R / torch.clamp(R_check, min=1e-12), min=1e-12))

    st_next = unpack_rule_state(z_next_phys.reshape(B * S, K))
    # The Q recursion needs next-period marginal benefits, which do not depend
    # on next-period repair financing.  Avoid a nested next-next Euler solve.
    drv_next = derive_free(st_next, out_next, params, R=torch.full_like(out_next["C"], float(params.bar_R)))

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

    y_scale = out["Y"].detach().clamp_min(1e-12)
    s_scale = out["S_p"].detach().clamp_min(1e-12)
    f_scale = out["F_p"].detach().clamp_min(1e-12)
    q_scale = 1.0 + out["Q_A"].detach().abs()

    res: TensorDict = {}
    res["resource"] = (
        out["Y"]
        - out["C"]
        - drv["pm"] * drv["M"]
        - float(params.p_d) * drv["S"]
        - float(params.p_a) * psi(drv["I_A_effective"], params)
    ) / y_scale
    price_index_lhs = (
        (1.0 - float(params.theta)) * drv["p_star"].pow(1.0 - float(params.epsilon))
        + float(params.theta) * out["Pi"].pow(float(params.epsilon) - 1.0)
    )
    res["price_index"] = torch.log(torch.clamp(price_index_lhs, min=1e-12))
    res["calvo_S"] = (
        out["S_p"]
        - drv["mc"] * out["Y"]
        - float(params.theta) * _mean_over_nodes(Mdisc * Pi_next.pow(float(params.epsilon)) * S_p_next)
    ) / s_scale
    res["calvo_F"] = (
        out["F_p"]
        - out["Y"]
        - float(params.theta) * _mean_over_nodes(Mdisc * Pi_next.pow(float(params.epsilon) - 1.0) * F_p_next)
    ) / f_scale
    if adaptation_enabled(params):
        res["Q"] = (
            out["Q_A"]
            - _mean_over_nodes(Mdisc * (benefit_A_next + (1.0 - float(params.delta_A)) * Q_next))
        ) / q_scale
    else:
        res["Q"] = out["Q_A"]
    return res, {**out, **drv, "z_next": z_next, "out_next": out_next, "R_euler_check": R_check, "euler_rate_residual": euler_rate_residual}


def _optimal_next_outputs(
    st: State,
    out: TensorDict,
    drv: TensorDict,
    policy_net,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    commitment: bool,
) -> tuple[torch.Tensor, torch.Tensor, TensorDict, int, int, int]:
    z_next_phys = transition_physical_states(st, drv["A_next"], drv["Delta"], nodes, params, qmc_cfg)
    B, S, K = z_next_phys.shape
    if commitment:
        p_next = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
        p_next = p_next[:, None, :].expand(B, S, len(COMMITMENT_PROMISE_NAMES))
        z_next = torch.cat([z_next_phys, p_next], dim=-1)
        raw_next = policy_net(z_next.reshape(B * S, K + len(COMMITMENT_PROMISE_NAMES)))
        out_next = decode_commitment(raw_next, params=params)
    else:
        z_next = z_next_phys
        raw_next = policy_net(z_next.reshape(B * S, K))
        out_next = decode_discretion(raw_next, params=params)
    return z_next, z_next_phys, out_next, B, S, K


def _euler_implied_gross_rate(
    out: TensorDict,
    drv: TensorDict,
    out_next: TensorDict,
    B: int,
    S: int,
    params: BaselineParams,
) -> torch.Tensor:
    Lambda = drv["Lambda"]
    Lambda_next = out_next["C"].pow(-float(params.sigma)).reshape(B, S)
    Pi_next = out_next["Pi"].reshape(B, S)
    sdf = _mean_over_nodes(float(params.beta) * Lambda_next / Lambda[:, None] / Pi_next)
    return 1.0 / torch.clamp(sdf, min=1e-12)


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
    return torch.stack([res[name] for name in OPT_PRIVATE_RESIDUAL_NAMES], dim=-1)


def _stationarity_scale(name: str, out: TensorDict, params: BaselineParams) -> torch.Tensor:
    """Positive local scale for FOC residuals.

    The stationarity equations are derivatives of a normalized Hamiltonian with
    respect to controls that live in different units.  Scaling by local control
    magnitudes and the main analytic derivative scale preserves the zero set
    while keeping the DEQN loss from treating, for example, the inflation FOC
    and the repair-value FOC as numerically incomparable objects.
    """

    value = out[name].detach().abs()
    if name == "C":
        lambda_scale = out["C"].detach().clamp_min(1e-12).pow(-float(params.sigma)).abs()
        return 1.0 + value + lambda_scale
    if name == "Pi":
        return 1.0 + float(params.epsilon) * value
    if name == "Q_A":
        return 1.0 + value
    if name in {"S_p", "F_p", "Y"}:
        return 1.0 + value
    return torch.ones_like(value)


def stationarity_from_lagrangian(
    lagrangian: torch.Tensor,
    out: TensorDict,
    *,
    params: BaselineParams,
) -> tuple[TensorDict, TensorDict]:
    grads = torch.autograd.grad(
        lagrangian.sum(),
        controls(out),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    stat: TensorDict = {}
    diagnostics: TensorDict = {}
    for name, grad in zip(OPT_CONTROL_NAMES, grads):
        if grad is None:
            grad = torch.zeros_like(out[name])
        scale = _stationarity_scale(name, out, params)
        stat[f"stat_{name}"] = grad / torch.clamp(scale, min=1e-12)
        diagnostics[f"raw_stat_{name}"] = grad
        diagnostics[f"stat_scale_{name}"] = scale
    return stat, diagnostics


def _discretion_costate_continuation(drv: TensorDict, params: BaselineParams) -> torch.Tensor:
    """Continuation value term for the explicit discretion envelope states."""

    z_next = drv["z_next"][..., :7]
    out_next = drv["out_next"]
    B, S, _ = z_next.shape
    xi_A_next = out_next["xi_A"].reshape(B, S).detach()
    xi_Delta_next = out_next["xi_log_Delta"].reshape(B, S).detach()
    A_next = z_next[..., 5]
    log_Delta_next = z_next[..., 6]
    return float(params.beta) * _mean_over_nodes(xi_A_next * A_next + xi_Delta_next * log_Delta_next)


def _normalized_envelope_residual(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = 1.0 + torch.maximum(value.detach().abs(), target.detach().abs())
    return (value - target) / scale


def discretion_envelope_residuals(
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
    """Envelope residuals for the two endogenous predetermined physical states.

    The envelope derivative holds the current controls and multipliers fixed,
    as in the envelope theorem, while allowing current states to move current
    feasibility objects and next-period states.  Future costates enter as
    continuation prices and are detached in the current FOC/envelope step.
    In commitment, inherited promise terms are part of the current Hamiltonian,
    but the promise states themselves do not replace physical costates.
    """

    z_req = z.detach().clone().requires_grad_(True)
    out_const = {name: value.detach() for name, value in out.items()}
    priv, drv = private_residuals_free(
        z_req,
        out_const,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=commitment,
    )
    U = period_utility(out_const["C"], drv["N"], params)
    H = private_residual_matrix(priv)
    mu = multipliers(out_const)
    envelope_objective = U + (mu * H).sum(dim=-1) + _discretion_costate_continuation(drv, params)
    if commitment:
        envelope_objective = envelope_objective + commitment_promise_term(z_req, out_const, drv, params)
    grad_z = torch.autograd.grad(
        envelope_objective.sum(),
        z_req,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    target_A = grad_z[..., 5]
    target_log_Delta = grad_z[..., 6]
    residuals = {
        "env_A": _normalized_envelope_residual(out["xi_A"], target_A),
        "env_log_Delta": _normalized_envelope_residual(out["xi_log_Delta"], target_log_Delta),
    }
    diagnostics = {
        "xi_A_target": target_A,
        "xi_log_Delta_target": target_log_Delta,
    }
    return residuals, diagnostics


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
    out = decode_discretion(raw, params=params)
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
    U = period_utility(out["C"], drv["N"], params)
    H = private_residual_matrix(priv)
    mu = multipliers(out)
    lagrangian = U + (mu * H).sum(dim=-1) + _discretion_costate_continuation(drv, params)
    stat, stat_diag = stationarity_from_lagrangian(lagrangian, out, params=params)
    env, env_diag = discretion_envelope_residuals(
        z,
        out,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
    )
    res: TensorDict = {f"priv_{k}": v for k, v in priv.items()}
    res.update(stat)
    res.update(env)
    return res, {**drv, **env_diag, **stat_diag}


def commitment_promise_term(z: torch.Tensor, out: TensorDict, drv: TensorDict, params: BaselineParams) -> torch.Tensor:
    """Inherited scaled-promise term in the Ramsey stationarity conditions.

    The promise states contain the previous-period inverse marginal-utility
    normalizer, residual normalization, and sign of the forward-looking term.
    The beta in the period-t residual is accounted for by the current-value
    time shift, so the carried promise does not include an extra beta.
    Multiplying by current Lambda and the current forward-looking object
    reconstructs the lagged Ramsey term.
    """

    pS, pF, pQ = [z[..., 7 + i] for i in range(len(COMMITMENT_PROMISE_NAMES))]
    st = unpack_rule_state(z[..., :7])
    Lambda = drv["Lambda"]
    p_x_A = p_x_derivative_A(st.A, drv["p_m_eff"], drv["p_d"], params)
    mc_A = mc_derivative_A(drv["mc"], drv["p_x"], p_x_A, params)
    benefit_A = -(mc_A * drv["Delta"] * out["Y"])
    return (
        pS * Lambda * out["Pi"].pow(float(params.epsilon)) * out["S_p"]
        + pF * Lambda * out["Pi"].pow(float(params.epsilon) - 1.0) * out["F_p"]
        + pQ * Lambda * (benefit_A + (1.0 - float(params.delta_A)) * out["Q_A"])
    )


def commitment_promise_map(out: TensorDict, drv: TensorDict, params: BaselineParams) -> torch.Tensor:
    """Map current Ramsey multipliers into next-period scaled promise states.

    The multipliers are attached to the normalized private residuals used in
    ``private_residuals_free``.  The carried promise therefore includes the
    current-value forward-looking coefficient and that residual normalization:

    calvo_S: -theta / (Lambda_t * S_t)
    calvo_F: -theta / (Lambda_t * F_t)
    Q:       -1 / (Lambda_t * (1 + |Q_t|))

    The beta in the original forward-looking residual is already accounted for
    by the shift from the period-t current-value multiplier to the period-t+1
    promise term; adding another beta here would double-discount the promise.

    With this scaling the next-period promise term can be written compactly as
    p_S * Lambda * Pi^epsilon * S
    + p_F * Lambda * Pi^(epsilon - 1) * F
    + p_Q * Lambda * (benefit_A + (1 - delta_A) * Q_A).
    """

    Lambda = drv["Lambda"]
    inv_lambda = 1.0 / torch.clamp(Lambda, min=1e-12)
    theta = float(params.theta)
    return torch.stack(
        [
            -theta * out["mu_calvo_S"] * inv_lambda / torch.clamp(out["S_p"], min=1e-12),
            -theta * out["mu_calvo_F"] * inv_lambda / torch.clamp(out["F_p"], min=1e-12),
            -out["mu_Q"] * inv_lambda / (1.0 + out["Q_A"].abs()),
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
    out = decode_commitment(raw, params=params)
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
    U = period_utility(out["C"], drv["N"], params)
    promise_term = commitment_promise_term(zc, out, drv, params)

    # The recursive commitment state carries selected current multipliers as
    # promises.  Re-evaluating the full next-period private residual block here
    # would create a nested B x S x S expectation tensor and double-count that
    # promise recursion.
    lagrangian = U + promise_term + (mu * H).sum(dim=-1) + _discretion_costate_continuation(drv, params)
    stat, stat_diag = stationarity_from_lagrangian(lagrangian, out, params=params)
    env, env_diag = discretion_envelope_residuals(
        zc,
        out,
        policy_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=True,
    )

    selected_mu = commitment_promise_map(out, drv, params)
    promised = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
    raw_promise_resid = promised - selected_mu
    promise_scale = 1.0 + torch.maximum(promised.detach().abs(), selected_mu.detach().abs())
    promise_resid = raw_promise_resid / torch.clamp(promise_scale, min=1e-12)

    res: TensorDict = {f"priv_{k}": v for k, v in priv.items()}
    res.update(stat)
    res.update(env)
    promise_diag: TensorDict = {}
    for i, name in enumerate(COMMITMENT_PROMISE_NAMES):
        res[name] = promise_resid[..., i]
        promise_diag[f"raw_{name}"] = raw_promise_resid[..., i]
        promise_diag[f"{name}_scale"] = promise_scale[..., i]
    return res, {**drv, **env_diag, **stat_diag, **promise_diag}


def random_physical_step(
    z_phys: torch.Tensor,
    out: TensorDict,
    *,
    params: BaselineParams,
    drv: TensorDict | None = None,
) -> torch.Tensor:
    st = unpack_rule_state(z_phys)
    if drv is None:
        drv = derive_free(st, out, params, R=torch.full_like(out["C"], float(params.bar_R)))
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
    nodes: QMCNodes | None = None,
    qmc_cfg: QMCConfig | None = None,
) -> torch.Tensor:
    states = [initial_z]
    z = initial_z
    with torch.no_grad():
        for _ in range(1, int(length)):
            if kind == "discretion":
                out = decode_discretion(policy_net(z), params=params)
                drv = None
                if nodes is not None and qmc_cfg is not None:
                    _, drv = private_residuals_free(
                        z,
                        out,
                        policy_net,
                        nodes,
                        params=params,
                        qmc_cfg=qmc_cfg,
                        fb_epsilon=0.0,
                        commitment=False,
                    )
                z = random_physical_step(z, out, params=params, drv=drv)
            elif kind == "commitment":
                out = decode_commitment(policy_net(z), params=params)
                drv = None
                if nodes is not None and qmc_cfg is not None:
                    _, drv = private_residuals_free(
                        z,
                        out,
                        policy_net,
                        nodes,
                        params=params,
                        qmc_cfg=qmc_cfg,
                        fb_epsilon=0.0,
                        commitment=True,
                    )
                z_phys = random_physical_step(z[..., :7], out, params=params, drv=drv)
                p_next = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
                z = torch.cat([z_phys, p_next], dim=-1)
            else:
                raise ValueError("kind must be 'discretion' or 'commitment'.")
            states.append(z)
    return torch.stack(states, dim=0)
