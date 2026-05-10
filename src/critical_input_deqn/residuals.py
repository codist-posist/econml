from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from .config import (
    BaselineParams,
    NATURAL_OUTPUT_NAMES,
    QMCConfig,
    RULE_OUTPUT_NAMES,
)
from .economics import (
    derive_natural,
    derive_rule,
    external_conditions,
    fischer_burmeister,
    adaptation_enabled,
    marginal_cost,
    mc_derivative_A,
    p_x_derivative_A,
    psi,
    psi_prime,
    unit_intermediate_price,
    unpack_natural_state,
    unpack_rule_state,
)
from .qmc import QMCNodes
from .transforms import decode_natural_outputs, decode_rule_outputs
from .transitions import transition_natural_states, transition_rule_states


TensorDict = Dict[str, torch.Tensor]
NaturalPolicy = Callable[[torch.Tensor], torch.Tensor]
RulePolicy = Callable[[torch.Tensor], torch.Tensor]


def _mean_over_nodes(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)


def natural_residuals(
    z_n: torch.Tensor,
    raw_n: torch.Tensor,
    natural_net: NaturalPolicy,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    fb_epsilon: float,
) -> Tuple[TensorDict, TensorDict]:
    """Flexible-price benchmark residuals.

    This auxiliary network is solved, validated, and frozen before the
    bottleneck-adjusted Taylor rule is trained.
    """

    st = unpack_natural_state(z_n)
    out = decode_natural_outputs(raw_n, NATURAL_OUTPUT_NAMES)
    drv = derive_natural(st, out, params)
    C, Y, chi, Rn = out["C_n"], out["Y_n"], out["chi_n"], out["R_n_real"]

    z_next = transition_natural_states(st, nodes, params, qmc_cfg)
    B, S, K = z_next.shape
    raw_next = natural_net(z_next.reshape(B * S, K))
    out_next = decode_natural_outputs(raw_next, NATURAL_OUTPUT_NAMES)
    C_next = out_next["C_n"].reshape(B, S)
    lambda_ratio = C_next.pow(-float(params.sigma)) / C[:, None].pow(-float(params.sigma))

    mc_flex = torch.full_like(C, (float(params.epsilon) - 1.0) / float(params.epsilon))
    res: TensorDict = {}
    res["n_mc"] = drv["mc"] / mc_flex - 1.0
    res["n_resource"] = (Y - C - drv["pm"] * drv["M"] - float(params.p_d) * drv["S"]) / Y
    res["n_euler"] = torch.log(torch.clamp(float(params.beta) * Rn * _mean_over_nodes(lambda_ratio), min=1e-12))
    cap_slack = (drv["mbar"] - drv["M"]) / torch.clamp(drv["mbar"], min=1e-12)
    cap_rent = chi / torch.clamp(drv["pm"], min=1e-12)
    res["n_cap_fb"] = fischer_burmeister(cap_rent, cap_slack, fb_epsilon)
    return res, {**out, **drv}


def rule_residuals(
    z: torch.Tensor,
    raw: torch.Tensor,
    rule_net: RulePolicy,
    natural_net: NaturalPolicy,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    fb_epsilon: float,
    policy: str,
) -> Tuple[TensorDict, TensorDict]:
    """Residuals for fixed Taylor and bottleneck-adjusted Taylor policies."""

    st = unpack_rule_state(z)
    out = decode_rule_outputs(raw, RULE_OUTPUT_NAMES)

    z_n = z[..., :6]
    raw_n = natural_net(z_n)
    out_n = decode_natural_outputs(raw_n, NATURAL_OUTPUT_NAMES)
    Y_n = out_n["Y_n"]
    R_n = out_n["R_n_real"]
    drv = derive_rule(st, out, params, Y_n=Y_n, R_n=R_n, policy=policy)

    z_next = transition_rule_states(st, drv["A_next"], drv["Delta"], nodes, params, qmc_cfg)
    B, S, K = z_next.shape
    raw_next = rule_net(z_next.reshape(B * S, K))
    out_next = decode_rule_outputs(raw_next, RULE_OUTPUT_NAMES)
    z_next_flat = z_next.reshape(B * S, K)
    st_next = unpack_rule_state(z_next_flat)
    raw_n_next = natural_net(z_next_flat[..., :6])
    out_n_next = decode_natural_outputs(raw_n_next, NATURAL_OUTPUT_NAMES)
    drv_next = derive_rule(
        st_next,
        out_next,
        params,
        Y_n=out_n_next["Y_n"],
        R_n=out_n_next["R_n_real"],
        policy=policy,
    )

    C = out["C"]
    Lambda = drv["Lambda"]
    Lambda_next = out_next["C"].pow(-float(params.sigma)).reshape(B, S)
    Pi_next = out_next["Pi"].reshape(B, S)
    S_p_next = out_next["S_p"].reshape(B, S)
    F_p_next = out_next["F_p"].reshape(B, S)
    Q_next = out_next["Q_A"].reshape(B, S)
    Mdisc = float(params.beta) * Lambda_next / Lambda[:, None]

    # Next-period adaptation benefit -C_A.  This is evaluated at next states
    # and next controls, as in the continuation-value recursion.
    p_x_A_next = p_x_derivative_A(
        st_next.A,
        drv_next["p_m_eff"],
        drv_next["p_d"],
        params,
    )
    mc_A_next = mc_derivative_A(drv_next["mc"], drv_next["p_x"], p_x_A_next, params)
    benefit_A_next = -(mc_A_next * drv_next["Delta"] * out_next["Y"]).reshape(B, S)

    res: TensorDict = {}
    res["hh_euler"] = torch.log(
        torch.clamp(
            _mean_over_nodes(float(params.beta) * drv["R"][:, None] * Lambda_next / Lambda[:, None] / Pi_next),
            min=1e-12,
        )
    )
    res["resource"] = (
        out["Y"]
        - out["C"]
        - drv["pm"] * drv["M"]
        - float(params.p_d) * drv["S"]
        - float(params.p_a) * psi(drv["I_A_effective"], params)
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
    cap_rent = out["chi"] / torch.clamp(drv["pm"], min=1e-12)
    res["cap_fb"] = fischer_burmeister(cap_rent, cap_slack, fb_epsilon)
    if adaptation_enabled(params):
        repair_gap = drv["Omega_A"] * float(params.p_a) * psi_prime(out["I_A"], params) - out["Q_A"]
        repair_quantity = out["I_A"] / (1.0 + out["I_A"])
        repair_value = repair_gap / torch.clamp(drv["Omega_A"] * float(params.p_a), min=1e-12)
        res["repair_fb"] = fischer_burmeister(repair_quantity, repair_value, fb_epsilon)
        res["Q"] = (
            out["Q_A"]
            - _mean_over_nodes(Mdisc * (benefit_A_next + (1.0 - float(params.delta_A)) * Q_next))
        ) / (1.0 + out["Q_A"])
    else:
        res["repair_fb"] = out["I_A"]
        res["Q"] = out["Q_A"]

    return res, {**out, **drv, "Y_n": Y_n, "R_n_real": R_n}


def stack_residuals(res: TensorDict, keys: tuple[str, ...] | None = None) -> torch.Tensor:
    if keys is None:
        keys = tuple(res.keys())
    return torch.stack([res[k] for k in keys], dim=-1)
