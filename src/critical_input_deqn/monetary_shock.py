from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

from .config import (
    NATURAL_OUTPUT_NAMES,
    RULE_OUTPUT_NAMES,
    RULE_STATE_NAMES,
    BaselineParams,
    NetworkConfig,
    QMCConfig,
    TrainConfig,
)
from .economics import (
    State,
    adaptation_enabled,
    bounded_repair_investment,
    derive_rule,
    mc_derivative_A,
    omega_A_cost,
    p_x_derivative_A,
    psi,
    psi_prime,
    unpack_rule_state,
)
from .networks import MLP
from .qmc import QMCNodes, make_qmc_nodes
from .residuals import _mean_over_nodes, stack_residuals
from .sampling import sample_rule_states
from .train import (
    TrainLog,
    _RULE_SCENARIO_LABELS,
    _RULE_SCENARIO_OFFSETS,
    _RULE_SCENARIO_POINTS,
    _announce_training,
    _calm_anchor_diagnostics_from_terms,
    _calm_anchor_loss_from_terms,
    _log_metrics,
    _mark_stopped,
    _maybe_save_training_state,
    _maybe_update_best_state,
    _passes_stop_criteria,
    _pricing_sum_targets,
    _progress_range,
    _report_progress,
    _residual_matrix_diagnostics,
    _restore_best_state,
    _rule_scenario_additions,
    _scenario_q_diagnostics,
    _should_apply_scenario_loss,
    _top_residual_summary,
    _validation_nodes,
    exact_condition_diagnostics,
    freeze,
    residual_diagnostics,
    residual_loss,
    save_checkpoint,
)
from .transforms import decode_natural_outputs, decode_rule_outputs
from .transitions import transition_rule_states


RULE_SHOCK_STATE_NAMES = RULE_STATE_NAMES + ("eps_R",)


@dataclass(frozen=True)
class MonetaryShockConfig:
    """Configuration for rule-based monetary-policy shocks.

    eps_R is a log gross-rate wedge in the Taylor rule:
    R_t = R_t^{rule} exp(eps_R_t).
    """

    rho_R: float = 0.50
    train_shock_std: float = 0.005
    train_shock_span: float = 0.030
    small_bp_annualized: float = 25.0
    large_bp_annualized: float = 100.0


def make_rule_shock_net(
    net_cfg: NetworkConfig = NetworkConfig(),
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MLP:
    """Build a Taylor-rule DEQN network with the monetary-shock state."""

    return MLP(len(RULE_SHOCK_STATE_NAMES), len(RULE_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def unpack_rule_shock_state(z: torch.Tensor) -> tuple[State, torch.Tensor]:
    """Split the augmented Taylor state into physical state and policy shock."""

    return unpack_rule_state(z[..., : len(RULE_STATE_NAMES)]), z[..., len(RULE_STATE_NAMES)]


def natural_from_rule_shock_states(z: torch.Tensor) -> torch.Tensor:
    """Flexible-price benchmark uses the physical state, not the policy shock."""

    return z[..., :6]


def sample_rule_shock_states(
    n: int,
    *,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig = MonetaryShockConfig(),
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> torch.Tensor:
    """Mixture sampler over physical states and transitory policy shocks."""

    base = sample_rule_states(n, params=params, device=device, dtype=dtype, seed=seed)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 97_531)
    u = torch.rand((int(n),), generator=gen, device=device, dtype=dtype)
    normal = u < 0.60
    hawkish = (u >= 0.60) & (u < 0.82)
    dovish = u >= 0.82
    eps = float(shock_cfg.train_shock_std) * torch.randn((int(n),), generator=gen, device=device, dtype=dtype)
    eps = torch.where(hawkish, float(shock_cfg.train_shock_span) * torch.rand_like(u), eps)
    eps = torch.where(dovish, -float(shock_cfg.train_shock_span) * torch.rand_like(u), eps)
    eps = torch.where(normal, 0.25 * eps, eps)
    return torch.cat([base, eps[:, None]], dim=-1)


def derive_rule_with_monetary_shock(
    st: State,
    out: Dict[str, torch.Tensor],
    params: BaselineParams,
    *,
    Y_n: torch.Tensor,
    R_n: torch.Tensor | None,
    policy: str,
    eps_R: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Derived equilibrium objects when Taylor rules include eps_R."""

    drv = derive_rule(st, out, params, Y_n=Y_n, R_n=R_n, policy=policy)
    R_rule = drv["R"]
    R = R_rule * torch.exp(eps_R)
    drv = dict(drv)
    drv["R_rule"] = R_rule
    drv["eps_R"] = eps_R
    drv["R"] = R
    drv["Omega_A"] = omega_A_cost(R, params)
    I = bounded_repair_investment(out["Q_A"], drv["Omega_A"], drv["p_a"], params)
    drv["I_A"] = I
    drv["I_A_effective"] = I
    drv["A_next"] = (1.0 - float(params.delta_A)) * st.A + I
    return drv


def transition_rule_shock_states(
    st: State,
    eps_R: torch.Tensor,
    A_next: torch.Tensor,
    Delta_next: torch.Tensor,
    nodes: QMCNodes,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    shock_cfg: MonetaryShockConfig,
) -> torch.Tensor:
    """Next augmented Taylor state for every current state and QMC node."""

    z_phys_next = transition_rule_states(st, A_next, Delta_next, nodes, params, qmc_cfg)
    B, S, _ = z_phys_next.shape
    eps_next = float(shock_cfg.rho_R) * eps_R[:, None].expand(B, S)
    return torch.cat([z_phys_next, eps_next[..., None]], dim=-1)


def rule_shock_residuals(
    z: torch.Tensor,
    raw: torch.Tensor,
    rule_net,
    natural_net,
    nodes: QMCNodes,
    *,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    shock_cfg: MonetaryShockConfig,
    fb_epsilon: float,
    policy: str,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Residuals for Taylor rules with a transitory monetary-policy shock."""

    st, eps_R = unpack_rule_shock_state(z)
    z_n = natural_from_rule_shock_states(z)
    out_n = decode_natural_outputs(natural_net(z_n), NATURAL_OUTPUT_NAMES, params=params)
    Y_n = out_n["Y_n"]
    R_n = out_n["R_n_real"]
    out = decode_rule_outputs(raw, RULE_OUTPUT_NAMES, params=params, y_ref=Y_n)
    drv = derive_rule_with_monetary_shock(
        st,
        out,
        params,
        Y_n=Y_n,
        R_n=R_n,
        policy=policy,
        eps_R=eps_R,
    )

    z_next = transition_rule_shock_states(
        st,
        eps_R,
        drv["A_next"],
        drv["Delta"],
        nodes,
        params,
        qmc_cfg,
        shock_cfg,
    )
    B, S, K = z_next.shape
    z_next_flat = z_next.reshape(B * S, K)
    st_next, eps_next = unpack_rule_shock_state(z_next_flat)
    out_n_next = decode_natural_outputs(
        natural_net(natural_from_rule_shock_states(z_next_flat)),
        NATURAL_OUTPUT_NAMES,
        params=params,
    )
    out_next = decode_rule_outputs(rule_net(z_next_flat), RULE_OUTPUT_NAMES, params=params, y_ref=out_n_next["Y_n"])
    drv_next = derive_rule_with_monetary_shock(
        st_next,
        out_next,
        params,
        Y_n=out_n_next["Y_n"],
        R_n=out_n_next["R_n_real"],
        policy=policy,
        eps_R=eps_next,
    )

    C = out["C"]
    Lambda = drv["Lambda"]
    Lambda_next = out_next["C"].pow(-float(params.sigma)).reshape(B, S)
    Pi_next = out_next["Pi"].reshape(B, S)
    S_p_next = out_next["S_p"].reshape(B, S)
    F_p_next = out_next["F_p"].reshape(B, S)
    Q_next = out_next["Q_A"].reshape(B, S)
    Mdisc = float(params.beta) * Lambda_next / Lambda[:, None]

    p_x_A_next = p_x_derivative_A(
        st_next.A,
        drv_next["p_m_eff"],
        drv_next["p_d"],
        params,
    )
    mc_A_next = mc_derivative_A(drv_next["mc"], drv_next["p_x"], p_x_A_next, params)
    benefit_A_next = -(mc_A_next * drv_next["Delta"] * out_next["Y"]).reshape(B, S)

    res: Dict[str, torch.Tensor] = {}
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
    price_index_lhs = (
        (1.0 - float(params.theta)) * drv["p_star"].pow(1.0 - float(params.epsilon))
        + float(params.theta) * out["Pi"].pow(float(params.epsilon) - 1.0)
    )
    res["price_index"] = torch.log(torch.clamp(price_index_lhs, min=1e-12))
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
    if adaptation_enabled(params):
        res["Q"] = (
            out["Q_A"]
            - _mean_over_nodes(Mdisc * (benefit_A_next + (1.0 - float(params.delta_A)) * Q_next))
        ) / (1.0 + out["Q_A"].abs())
    else:
        res["Q"] = out["Q_A"]

    return res, {**out, **drv, "Y_n": Y_n, "R_n_real": R_n}


def random_rule_shock_step(
    z: torch.Tensor,
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
) -> torch.Tensor:
    """One simulated augmented Taylor-state transition."""

    st, eps_R = unpack_rule_shock_state(z)
    out_n = decode_natural_outputs(natural_net(natural_from_rule_shock_states(z)), NATURAL_OUTPUT_NAMES, params=params)
    out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES, params=params, y_ref=out_n["Y_n"])
    drv = derive_rule_with_monetary_shock(
        st,
        out,
        params,
        Y_n=out_n["Y_n"],
        R_n=out_n["R_n_real"],
        policy=policy,
        eps_R=eps_R,
    )

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
    eps_next = float(shock_cfg.rho_R) * eps_R

    return torch.stack(
        [
            D_next,
            X_next,
            ell_D_next,
            ell_X_next,
            log_Z_next,
            drv["A_next"],
            log_Delta_next,
            eps_next,
        ],
        dim=-1,
    )


def simulate_rule_shock_episode(
    initial_z: torch.Tensor,
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    length: int,
) -> torch.Tensor:
    """Simulate augmented Taylor episodes for DEQN state sampling."""

    states = [initial_z]
    z = initial_z
    with torch.no_grad():
        for _ in range(1, int(length)):
            z = random_rule_shock_step(z, rule_net, natural_net, policy=policy, params=params, shock_cfg=shock_cfg)
            states.append(z)
    return torch.stack(states, dim=0)


def train_rule_shock_episode(
    natural_net,
    *,
    policy: str,
    params: BaselineParams = BaselineParams(),
    shock_cfg: MonetaryShockConfig = MonetaryShockConfig(),
    net_cfg: NetworkConfig = NetworkConfig(),
    qmc_cfg: QMCConfig = QMCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    episodes: int | None = None,
    log_every: int = 10,
) -> tuple[MLP, TrainLog]:
    """Train a Taylor-rule network with an explicit policy-shock state."""

    if policy.lower() not in {"fixed", "ba"}:
        raise ValueError("policy must be 'fixed' or 'ba'.")
    freeze(natural_net)
    device = train_cfg.device
    dtype = train_cfg.dtype
    net = make_rule_shock_net(net_cfg, device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=float(train_cfg.lr))
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=device, dtype=dtype)
    val_nodes = _validation_nodes(qmc_cfg, device=device, dtype=dtype, seed_offset=40_201)
    val_qmc_cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + 40_201)
    val_z = sample_rule_shock_states(
        train_cfg.stop_val_states,
        params=params,
        shock_cfg=shock_cfg,
        device=device,
        dtype=dtype,
        seed=9_001,
    )
    n_episodes = int(train_cfg.steps if episodes is None else episodes)
    log = TrainLog()
    stop_hits = 0
    best_state = None

    current_state = sample_rule_shock_states(
        train_cfg.sim_batch_size,
        params=params,
        shock_cfg=shock_cfg,
        device=device,
        dtype=dtype,
    )
    _announce_training(
        kind=f"rule-{policy.lower()}-monetary-shock",
        total=n_episodes,
        train_cfg=train_cfg,
        qmc_cfg=qmc_cfg,
        log_every=log_every,
    )
    progress = _progress_range(n_episodes, desc=f"rule-{policy.lower()}-mp", enabled=train_cfg.show_progress)
    for episode in progress:
        state_episode = simulate_rule_shock_episode(
            current_state,
            net,
            natural_net,
            policy=policy,
            params=params,
            shock_cfg=shock_cfg,
            length=train_cfg.episode_length,
        )
        current_state = state_episode[-1].detach()
        flat_states = state_episode.reshape(-1, state_episode.shape[-1]).detach()
        last_mat = None
        last_loss = None
        batch_size = int(train_cfg.batch_size)
        broad_n = int(round(batch_size * float(train_cfg.episode_broad_share)))
        broad_n = min(max(broad_n, 0), batch_size)
        episode_n = batch_size - broad_n
        updates = max(1, int(train_cfg.episode_updates_per_episode))
        for _ in range(updates):
            pieces = []
            if episode_n > 0:
                idx = torch.randint(flat_states.shape[0], (episode_n,), device=flat_states.device)
                pieces.append(flat_states[idx])
            if broad_n > 0:
                pieces.append(
                    sample_rule_shock_states(
                        broad_n,
                        params=params,
                        shock_cfg=shock_cfg,
                        device=device,
                        dtype=dtype,
                    )
                )
            z = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
            raw = net(z)
            res, _ = rule_shock_residuals(
                z,
                raw,
                net,
                natural_net,
                nodes,
                params=params,
                qmc_cfg=qmc_cfg,
                shock_cfg=shock_cfg,
                fb_epsilon=train_cfg.fb_epsilon_start,
                policy=policy,
            )
            mat = stack_residuals(res)
            loss = residual_loss(mat, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)
            loss = loss + _rule_shock_auxiliary_training_loss(
                net,
                natural_net,
                nodes,
                policy=policy,
                params=params,
                shock_cfg=shock_cfg,
                qmc_cfg=qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=train_cfg.fb_epsilon_start,
                step=episode,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            last_mat = mat.detach()
            last_loss = loss.detach()
        if last_mat is not None and last_loss is not None and (
            episode == 1 or episode % int(log_every) == 0 or episode == n_episodes
        ):
            with torch.no_grad():
                val_res, _ = rule_shock_residuals(
                    val_z,
                    net(val_z),
                    net,
                    natural_net,
                    val_nodes,
                    params=params,
                    qmc_cfg=val_qmc_cfg,
                    shock_cfg=shock_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                    policy=policy,
                )
                val_mat = stack_residuals(val_res).detach()
                scenario_val_res, _, scenario_names = _rule_shock_scenario_residuals(
                    net,
                    natural_net,
                    val_nodes,
                    policy=policy,
                    params=params,
                    shock_cfg=shock_cfg,
                    qmc_cfg=val_qmc_cfg,
                    train_cfg=train_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                )
            metrics = _log_metrics(episode, last_mat, log, last_loss, val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            scenario_diag = _scenario_q_diagnostics(scenario_val_res, scenario_names, q_key="Q")
            metrics.update(scenario_diag)
            calm_diag = _rule_shock_calm_anchor_diagnostics(
                net,
                natural_net,
                policy=policy,
                params=params,
                shock_cfg=shock_cfg,
                train_cfg=train_cfg,
            )
            metrics.update(calm_diag)
            calm_resid_diag = _rule_shock_calm_residual_diagnostics(
                net,
                natural_net,
                val_nodes,
                policy=policy,
                params=params,
                shock_cfg=shock_cfg,
                qmc_cfg=val_qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=train_cfg.fb_epsilon_final,
            )
            metrics.update(calm_resid_diag)
            best_state = _maybe_update_best_state(
                net,
                log,
                metrics,
                episode,
                best_state,
                train_cfg,
                extra={"kind": "rule_monetary_shock", "policy": policy.lower(), "current_state": current_state},
            )
            log.extra_metrics.append({"step": float(episode), **scenario_diag, **calm_diag, **calm_resid_diag, "selection_score": float(metrics.get("selection_score", float("nan")))})
            _maybe_save_training_state(
                step=episode,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "rule_monetary_shock", "policy": policy.lower(), "current_state": current_state},
            )
            if _passes_stop_criteria(val_mat, train_cfg, episode, metrics):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, episode, train_cfg)
                    _report_progress(
                        progress,
                        metrics,
                        step=episode,
                        total=n_episodes,
                        stop_hits=stop_hits,
                        enabled=train_cfg.show_progress,
                    )
                    break
            else:
                stop_hits = 0
            _report_progress(
                progress,
                metrics,
                step=episode,
                total=n_episodes,
                stop_hits=stop_hits,
                enabled=train_cfg.show_progress,
            )
    _restore_best_state(net, best_state)
    return net, log


def evaluate_rule_shock(
    net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams = BaselineParams(),
    shock_cfg: MonetaryShockConfig = MonetaryShockConfig(),
    qmc_cfg: QMCConfig = QMCConfig(n_train=4096),
    train_cfg: TrainConfig = TrainConfig(),
    n_states: int = 4096,
) -> Dict[str, float]:
    """Residual diagnostics for a trained Taylor monetary-shock network."""

    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=train_cfg.device, dtype=train_cfg.dtype)
    z = sample_rule_shock_states(
        n_states,
        params=params,
        shock_cfg=shock_cfg,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
        seed=4_321,
    )
    with torch.no_grad():
        res, drv = rule_shock_residuals(
            z,
            net(z),
            net,
            natural_net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            shock_cfg=shock_cfg,
            fb_epsilon=train_cfg.fb_epsilon_final,
            policy=policy,
        )
        mat = stack_residuals(res)
        scenario_res, _, scenario_names = _rule_shock_scenario_residuals(
            net,
            natural_net,
            nodes,
            policy=policy,
            params=params,
            shock_cfg=shock_cfg,
            qmc_cfg=qmc_cfg,
            train_cfg=train_cfg,
            fb_epsilon=train_cfg.fb_epsilon_final,
        )
        return {
            "loss": float(mat.pow(2).mean().cpu()),
            "rms": float(torch.sqrt(mat.pow(2).mean()).cpu()),
            "max_abs": float(mat.abs().max().cpu()),
            **residual_diagnostics(res),
            **exact_condition_diagnostics(drv, params),
            **_scenario_q_diagnostics(scenario_res, scenario_names, q_key="Q"),
        }


def _normal_initial_rule_shock_state(
    batch_size: int,
    *,
    params: BaselineParams,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    D = torch.zeros(batch_size, device=device, dtype=dtype)
    X = torch.zeros_like(D)
    ell_D = torch.full_like(D, float(params.log_bar_lambda_D))
    ell_X = torch.full_like(D, float(params.log_bar_lambda_X))
    log_Z = torch.full_like(D, float(-0.5 * params.sigma_z**2))
    A = torch.zeros_like(D)
    log_Delta = torch.zeros_like(D)
    eps_R = torch.zeros_like(D)
    return torch.stack([D, X, ell_D, ell_X, log_Z, A, log_Delta, eps_R], dim=-1)


def _deterministic_rule_shock_step(
    z: torch.Tensor,
    *,
    A_next: torch.Tensor,
    Delta_next: torch.Tensor,
    add_eps_R: torch.Tensor,
    add_D: torch.Tensor | None = None,
    add_X: torch.Tensor | None = None,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
) -> torch.Tensor:
    st, eps_R = unpack_rule_shock_state(z)
    D_next = (1.0 - float(params.delta_D)) * st.D + (0.0 if add_D is None else add_D)
    X_next = (1.0 - float(params.delta_X)) * st.X + (0.0 if add_X is None else add_X)
    ell_D_next = (
        (1.0 - float(params.rho_lambda_D)) * float(params.log_bar_lambda_D)
        + float(params.rho_lambda_D) * st.ell_D
        + float(params.kappa_D_lambda) * st.D
    )
    ell_X_next = (
        (1.0 - float(params.rho_lambda_X)) * float(params.log_bar_lambda_X)
        + float(params.rho_lambda_X) * st.ell_X
        + float(params.beta_X) * st.D
    )
    log_Z_next = float(params.rho_z) * st.log_Z
    log_Delta_next = torch.log(torch.clamp(Delta_next, min=1e-12))
    eps_next = float(shock_cfg.rho_R) * eps_R + add_eps_R
    return torch.stack([D_next, X_next, ell_D_next, ell_X_next, log_Z_next, A_next, log_Delta_next, eps_next], dim=-1)


def _rule_shock_training_scenario_states(
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, list[str]]:
    device = train_cfg.device
    dtype = train_cfg.dtype
    burnin = max(1, int(train_cfg.rule_scenario_burnin))
    horizon = max(5, int(train_cfg.rule_scenario_horizon))
    total = burnin + horizon + 1
    z = _normal_initial_rule_shock_state(len(_RULE_SCENARIO_LABELS), params=params, device=device, dtype=dtype)
    states = [z]
    with torch.no_grad():
        for t in range(1, total):
            st, eps_R = unpack_rule_shock_state(z)
            out_n = decode_natural_outputs(
                natural_net(natural_from_rule_shock_states(z)),
                NATURAL_OUTPUT_NAMES,
                params=params,
            )
            out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES, params=params, y_ref=out_n["Y_n"])
            drv = derive_rule_with_monetary_shock(
                st,
                out,
                params,
                Y_n=out_n["Y_n"],
                R_n=out_n["R_n_real"],
                policy=policy,
                eps_R=eps_R,
            )
            add_D, add_X = _rule_scenario_additions(
                t=t,
                pulse=burnin,
                relief_lag=4,
                params=params,
                device=z.device,
                dtype=z.dtype,
            )
            z = _deterministic_rule_shock_step(
                z,
                A_next=drv["A_next"],
                Delta_next=drv["Delta"],
                add_eps_R=torch.zeros_like(add_D),
                add_D=add_D,
                add_X=add_X,
                params=params,
                shock_cfg=shock_cfg,
            )
            states.append(z)
    stacked = torch.stack(states, dim=0)
    label_to_idx = {label: j for j, label in enumerate(_RULE_SCENARIO_LABELS)}
    selected = []
    names = []
    for label, tag in _RULE_SCENARIO_POINTS:
        idx = min(max(burnin + int(_RULE_SCENARIO_OFFSETS[tag]), 0), stacked.shape[0] - 1)
        selected.append(stacked[idx, label_to_idx[label]])
        names.append(f"{label}.{tag}")
    return torch.stack(selected, dim=0).detach(), names


def _rule_shock_scenario_residuals(
    rule_net,
    natural_net,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list[str]]:
    z, names = _rule_shock_training_scenario_states(
        rule_net,
        natural_net,
        policy=policy,
        params=params,
        shock_cfg=shock_cfg,
        train_cfg=train_cfg,
    )
    res, drv = rule_shock_residuals(
        z,
        rule_net(z),
        rule_net,
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        shock_cfg=shock_cfg,
        fb_epsilon=fb_epsilon,
        policy=policy,
    )
    return res, drv, names


def _rule_shock_calm_anchor_loss(
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    return _calm_anchor_loss_from_terms(
        _rule_shock_calm_anchor_terms(
            rule_net,
            natural_net,
            policy=policy,
            params=params,
            shock_cfg=shock_cfg,
            train_cfg=train_cfg,
        )
    )


def _rule_shock_calm_anchor_terms(
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    train_cfg: TrainConfig,
) -> list[torch.Tensor]:
    z = _normal_initial_rule_shock_state(1, params=params, device=train_cfg.device, dtype=train_cfg.dtype)
    st, eps_R = unpack_rule_shock_state(z)
    out_n = decode_natural_outputs(natural_net(natural_from_rule_shock_states(z)), NATURAL_OUTPUT_NAMES, params=params)
    out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES, params=params, y_ref=out_n["Y_n"])
    drv = derive_rule_with_monetary_shock(
        st,
        out,
        params,
        Y_n=out_n["Y_n"],
        R_n=out_n["R_n_real"],
        policy=policy,
        eps_R=eps_R,
    )
    pressure = drv["M_zero_rent"] / torch.clamp(drv["mbar"], min=1e-12)
    target_pressure = torch.full_like(pressure, 1.0 / (1.0 + float(params.normal_capacity_slack)))
    repair_scale = max(float(params.repair_capacity), 1e-6)
    s_target, f_target = _pricing_sum_targets(out_n["Y_n"], params)
    return [
        torch.log(torch.clamp(out["C"] / torch.clamp(out_n["C_n"], min=1e-12), min=1e-12)),
        torch.log(torch.clamp(out["Y"] / torch.clamp(out_n["Y_n"], min=1e-12), min=1e-12)),
        torch.log(torch.clamp(out["Pi"] / float(params.bar_pi), min=1e-12)),
        torch.log(torch.clamp(out["S_p"] / torch.clamp(s_target, min=1e-12), min=1e-12)),
        torch.log(torch.clamp(out["F_p"] / torch.clamp(f_target, min=1e-12), min=1e-12)),
        pressure - target_pressure,
        drv["chi"] / torch.clamp(drv["pm"], min=1e-12),
        drv["I_A"] / repair_scale,
    ]


def _rule_shock_calm_anchor_diagnostics(
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    return _calm_anchor_diagnostics_from_terms(
        _rule_shock_calm_anchor_terms(
            rule_net,
            natural_net,
            policy=policy,
            params=params,
            shock_cfg=shock_cfg,
            train_cfg=train_cfg,
        )
    )


def _rule_shock_calm_residuals(
    rule_net,
    natural_net,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    z = _normal_initial_rule_shock_state(1, params=params, device=train_cfg.device, dtype=train_cfg.dtype)
    return rule_shock_residuals(
        z,
        rule_net(z),
        rule_net,
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        shock_cfg=shock_cfg,
        fb_epsilon=fb_epsilon,
        policy=policy,
    )


def _rule_shock_calm_residual_loss(
    rule_net,
    natural_net,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> torch.Tensor:
    res, _ = _rule_shock_calm_residuals(
        rule_net,
        natural_net,
        nodes,
        policy=policy,
        params=params,
        shock_cfg=shock_cfg,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=fb_epsilon,
    )
    return residual_loss(stack_residuals(res), loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)


def _rule_shock_calm_residual_diagnostics(
    rule_net,
    natural_net,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> Dict[str, float]:
    res, _ = _rule_shock_calm_residuals(
        rule_net,
        natural_net,
        nodes,
        policy=policy,
        params=params,
        shock_cfg=shock_cfg,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=fb_epsilon,
    )
    return _residual_matrix_diagnostics("calm_residual", stack_residuals(res))


def _rule_shock_auxiliary_training_loss(
    rule_net,
    natural_net,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
    step: int | None = None,
) -> torch.Tensor:
    pieces = []
    q_weight = float(train_cfg.rule_scenario_q_weight)
    if q_weight > 0.0 and _should_apply_scenario_loss(step, train_cfg):
        scenario_res, _, _ = _rule_shock_scenario_residuals(
            rule_net,
            natural_net,
            nodes,
            policy=policy,
            params=params,
            shock_cfg=shock_cfg,
            qmc_cfg=qmc_cfg,
            train_cfg=train_cfg,
            fb_epsilon=fb_epsilon,
        )
        pieces.append(q_weight * residual_loss(scenario_res["Q"].reshape(-1, 1), loss=train_cfg.loss, huber_delta=train_cfg.huber_delta))
    calm_weight = float(train_cfg.rule_calm_anchor_weight)
    if calm_weight > 0.0:
        pieces.append(
            calm_weight
            * _rule_shock_calm_anchor_loss(
                rule_net,
                natural_net,
                policy=policy,
                params=params,
                shock_cfg=shock_cfg,
                train_cfg=train_cfg,
            )
        )
    calm_resid_weight = float(train_cfg.rule_calm_residual_weight)
    if calm_resid_weight > 0.0:
        pieces.append(
            calm_resid_weight
            * _rule_shock_calm_residual_loss(
                rule_net,
                natural_net,
                nodes,
                policy=policy,
                params=params,
                shock_cfg=shock_cfg,
                qmc_cfg=qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=fb_epsilon,
            )
        )
    if not pieces:
        return torch.zeros((), device=train_cfg.device, dtype=train_cfg.dtype)
    return torch.stack(pieces).sum()


def annualized_bp_to_log_quarterly(bp: float) -> float:
    """Convert annualized basis points into a quarterly gross-rate log wedge."""

    return float(np.log1p(float(bp) / 40_000.0))


def simulate_rule_monetary_ir_scenarios(
    *,
    policy: str,
    rule_net,
    natural_net,
    params: BaselineParams,
    shock_cfg: MonetaryShockConfig,
    burnin: int,
    horizon: int,
    presteps: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor]:
    """Deterministic one-time monetary-policy shock IRFs for Taylor rules."""

    pulse = int(burnin)
    small = annualized_bp_to_log_quarterly(float(shock_cfg.small_bp_annualized))
    large = annualized_bp_to_log_quarterly(float(shock_cfg.large_bp_annualized))
    scenarios: dict[str, dict[int, float]] = {
        "no_shock": {},
        f"mp_{int(shock_cfg.small_bp_annualized)}bp": {pulse: small},
        f"mp_{int(shock_cfg.large_bp_annualized)}bp": {pulse: large},
    }
    labels = list(scenarios.keys())
    z = _normal_initial_rule_shock_state(len(labels), params=params, device=device, dtype=dtype)
    states = [z]
    total = int(burnin) + int(horizon)
    with torch.no_grad():
        for t in range(1, total):
            st, eps_R = unpack_rule_shock_state(z)
            out_n = decode_natural_outputs(
                natural_net(natural_from_rule_shock_states(z)),
                NATURAL_OUTPUT_NAMES,
                params=params,
            )
            out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES, params=params, y_ref=out_n["Y_n"])
            drv = derive_rule_with_monetary_shock(
                st,
                out,
                params,
                Y_n=out_n["Y_n"],
                R_n=out_n["R_n_real"],
                policy=policy,
                eps_R=eps_R,
            )
            add_eps = torch.tensor(
                [spec.get(int(t), 0.0) for spec in scenarios.values()],
                device=z.device,
                dtype=z.dtype,
            )
            z = _deterministic_rule_shock_step(
                z,
                A_next=drv["A_next"],
                Delta_next=drv["Delta"],
                add_eps_R=add_eps,
                params=params,
                shock_cfg=shock_cfg,
            )
            states.append(z)
    start = max(0, int(burnin) - int(presteps))
    return labels, torch.stack(states, dim=0)[start:]


def _numpy_dict(data: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in data.items()}


def _state_dict(z: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {name: z[..., i] for i, name in enumerate(RULE_SHOCK_STATE_NAMES)}


def _add_common_ratios(data: Dict[str, torch.Tensor], params: BaselineParams) -> Dict[str, torch.Tensor]:
    if "Y" in data and "Y_n" in data:
        y_ratio = data["Y"] / torch.clamp(data["Y_n"], min=1e-12)
        data["output_gap"] = torch.log(torch.clamp(y_ratio, min=1e-12))
        data["output_gap_level"] = y_ratio - 1.0
    if "M" in data and "mbar" in data:
        data["cap_gap"] = data["mbar"] - data["M"]
        data["cap_slack"] = data["cap_gap"] / torch.clamp(data["mbar"], min=1e-12)
        pressure_source = data.get("M_zero_rent", data["M"])
        data["cap_pressure_ratio"] = pressure_source / torch.clamp(data["mbar"], min=1e-12)
        if "M_zero_rent" in data:
            data["cap_gap_zero_rent"] = data["mbar"] - data["M_zero_rent"]
        if "M_zero_rent" in data and "M_at_rent" in data:
            cap_target = torch.where(data["M_zero_rent"] > data["mbar"], data["mbar"], data["M_zero_rent"])
            data["cap_solve_error"] = data["M_at_rent"] - cap_target
            data["cap_solve_error_rel"] = data["cap_solve_error"] / torch.clamp(data["mbar"], min=1e-12)
        if "chi" in data:
            data["cap_product"] = data["chi"] * data["cap_gap"]
            if "pm" in data:
                data["cap_product_scaled"] = (
                    data["chi"] / torch.clamp(data["pm"], min=1e-12)
                ) * data["cap_slack"]
            data["cap_binding_indicator"] = (
                (data["cap_slack"].abs() <= 1e-3) & (data["chi"] > 1e-5)
            ).to(data["M"].dtype)
    if "I_A" in data and "A_next" in data and "A" in data:
        data["A_growth"] = data["A_next"] - data["A"]
    if adaptation_enabled(params) and {"I_A", "Q_A", "Omega_A", "p_a"}.issubset(data):
        data["repair_gap"] = data["Omega_A"] * data["p_a"] * psi_prime(data["I_A"], params) - data["Q_A"]
        data["repair_gap_scaled"] = data["repair_gap"] / torch.clamp(data["Omega_A"] * data["p_a"], min=1e-12)
        eta = 1.0 / torch.clamp(data["Omega_A"] * data["p_a"] * float(params.phi_A), min=1e-12)
        projected = torch.clamp(data["I_A"] - eta * data["repair_gap"], min=0.0, max=float(params.repair_capacity))
        data["repair_projection_residual"] = data["I_A"] - projected
        data["repair_activation_ratio"] = data["Q_A"] / torch.clamp(
            data["Omega_A"] * data["p_a"] * float(params.psi_A), min=1e-12
        )
        repair_tol = 1e-5
        repair_gap_tol = 1e-3
        repair_capacity = float(params.repair_capacity)
        lower_region = data["I_A"] <= repair_tol
        interior_region = (data["I_A"] > repair_tol) & (data["I_A"] < repair_capacity - repair_tol)
        upper_region = data["I_A"] >= repair_capacity - repair_tol
        data["repair_lower_corner_indicator"] = (
            lower_region & (data["repair_gap_scaled"] >= -repair_gap_tol)
        ).to(data["I_A"].dtype)
        data["repair_interior_indicator"] = (
            interior_region & (data["repair_gap_scaled"].abs() <= repair_gap_tol)
        ).to(data["I_A"].dtype)
        data["repair_capacity_bound_indicator"] = (
            upper_region & (data["repair_gap_scaled"] <= repair_gap_tol)
        ).to(data["I_A"].dtype)
        data["repair_positive_indicator"] = (data["I_A"] > repair_tol).to(data["I_A"].dtype)
        data["repair_active_or_capacity_indicator"] = (
            (data["repair_interior_indicator"] > 0.5) | (data["repair_capacity_bound_indicator"] > 0.5)
        ).to(data["I_A"].dtype)
        data["repair_active_indicator"] = (
            (data["I_A"] > 1e-5)
            & (data["I_A"] < repair_capacity - 1e-5)
            & (data["repair_gap_scaled"].abs() <= repair_gap_tol)
        ).to(data["I_A"].dtype)
        data["repair_capacity_indicator"] = (
            (data["I_A"] >= repair_capacity - 1e-5) & (data["repair_gap_scaled"] <= repair_gap_tol)
        ).to(data["I_A"].dtype)
    if "R" in data and "Pi" in data:
        data["real_rate_ex_post_proxy"] = data["R"] / torch.clamp(data["Pi"], min=1e-12)
    if "R" in data and "R_rule" in data:
        data["R_shock_multiplier"] = data["R"] / torch.clamp(data["R_rule"], min=1e-12)
    return data


def evaluate_rule_shock_path(
    states: torch.Tensor,
    *,
    policy: str,
    rule_net,
    natural_net,
    params: BaselineParams,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate definitions along an augmented Taylor-state path."""

    T, B, K = states.shape
    z = states.reshape(T * B, K)
    st, eps_R = unpack_rule_shock_state(z)
    out_n = decode_natural_outputs(natural_net(natural_from_rule_shock_states(z)), NATURAL_OUTPUT_NAMES, params=params)
    out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES, params=params, y_ref=out_n["Y_n"])
    drv = derive_rule_with_monetary_shock(
        st,
        out,
        params,
        Y_n=out_n["Y_n"],
        R_n=out_n["R_n_real"],
        policy=policy,
        eps_R=eps_R,
    )
    data: Dict[str, torch.Tensor] = {}
    data.update(_state_dict(z))
    for k, v in out.items():
        data[k if k not in data else f"out_{k}"] = v
    data.update({k: v for k, v in out_n.items() if k not in data})
    data.update({k: v for k, v in drv.items() if k not in data})
    if "I_A_effective" in data:
        data["I_A"] = data["I_A_effective"]
    data["Y_n"] = out_n["Y_n"]
    data["R_n_real"] = out_n["R_n_real"]
    data = _add_common_ratios(data, params)
    shaped = {k: v.reshape(T, B) for k, v in data.items()}
    return _numpy_dict(_state_dict(states)), _numpy_dict(shaped)


def save_ir_artifacts(
    *,
    policy: str,
    labels: list[str],
    states_np: dict[str, np.ndarray],
    outputs_np: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Save one-time monetary-shock IRFs in the same npz style as baseline IRFs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    flat_states: dict[str, np.ndarray] = {"labels": np.asarray(labels)}
    flat_defs: dict[str, np.ndarray] = {"labels": np.asarray(labels)}
    for i, label in enumerate(labels):
        for name, arr in states_np.items():
            flat_states[f"{label}__{name}"] = arr[:, i]
        for name, arr in outputs_np.items():
            flat_defs[f"{label}__{name}"] = arr[:, i]
    np.savez_compressed(out_dir / f"IR_{policy}_monetary_shock_states.npz", **flat_states)
    np.savez_compressed(out_dir / f"IR_{policy}_monetary_shock_definitions.npz", **flat_defs)


def checkpoint_metadata(
    *,
    policy: str,
    run_config: Mapping[str, object],
    shock_cfg: MonetaryShockConfig,
    selection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "kind": "rule_monetary_shock",
        "policy": policy,
        "state_names": RULE_SHOCK_STATE_NAMES,
        "shock_config": asdict(shock_cfg),
        "config": dict(run_config),
        "selection": dict(selection or {}),
    }
