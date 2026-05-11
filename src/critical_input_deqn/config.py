from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaselineParams:
    """Quarterly baseline calibration for the critical-input DEQN model."""

    # Household
    beta: float = 0.9975
    sigma: float = 2.0
    varphi: float = 1.0

    # Production and pricing
    alpha: float = 0.50
    epsilon: float = 7.0
    theta: float = 0.75
    rho: float = 0.20
    target_import_cost_share: float = 0.30
    target_min_import_cost_share: float = 0.10
    steady_state_output: float = 0.6223362929574802
    omega0: float = 0.30
    kappa_a: float = 1.0

    # Competitive supply-price normalizations
    p_d: float = 1.0
    p_a: float = 1.0
    bar_p_m: float = 1.0

    # Imported-input availability and external mapping
    normal_capacity_slack: float = 0.10
    bar_m: float = 0.08801613286112933
    nu_pD: float = 0.15
    nu_pX: float = 0.05
    nu_qD: float = 1.00
    nu_qX: float = 0.60

    # External-state dynamics
    delta_D: float = 1.0 - 2.0 ** (-1.0 / 24.0)
    delta_X: float = 1.0 - 2.0 ** (-1.0 / 12.0)
    log_bar_lambda_D: float = -3.871201010907891  # log(1/48)
    log_bar_lambda_X: float = -3.1780538303479458  # log(1/24)
    rho_lambda_D: float = 0.50
    rho_lambda_X: float = 0.50
    kappa_D_lambda: float = 0.05
    beta_X: float = 0.05
    sigma_lambda_D: float = 0.05
    sigma_lambda_X: float = 0.05
    mark_D: float = 0.25
    mark_X: float = 0.15

    # Productivity
    rho_z: float = 0.99
    sigma_z: float = 0.009

    # Adaptation
    adaptation_enabled: float = 1.0
    delta_A: float = 0.035
    repair_cost_share_10pct: float = 0.05
    repair_convex_share_10pct: float = 0.25
    repair_horizon_quarters: float = 4.0
    repair_capacity: float = 0.03
    psi_A: float = 0.29533658271880925
    phi_A: float = 5.606209895136772
    vartheta_A: float = 0.50

    # Policy rule
    bar_pi: float = 1.0
    phi_pi: float = 2.0
    phi_y: float = 0.0

    @property
    def bar_R(self) -> float:
        return self.bar_pi / self.beta


@dataclass(frozen=True)
class NetworkConfig:
    hidden_width: int = 192
    hidden_depth: int = 2
    activation: str = "selu"
    init_scale: float = 0.01


@dataclass(frozen=True)
class QMCConfig:
    n_train: int = 512
    n_val: int = 4096
    scramble: bool = True
    seed: int = 123
    poisson_max_count: int = 8


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 2048
    sim_batch_size: int = 1024
    episode_length: int = 30
    episode_updates_per_episode: int = 2
    episode_broad_share: float = 0.50
    lr: float = 1e-4
    steps: int = 50_000
    loss: str = "huber"
    huber_delta: float = 1.0
    target_rms: float | None = None
    target_max_abs: float | None = None
    early_stop_patience: int = 5
    min_steps_before_stop: int = 0
    stop_val_states: int = 2048
    show_progress: bool = True
    promise_init_scale: float = 1.0
    checkpoint_dir: str | None = None
    checkpoint_name: str = "train"
    checkpoint_every: int = 5_000
    checkpoint_keep: int = 3
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    fb_epsilon_start: float = 1e-4
    fb_epsilon_final: float = 1e-8
    rule_scenario_q_weight: float = 25.0
    rule_calm_anchor_weight: float = 5.0
    rule_scenario_burnin: int = 5
    rule_scenario_horizon: int = 10


STOP_PROFILES = {
    "natural": {"target_rms": 1e-4, "target_max_abs": 1e-2, "min_steps_before_stop": 10_000, "early_stop_patience": 10},
    "fixed": {"target_rms": 1e-4, "target_max_abs": 1e-2, "min_steps_before_stop": 10_000, "early_stop_patience": 10},
    "ba": {"target_rms": 1e-4, "target_max_abs": 1e-2, "min_steps_before_stop": 10_000, "early_stop_patience": 10},
    "discretion": {"target_rms": 5e-4, "target_max_abs": 2e-2, "min_steps_before_stop": 20_000, "early_stop_patience": 10},
    "commitment": {"target_rms": 5e-4, "target_max_abs": 2e-2, "min_steps_before_stop": 20_000, "early_stop_patience": 10},
}


def stop_profile(kind: str) -> dict[str, float | int]:
    return dict(STOP_PROFILES[kind.lower()])


RULE_STATE_NAMES = (
    "D",
    "X",
    "ell_D",
    "ell_X",
    "log_Z",
    "A",
    "log_Delta_prev",
)

NATURAL_STATE_NAMES = (
    "D",
    "X",
    "ell_D",
    "ell_X",
    "log_Z",
    "A",
)

RULE_OUTPUT_NAMES = (
    "C",
    "Y",
    "Pi",
    "Q_A",
    "S_p",
    "F_p",
)

NATURAL_OUTPUT_NAMES = (
    "C_n",
    "Y_n",
    "R_n_real",
)

PRIVATE_RESIDUAL_NAMES = (
    "hh_euler",
    "resource",
    "price_index",
    "calvo_S",
    "calvo_F",
    "Q",
)

OPT_CONTROL_NAMES = (
    "C",
    "Y",
    "R",
    "Pi",
    "Q_A",
    "S_p",
    "F_p",
)

OPT_MULTIPLIER_NAMES = tuple(f"mu_{name}" for name in PRIVATE_RESIDUAL_NAMES)

DISCRETION_OUTPUT_NAMES = OPT_CONTROL_NAMES + ("V",) + OPT_MULTIPLIER_NAMES

COMMITMENT_PROMISE_NAMES = (
    "promise_E",
    "promise_S",
    "promise_F",
    "promise_Q",
)

# Author-style commitment initialization for scaled promises.  The local
# Galo--Nuno code initializes raw pricing promises at vartheta_old=-0.019182
# and rho_old=0.016500, and separately carries c_old=0.921336.  Here we do not
# carry c_old as a state; instead the inherited promises are scaled by the
# inverse marginal utility normalizer, so the pricing means and standard
# deviations below multiply the author raw values by c_old**gamma.
_AUTHOR_C_OLD = 0.921336
_AUTHOR_GAMMA = 2.0
_AUTHOR_C_SCALE = _AUTHOR_C_OLD**_AUTHOR_GAMMA

COMMITMENT_PROMISE_INIT_MEAN = (
    0.0,  # promise_E: Euler-promise analogue; no direct author value
    -0.019182 * _AUTHOR_C_SCALE,  # scaled pricing-promise analogue of vartheta_old
    0.016500 * _AUTHOR_C_SCALE,  # scaled pricing-promise analogue of rho_old
    0.0,  # promise_Q: repair promise; no analogue in Galo--Nuno
)

COMMITMENT_PROMISE_INIT_STD = (
    0.010,  # centered cloud for the Euler promise
    0.027 * _AUTHOR_C_SCALE,
    0.023 * _AUTHOR_C_SCALE,
    0.010,  # centered cloud for the new repair promise
)

COMMITMENT_OUTPUT_NAMES = OPT_CONTROL_NAMES + OPT_MULTIPLIER_NAMES + COMMITMENT_PROMISE_NAMES

COMMITMENT_STATE_NAMES = RULE_STATE_NAMES + COMMITMENT_PROMISE_NAMES
