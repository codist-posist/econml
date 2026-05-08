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
    omega0: float = 0.30
    kappa_a: float = 0.15

    # Competitive supply-price normalizations
    p_d: float = 1.0
    p_a: float = 1.0
    bar_p_m: float = 1.0

    # Imported-input availability and external mapping
    normal_capacity_slack: float = 0.10
    bar_m: float = 0.30
    nu_pD: float = 0.15
    nu_pX: float = 0.05
    nu_qD: float = 0.30
    nu_qX: float = 0.20

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
    delta_A: float = 0.035
    phi_A: float = 3.0
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
    lr: float = 1e-4
    steps: int = 50_000
    loss: str = "huber"
    huber_delta: float = 1.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    fb_epsilon_start: float = 1e-4
    fb_epsilon_final: float = 1e-8


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
    "N",
    "Pi",
    "chi",
    "I_A",
    "Q_A",
    "S_p",
    "F_p",
)

NATURAL_OUTPUT_NAMES = (
    "C_n",
    "Y_n",
    "N_n",
    "chi_n",
    "R_n_real",
)
