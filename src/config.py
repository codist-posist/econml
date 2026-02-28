from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal, Tuple
import torch

PolicyName = Literal["taylor", "mod_taylor", "discretion", "commitment"]
RunMode = Literal["full", "mid", "dev"]


@dataclass(frozen=True)
class ModelParams:
    # Table 1 (Galo–Nuno): key parameters
    beta: float = 0.9975
    gamma: float = 2.0
    omega: float = 1.0

    theta: float = 0.75
    eps: float = 7.0
    tau_bar: float = 1.0 / 7.0  # = 1/eps

    # Shock persistence
    rho_A: float = 0.99
    rho_tau: float = 0.90
    rho_g: float = 0.97

    # Shock volatilities
    sigma_A: float = 0.009
    sigma_tau: float = 0.0014
    sigma_g: float = 0.0052

    # Levels / regimes
    g_bar: float = 0.20
    eta_bar: float = 1.0 / 7.0  # = 1/eps
    bad_state: int = 1  # s=0 normal, s=1 bad

    # Markov transition probabilities (normal->bad, bad->normal)
    p12: float = 1.0 / 48.0
    p21: float = 1.0 / 24.0

    # Taylor-rule objects (paper uses pi_bar = 0)
    pi_bar: float = 0.0
    psi: float = 2.0

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def to_torch(self) -> "ModelParams":
        # Smoke check that device/dtype are valid (does not change params)
        torch.zeros(1, device=self.device, dtype=self.dtype)
        return self

    @property
    def M(self) -> float:
        return self.eps / (self.eps - 1.0)

    @property
    def P(self) -> torch.Tensor:
        # IMPORTANT: P[s_current, s_next] (row-stochastic, as in paper Appendix A.1)
        p11 = 1.0 - self.p12
        p22 = 1.0 - self.p21
        return torch.tensor(
            [[p11, self.p12],
             [self.p21, p22]],
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class PhaseConfig:
    """
    One training phase. Purely compute knobs.

    IMPORTANT CONTRACT:
    - Must NOT change equations, residual definitions, or variable meaning.
    - May change numerical approximation accuracy (GH order) and optimizer settings.
    """
    steps: int
    lr: float
    batch_size: int
    gh_n_train: int
    use_float64: bool = False  # optional: switch dtype for this phase
    eps_stop: float | None = None  # stop early if loss < eps_stop (Appendix B)


@dataclass(frozen=True)
class TrainConfig:
    """
    Training / compute configuration with a fixed network architecture and two phases.
    """
    mode: RunMode = "full"
    seed: int = 123

    # ---- Network (FIXED across phases) ----
    hidden_layers: Tuple[int, int] = (512, 512)
    activation: str = "selu"

    # ---- DEQN simulated path length (Appendix B: N_path_length) ----
    n_path: int = 128

    # ---- Number of simulated paths per optimizer step (vectorized batch of paths) ----
    n_paths_per_step: int = 1

    # ---- Output transforms floors ----
    c_floor: float = 1e-6
    delta_floor: float = 1e-8
    pstar_floor: float = 1e-8

    # ---- Optimization ----
    grad_clip: float = 1.0

    # ---- Artifacts ----
    artifacts_root: str = "../artifacts"
    run_dir: str | None = None
    save_best: bool = True
    # Single canonical checkpoint file for trained/best model weights.
    best_weights_name: str = "weights.pt"

    # ---- CPU / performance knobs (safe: does not change equations) ----
    cpu_num_threads: int | None = None
    cpu_num_interop_threads: int | None = None
    matmul_precision: str = "medium"  # "highest"/"high"/"medium"

    # ---- Logging / profiling ----
    log_every: int = 50
    enable_timers: bool = True
    profile: bool = False
    profile_dir: str = "../artifacts/profiles"
    profile_steps: int = 200

    # ---- Validation / early stopping ----
    val_size: int = 1024
    val_every: int = 2000
    early_stopping: bool = False
    patience: int = 5000
    min_delta: float = 1e-5

    # ---- Paper-style stopping rule (Appendix B): stop on epsilon ----
    # If enabled and a phase has eps_stop set, training runs until loss < eps_stop,
    # with strict_eps_max_steps acting only as a safety cap.
    strict_eps_stop: bool = False
    strict_eps_max_steps: int | None = None

    reduce_lr_on_plateau: bool = False
    plateau_patience: int = 5000
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-7

    # ---- Two-phase schedule (network fixed) ----
    phase1: PhaseConfig = PhaseConfig(
        steps=10_000,
        lr=1e-5,
        batch_size=256,
        gh_n_train=2,
        use_float64=False,
        eps_stop=1e-8,
    )
    phase2: PhaseConfig = PhaseConfig(
        steps=20_000,
        lr=1e-5,
        batch_size=256,
        gh_n_train=3,
        use_float64=True,
        eps_stop=1e-12,
    )

    # ---- Commitment (timeless) initialization ----
    # Under commitment, lagged Ramsey multipliers are part of the state. In a timeless
    # perspective (as in the paper), we do NOT require steady-state multipliers as input;
    # instead we rely on burn-in to enter the ergodic region. These knobs ONLY affect the
    # initial distribution used to start simulations/training.
    commitment_init_multiplier_std: float = 0.0
    commitment_init_multiplier_clip: float = 25.0

    @staticmethod
    def full(**overrides) -> "TrainConfig":
        """
        Full/reproduction preset (more compute, closer to paper-level accuracy):
        - Fixed net: (512,512) SELU
        - Phase 1: GH=2 (stabilization)
        - Phase 2: GH=3 (refinement)
        """
        base = TrainConfig(
            mode="full",
            hidden_layers=(512, 512),
            activation="selu",val_size=1024,
            val_every=2000,
            strict_eps_stop=True,
            strict_eps_max_steps=200_000,
            # NOTE: We keep GH orders aligned with the paper's DEQN description.
            # "full" is intentionally larger than "mid" so there is a meaningful gap.
            phase1 = PhaseConfig(steps=12_000, lr=1e-5, batch_size=256, gh_n_train=2, use_float64=False, eps_stop=1e-8),
            phase2 = PhaseConfig(steps=6_000, lr=1e-5, batch_size=256, gh_n_train=3, use_float64=True, eps_stop=1e-12),
        )
        return replace(base, **overrides)

    @staticmethod
    def mid(**overrides) -> "TrainConfig":
        """
        Mid preset tuned for ~4h CPU budget (16GB RAM):
        - Fixed net: (512,512) SELU (paper)
        - n_path: 50 (was 100)  [compute knob]
        - Phase 1: 6000 steps, GH=2, float32  [compute knob]
        - Phase 2: 1000 steps, GH=3, float64, lower LR for stability near convergence
        - Simulation workflow: quick checks in float32; final table/fig runs in float64 (set ModelParams.dtype accordingly).

        This preset changes ONLY compute knobs (path length / steps) and keeps equations, residual definitions,
        GH orders, and the two-phase float32→float64 methodology intact.
        """
        base = TrainConfig(
            mode="mid",
            hidden_layers=(512, 512),
            activation="selu",val_size=1024,
            val_every=1500,
            strict_eps_stop=False,
            strict_eps_max_steps=None,
            cpu_num_threads=16,
            cpu_num_interop_threads=1,
            log_every=100,
            n_path=50,
            phase1 = PhaseConfig(steps=6_000, lr=1e-5, batch_size=128, gh_n_train=2, use_float64=False, eps_stop=1e-7),
            phase2 = PhaseConfig(steps=1000, lr=3e-6, batch_size=128, gh_n_train=3, use_float64=True, eps_stop=1e-6),
        )
        return replace(base, **overrides)

    @staticmethod
    def dev(**overrides) -> "TrainConfig":
        """
        Dev preset for 8–16GB RAM debugging:
        - Fixed net (smaller by default): (256,256) SELU
        - Phase 1: GH=2
        - Phase 2: GH=3
        """
        base = TrainConfig(
            mode="dev",
            hidden_layers=(256, 256),
            activation="selu",val_size=1024,
            val_every=1000,
            strict_eps_stop=False,
            strict_eps_max_steps=None,
            cpu_num_threads=16,
            cpu_num_interop_threads=1,
            n_path=50,
            phase1 = PhaseConfig(steps=2_000, lr=1e-5, batch_size=64, gh_n_train=2, use_float64=False, eps_stop=5e-7),
            phase2 = PhaseConfig(steps=400, lr=1e-5, batch_size=64, gh_n_train=3, use_float64=True, eps_stop=1e-9),
        )
        return replace(base, **overrides)

    def phases(self) -> Tuple[PhaseConfig, PhaseConfig]:
        return (self.phase1, self.phase2)


def set_seeds(seed: int) -> None:
    import numpy as np, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
