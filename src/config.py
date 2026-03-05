from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal, Tuple
import os
import math
import torch

PolicyName = Literal[
    "taylor",
    "taylor_para",
    "mod_taylor",
    "discretion",
    "commitment",
    "taylor_zlb",
    "mod_taylor_zlb",
    "discretion_zlb",
    "commitment_zlb",
]
RunMode = Literal["full", "mid", "dev", "author"]
TrainingMode = Literal["author", "ours"]
ExogenousInitMode = Literal["author_hooks"]
CommitmentInitMode = Literal["author_hooks"]
WeightSelectionMode = Literal["best", "last"]
GradClipMode = Literal["norm", "value"]
InitMode = Literal["default", "author_variance_scaling"]
AutoStopMetric = Literal["loss", "loss_val", "rms_resid_val"]


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

    # Levels / regimes (3-regime baseline: normal/bad/severe)
    g_bar: float = 0.20
    eta_bar: float = 1.0 / 7.0  # = 1/eps
    eta0: float = 0.0
    eta1: float | None = None
    eta2: float | None = None
    bad_state: int = 1
    severe_state: int = 2

    # Markov transition probabilities:
    # - p12: normal(0) -> bad(1)
    # - p21: bad(1) -> normal(0)
    # - p23: bad(1) -> severe(2)
    # - p32: severe(2) -> bad(1)
    p12: float = 1.0 / 48.0
    p21: float = 1.0 / 24.0
    p23: float = 1.0 / 96.0
    p32: float = 1.0 / 48.0
    # Author taylor_para uses per-path p21 sampled in [p21_l, p21_u].
    p21_l: float = 1.0 / 60.0
    p21_u: float = 1.0
    # Optional state-dependent transitions for critique experiments.
    # If enabled, p12_t and p21_t are obtained by a logit shift driven by xi_t:
    #   logit(p21_t) = logit(p21_base) - p21_xi_slope * (xi_t - xi_ref)
    #   logit(p12_t) = logit(p12_base) + p12_xi_slope * (xi_t - xi_ref)
    # where xi_ref is transition_xi_ref or the stationary mean -sigma_tau^2/2.
    state_dependent_transitions: bool = False
    p21_xi_slope: float = 0.0
    p12_xi_slope: float = 0.0
    transition_xi_ref: float | None = None
    transition_prob_floor: float = 1e-8

    # Regime-dependent uncertainty extension.
    # sigma_tau(s=1) = sigma_tau * sigma_tau_bad_mult
    # sigma_tau(s=2) = sigma_tau * sigma_tau_severe_mult
    sigma_tau_bad_mult: float = 1.0
    sigma_tau_severe_mult: float = 1.0

    # Taylor-rule objects (paper uses pi_bar = 0)
    pi_bar: float = 0.0
    psi: float = 2.0
    rho_i: float = 0.0

    # Device selection:
    # - "auto": use CUDA when available, else CPU
    # - "cuda" / "cuda:N": force specific CUDA device
    # - "cpu": force CPU
    # Environment override: ECONML_DEVICE (same accepted values).
    device: str = "auto"
    dtype: torch.dtype = torch.float32

    def to_torch(self) -> "ModelParams":
        requested = os.environ.get("ECONML_DEVICE", self.device)
        dev_raw = str(requested).strip().lower()
        if dev_raw in ("", "auto"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev_raw in ("gpu", "cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested, but torch.cuda.is_available() is False. "
                    "Install CUDA-enabled PyTorch or set device='cpu'."
                )
            dev = "cuda"
        elif dev_raw.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device '{requested}' requested, but torch.cuda.is_available() is False. "
                    "Install CUDA-enabled PyTorch or set device='cpu'."
                )
            dev = dev_raw
        elif dev_raw == "cpu":
            dev = "cpu"
        else:
            raise ValueError(
                f"Unsupported device={requested!r}. Use 'auto', 'cpu', 'cuda', or 'cuda:N'."
            )

        # Optional hard requirement toggle for launch scripts/CI.
        require_cuda = str(os.environ.get("ECONML_REQUIRE_CUDA", "0")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        if require_cuda and not dev.startswith("cuda"):
            raise RuntimeError(
                "ECONML_REQUIRE_CUDA=1, but CUDA is not available in this environment."
            )

        # Smoke check that device/dtype are valid.
        torch.zeros(1, device=dev, dtype=self.dtype)
        if dev != self.device:
            return replace(self, device=dev)
        return self

    @property
    def M(self) -> float:
        return self.eps / (self.eps - 1.0)

    @property
    def n_regimes(self) -> int:
        return len(self.eta_by_regime)

    @property
    def eta_by_regime(self) -> Tuple[float, float, float]:
        eta1 = float(self.eta_bar) if self.eta1 is None else float(self.eta1)
        eta2 = float(2.0 * eta1) if self.eta2 is None else float(self.eta2)
        return (float(self.eta0), eta1, eta2)

    @property
    def sigma_tau_by_regime(self) -> Tuple[float, float, float]:
        sig0 = max(0.0, float(self.sigma_tau))
        sig1 = sig0 * max(0.0, float(self.sigma_tau_bad_mult))
        sig2 = sig0 * max(0.0, float(self.sigma_tau_severe_mult))
        return (sig0, sig1, sig2)

    @property
    def sigma_tau_by_regime_tensor(self) -> torch.Tensor:
        return torch.tensor(self.sigma_tau_by_regime, device=self.device, dtype=self.dtype)

    @property
    def P(self) -> torch.Tensor:
        # IMPORTANT: P[s_current, s_next] (row-stochastic, as in paper Appendix A.1)
        p12 = float(self.p12)
        p21 = float(self.p21)
        p23 = float(self.p23)
        p32 = float(self.p32)
        if not (0.0 <= p12 <= 1.0 and 0.0 <= p21 <= 1.0 and 0.0 <= p23 <= 1.0 and 0.0 <= p32 <= 1.0):
            raise ValueError(f"Invalid Markov probabilities: p12={p12}, p21={p21}, p23={p23}, p32={p32}")
        p11 = 1.0 - p12
        p22 = 1.0 - p21 - p23
        p33 = 1.0 - p32
        if not (0.0 <= p11 <= 1.0 and 0.0 <= p22 <= 1.0 and 0.0 <= p33 <= 1.0):
            raise ValueError(f"Invalid Markov rows: p11={p11}, p22={p22}, p33={p33}")
        P = torch.tensor(
            [
                [p11, p12, 0.0],
                [p21, p22, p23],
                [0.0, p32, p33],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        rows = P.sum(dim=1)
        if not torch.allclose(rows, torch.ones_like(rows), atol=1e-7, rtol=0.0):
            raise ValueError(f"Transition matrix rows must sum to 1, got {rows.tolist()}")
        return P

    def transition_xi_reference(self) -> float:
        if self.transition_xi_ref is not None and math.isfinite(float(self.transition_xi_ref)):
            return float(self.transition_xi_ref)
        return float(-(float(self.sigma_tau) ** 2) / 2.0)


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
    # Training semantics:
    # - "author": episode-oriented DEQN loop (closest to author code)
    # - "ours": step-oriented loop with optional strict epsilon stop
    training_mode: TrainingMode = "author"
    # Explicit number of episodes for author mode.
    # If None, episode budget is derived from phase.steps for backward compatibility.
    n_episodes: int | None = None
    seed: int = 123

    # ---- Network (FIXED across phases) ----
    hidden_layers: Tuple[int, int] = (512, 512)
    activation: str = "selu"
    init_mode: InitMode = "default"
    init_scale: float = 0.01

    # ---- DEQN simulated path length (Appendix B: N_path_length) ----
    n_path: int = 128

    # ---- Number of simulated paths per optimizer step (vectorized batch of paths) ----
    n_paths_per_step: int = 1
    # Optional per-phase overrides for n_paths_per_step.
    # If set, they are used for the corresponding phase; otherwise `n_paths_per_step` is used.
    n_paths_per_step_phase1: int | None = None
    n_paths_per_step_phase2: int | None = None

    # ---- Output transforms floors ----
    c_floor: float = 1e-6
    delta_floor: float = 1e-8
    pstar_floor: float = 1e-8
    # Author-like hard bounds for unstable nominal variables.
    use_author_bounds: bool = True
    pi_low: float = -0.1
    pi_high: float = 0.1
    pstar_low: float = 0.9
    pstar_high: float = 1.1
    # Penalty-based bounds (author-like): typically enabled for training_mode="author".
    use_penalty_bounds: bool = True
    use_author_raw_penalty: bool = True
    bounds_penalty_weight: float = 1.0

    # ---- Optimization ----
    grad_clip: float = 1.0
    grad_clip_mode: GradClipMode = "norm"
    # Author-like robust objective.
    loss_type: str = "huber"  # "huber" | "mse"
    huber_delta: float = 1.0
    # Optional micro-batch size for residual backpropagation.
    # This does not change the objective, only memory/throughput behavior.
    residual_chunk_size: int | None = None
    # Author-like per-episode LR scheduler (Hooks.py-compatible when decay=1.0).
    use_author_lr_scheduler: bool = False
    author_lr_decay: float = 1.0
    author_lr_min: float = 1e-7
    author_lr_warmup_episodes: int = 0
    # Author Main.py-style shuffle (limited buffer) in episode loop.
    author_limited_shuffle: bool = True
    # Author Variables.py N_sim_batch (can differ from minibatch size).
    author_n_sim_batch: int | None = None
    # Author Main.py initialize_each_episode + Hooks.post_init().
    author_initialize_each_episode: bool = True
    # Optional safety cap for author loop. None => no hard cap.
    author_step_cap: int | None = None
    # Optional author-code override used in dsge_zlb_commitment Variables.py.
    author_commitment_zlb_p12: float | None = None
    # Stabilization trick in author Taylor dynamics: keep some batches at zero shocks.
    author_n_steady_state_batches: int = 50
    author_n_steady_state_min_batch: int = 500

    # ---- Artifacts ----
    artifacts_root: str = "../artifacts"
    run_dir: str | None = None
    save_best: bool = True
    # Single canonical checkpoint file for trained/best model weights.
    best_weights_name: str = "weights.pt"
    # Which checkpoint should be treated as canonical training output:
    # - "best": minimum training loss (closest to author's epsilon-style target)
    # - "last": final optimizer iterate (closest to author's checkpoint-manager "latest")
    weights_selection: WeightSelectionMode = "best"

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

    # ---- Automatic training stop (useful for long author-mode runs) ----
    # Metrics:
    # - "loss": current optimization batch loss
    # - "loss_val": fixed validation states objective
    # - "rms_resid_val": fixed validation states residual RMS
    auto_stop_enabled: bool = False
    auto_stop_metric: AutoStopMetric = "loss"
    auto_stop_warmup_steps: int = 0
    auto_stop_patience_steps: int = 0
    auto_stop_min_delta: float = 0.0
    auto_stop_min_rel_delta: float = 0.0

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
    use_two_phase: bool = True
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

    # ---- Initialization (author Hooks.py semantics only) ----
    exogenous_init_mode: ExogenousInitMode = "author_hooks"
    commitment_init_mode: CommitmentInitMode = "author_hooks"
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
            training_mode="ours",
            hidden_layers=(512, 512),
            activation="selu",
            val_size=1024,
            val_every=2000,
            strict_eps_stop=True,
            strict_eps_max_steps=200_000,
            # NOTE: We keep GH orders aligned with the paper's DEQN description.
            # "full" is intentionally larger than "mid" so there is a meaningful gap.
            phase1=PhaseConfig(steps=12_000, lr=1e-5, batch_size=256, gh_n_train=2, use_float64=False, eps_stop=1e-8),
            phase2=PhaseConfig(steps=6_000, lr=1e-5, batch_size=256, gh_n_train=3, use_float64=True, eps_stop=1e-12),
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
            training_mode="ours",
            hidden_layers=(512, 512),
            activation="selu",
            val_size=1024,
            val_every=1500,
            strict_eps_stop=False,
            strict_eps_max_steps=None,
            cpu_num_threads=16,
            cpu_num_interop_threads=1,
            log_every=100,
            n_path=50,
            phase1=PhaseConfig(steps=6_000, lr=1e-5, batch_size=128, gh_n_train=2, use_float64=False, eps_stop=1e-7),
            phase2=PhaseConfig(steps=1000, lr=3e-6, batch_size=128, gh_n_train=3, use_float64=True, eps_stop=1e-6),
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
            training_mode="ours",
            hidden_layers=(256, 256),
            activation="selu",
            val_size=1024,
            val_every=1000,
            strict_eps_stop=False,
            strict_eps_max_steps=None,
            cpu_num_threads=16,
            cpu_num_interop_threads=1,
            n_path=50,
            phase1=PhaseConfig(steps=2_000, lr=1e-5, batch_size=64, gh_n_train=2, use_float64=False, eps_stop=5e-7),
            phase2=PhaseConfig(steps=400, lr=1e-5, batch_size=64, gh_n_train=3, use_float64=True, eps_stop=1e-9),
        )
        return replace(base, **overrides)

    @staticmethod
    def author_like(*, policy: PolicyName | None = None, **overrides) -> "TrainConfig":
        """
        Author-like compute preset (closest to public Keras code semantics):
        - author training mode
        - author Hooks.py state initialization (exogenous + commitment)
        - single phase by default (episode+minibatch training analog)
        - GH=3, Adam lr=1e-5, minibatch size=128, episode length=10
        - no hard eps-stop (monitor loss / checkpoints)
        """
        # Strict author-code architecture parity:
        # - dsge_taylor uses 128x128
        # - dsge_discretion / dsge_commitment / dsge_taylor_para /
        #   dsge_zlb_discretion / dsge_zlb_commitment use 512x512
        if policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            hidden = (128, 128)
        else:
            hidden = (512, 512)
        if policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "taylor_para"):
            sim_batch = 512
        else:
            sim_batch = 1024

        # Quality-oriented defaults for long author runs:
        # Use the longer Taylor-like early-stop window for discretion/commitment too,
        # as requested for better convergence robustness.
        long_patience_family = policy in (
            "taylor",
            "mod_taylor",
            "taylor_zlb",
            "mod_taylor_zlb",
            "discretion",
            "commitment",
        )
        auto_warmup = 10_000 if long_patience_family else 6_000
        auto_patience = 40_000 if long_patience_family else 25_000
        auto_min_rel_delta = 1e-5

        base = TrainConfig(
            mode="author",
            training_mode="author",
            exogenous_init_mode="author_hooks",
            commitment_init_mode="author_hooks",
            # Prefer the best checkpoint in long runs; `last` can be selected via override.
            weights_selection="best",
            n_episodes=10_000_000,
            hidden_layers=hidden,
            activation="selu",
            init_mode="author_variance_scaling",
            init_scale=0.01,
            use_two_phase=False,
            strict_eps_stop=False,
            strict_eps_max_steps=None,
            author_step_cap=None,
            # Keep baseline calibration unless user explicitly overrides via kwargs.
            author_commitment_zlb_p12=None,
            n_path=10,
            n_paths_per_step=1,
            grad_clip_mode="value",
            use_author_raw_penalty=False,
            use_author_lr_scheduler=True,
            author_lr_decay=1.0,
            author_lr_min=1e-7,
            author_lr_warmup_episodes=0,
            author_limited_shuffle=True,
            author_n_sim_batch=sim_batch,
            author_initialize_each_episode=True,
            author_n_steady_state_batches=50,
            author_n_steady_state_min_batch=500,
            # Practical default for very large episode budgets:
            # stop if fixed validation residual RMS plateaus for a long window.
            auto_stop_enabled=True,
            auto_stop_metric="rms_resid_val",
            auto_stop_warmup_steps=auto_warmup,
            auto_stop_patience_steps=auto_patience,
            auto_stop_min_delta=0.0,
            auto_stop_min_rel_delta=auto_min_rel_delta,
            log_every=100,
            val_size=2048,
            val_every=2000,
            phase1=PhaseConfig(steps=200_000, lr=1e-5, batch_size=128, gh_n_train=3, use_float64=False, eps_stop=None),
            # Kept for compatibility (not used when use_two_phase=False).
            phase2=PhaseConfig(steps=0, lr=1e-5, batch_size=128, gh_n_train=3, use_float64=False, eps_stop=None),
        )
        return replace(base, **overrides)

    def phases(self) -> Tuple[PhaseConfig, ...]:
        if not bool(getattr(self, "use_two_phase", True)):
            return (self.phase1,)
        if int(self.phase2.steps) <= 0:
            return (self.phase1,)
        return (self.phase1, self.phase2)


def set_seeds(seed: int) -> None:
    import numpy as np, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
