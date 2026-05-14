from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn

from .config import (
    BaselineParams,
    COMMITMENT_PROMISE_INIT_MEAN,
    COMMITMENT_PROMISE_INIT_STD,
    COMMITMENT_OUTPUT_NAMES,
    COMMITMENT_PROMISE_NAMES,
    COMMITMENT_STATE_NAMES,
    DISCRETION_OUTPUT_NAMES,
    NATURAL_OUTPUT_NAMES,
    RULE_OUTPUT_NAMES,
    RULE_STATE_NAMES,
    NetworkConfig,
    QMCConfig,
    TrainConfig,
)
from .networks import MLP
from .natural_oracle import natural_benchmark_outputs
from .qmc import make_qmc_nodes
from .residuals import natural_residuals, rule_residuals, stack_residuals
from .sampling import natural_from_rule_states, sample_rule_states
from .episode import simulate_rule_episode
from .economics import adaptation_enabled, derive_free, derive_rule, psi_prime, unpack_rule_state
from .optimal import (
    commitment_residuals,
    decode_commitment,
    decode_discretion,
    discretion_residuals,
    private_residuals_free,
    simulate_optimal_episode,
)
from .transforms import decode_natural_outputs, decode_rule_outputs, steady_decode_targets


@dataclass
class TrainLog:
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    max_abs: list[float] = field(default_factory=list)
    rms: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_max_abs: list[float] = field(default_factory=list)
    val_rms: list[float] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None
    best_step: int | None = None
    best_val_rms: float | None = None
    best_val_max_abs: float | None = None
    best_train_rms: float | None = None
    best_selection_score: float | None = None
    best_selection_criterion: str | None = None
    best_scenario_q_rms: float | None = None
    best_calm_anchor_rms: float | None = None
    best_calm_residual_rms: float | None = None
    extra_metrics: list[dict[str, float]] = field(default_factory=list)


def make_natural_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    """Build the auxiliary flexible-price benchmark network."""

    return MLP(6, len(NATURAL_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def make_rule_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    """Build a rule-based DEQN network."""

    return MLP(7, len(RULE_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def make_discretion_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    return MLP(len(RULE_STATE_NAMES), len(DISCRETION_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def make_commitment_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    return MLP(len(COMMITMENT_STATE_NAMES), len(COMMITMENT_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def commitment_promise_init_tensors(
    *, device: torch.device | str, dtype: torch.dtype, scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return author-style commitment promise means and scaled standard deviations."""

    mean = torch.tensor(COMMITMENT_PROMISE_INIT_MEAN, device=device, dtype=dtype)
    std = float(scale) * torch.tensor(COMMITMENT_PROMISE_INIT_STD, device=device, dtype=dtype)
    return mean, std


def initialize_commitment_promises(net: MLP, *, scale: float = 1.0) -> None:
    """Seed inherited commitment promises away from the degenerate zero state."""

    last = net.net[-1]
    if not isinstance(last, nn.Linear):
        return
    n_promises = len(COMMITMENT_PROMISE_NAMES)
    start = len(COMMITMENT_OUTPUT_NAMES) - n_promises
    mean, _ = commitment_promise_init_tensors(device=last.bias.device, dtype=last.bias.dtype, scale=scale)
    with torch.no_grad():
        last.bias[start : start + n_promises].copy_(mean)


def freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def residual_loss(
    residual_matrix: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    loss: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    kind = loss.lower().strip()
    if kind == "mse":
        values = residual_matrix.pow(2)
    elif kind == "huber":
        delta = float(huber_delta)
        abs_resid = residual_matrix.abs()
        values = torch.where(abs_resid <= delta, 0.5 * residual_matrix.pow(2), delta * (abs_resid - 0.5 * delta))
    else:
        raise ValueError("loss must be 'huber' or 'mse'.")
    if weights is None:
        return values.mean()
    return (values * weights.view(1, -1)).mean()


def _rule_residual_weight_vector(
    residuals: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights_by_name = {
        "hh_euler": train_cfg.rule_hh_euler_weight,
        "resource": train_cfg.rule_resource_weight,
        "price_index": train_cfg.rule_price_index_weight,
        "calvo_S": train_cfg.rule_calvo_s_weight,
        "calvo_F": train_cfg.rule_calvo_f_weight,
        "Q": train_cfg.rule_q_weight,
    }
    return torch.tensor([float(weights_by_name.get(name, 1.0)) for name in residuals.keys()], device=device, dtype=dtype)


def _rule_residual_loss(residuals: Dict[str, torch.Tensor], train_cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    mat = stack_residuals(residuals)
    weights = _rule_residual_weight_vector(residuals, train_cfg, device=mat.device, dtype=mat.dtype)
    loss = residual_loss(mat, weights=weights, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)
    return mat, loss


def _mse_rms_max(resid: torch.Tensor) -> tuple[float, float, float]:
    with torch.no_grad():
        loss = float(resid.pow(2).mean().detach().cpu())
        rms = float(torch.sqrt(resid.pow(2).mean()).detach().cpu())
        max_abs = float(resid.abs().max().detach().cpu())
    return loss, rms, max_abs


def _log_metrics(
    step: int,
    resid: torch.Tensor,
    log: TrainLog,
    objective: torch.Tensor | None = None,
    val_resid: torch.Tensor | None = None,
) -> dict[str, float]:
    log.steps.append(int(step))
    if objective is None:
        train_loss, train_rms, train_max = _mse_rms_max(resid)
    else:
        _, train_rms, train_max = _mse_rms_max(resid)
        train_loss = float(objective.detach().cpu())
    log.losses.append(train_loss)
    log.rms.append(train_rms)
    log.max_abs.append(train_max)

    metrics = {"train_loss": train_loss, "train_rms": train_rms, "train_max_abs": train_max}
    if val_resid is not None:
        val_loss, val_rms, val_max = _mse_rms_max(val_resid)
        log.val_losses.append(val_loss)
        log.val_rms.append(val_rms)
        log.val_max_abs.append(val_max)
        metrics.update({"val_loss": val_loss, "val_rms": val_rms, "val_max_abs": val_max})
    return metrics


def _passes_stop_criteria(
    resid: torch.Tensor,
    cfg: TrainConfig,
    step: int,
    metrics: dict[str, float] | None = None,
) -> bool:
    validation_target_active = cfg.target_rms is not None or cfg.target_max_abs is not None
    scenario_target_active = (
        cfg.target_scenario_q_rms is not None
        and metrics is not None
        and "scenario_Q.rms" in metrics
    )
    if not validation_target_active:
        return False
    if step < int(cfg.min_steps_before_stop):
        return False
    with torch.no_grad():
        rms = float(torch.sqrt(resid.pow(2).mean()).detach().cpu())
        max_abs = float(resid.abs().max().detach().cpu())
    rms_ok = True if cfg.target_rms is None else rms <= float(cfg.target_rms)
    max_ok = True if cfg.target_max_abs is None else max_abs <= float(cfg.target_max_abs)
    scenario_ok = True
    if scenario_target_active:
        scenario_ok = float(metrics["scenario_Q.rms"]) <= float(cfg.target_scenario_q_rms)
    return rms_ok and max_ok and scenario_ok


def _mark_stopped(log: TrainLog, step: int, cfg: TrainConfig) -> None:
    log.stopped_early = True
    log.stop_reason = (
        f"validation and scenario criteria satisfied for {int(cfg.early_stop_patience)} consecutive checks "
        f"at step/episode {int(step)}"
    )


def _progress_range(total: int, *, desc: str, enabled: bool):
    base = range(1, int(total) + 1)
    if not enabled:
        return base
    try:
        from tqdm.auto import tqdm

        return tqdm(base, total=int(total), desc=desc, dynamic_ncols=True)
    except Exception:
        return base


def _report_progress(progress, metrics: dict[str, float], *, step: int, total: int, stop_hits: int, enabled: bool) -> None:
    if not enabled:
        return
    stage = metrics.get("stage")
    payload = {
        "train_rms": f"{metrics['train_rms']:.2e}",
        "val_rms": f"{metrics.get('val_rms', float('nan')):.2e}",
        "val_max": f"{metrics.get('val_max_abs', float('nan')):.2e}",
        "stop": int(stop_hits),
    }
    if stage:
        payload["stage"] = str(stage)
    message = (
        f"[{step}/{total}] train_rms={payload['train_rms']} "
        f"val_rms={payload['val_rms']} val_max={payload['val_max']} stop_hits={payload['stop']}"
    )
    if stage:
        message += f" stage={stage}"
    score = metrics.get("selection_score")
    if score is not None and math.isfinite(float(score)):
        payload["score"] = f"{float(score):.2e}"
        message += f" score={float(score):.2e}"
    q_rms = metrics.get("scenario_Q.rms")
    if q_rms is not None and math.isfinite(float(q_rms)):
        payload["q_rms"] = f"{float(q_rms):.2e}"
        message += f" qRMS={float(q_rms):.2e}"
    calm = metrics.get("calm_anchor.rms")
    if calm is not None and math.isfinite(float(calm)):
        payload["calm"] = f"{float(calm):.2e}"
        message += f" calm={float(calm):.2e}"
    calm_resid = metrics.get("calm_residual.rms")
    if calm_resid is not None and math.isfinite(float(calm_resid)):
        payload["calm_res"] = f"{float(calm_resid):.2e}"
        message += f" calmRes={float(calm_resid):.2e}"
    full_weight = metrics.get("full_weight")
    if full_weight is not None and math.isfinite(float(full_weight)):
        payload["full_w"] = f"{float(full_weight):.2f}"
        message += f" fullW={float(full_weight):.2f}"
    val_top = metrics.get("val_top")
    if val_top:
        message += f" top_val={val_top}"
    q_d3 = metrics.get("scenario_Q.D_3x.event")
    if q_d3 is not None:
        message += f" qD3={float(q_d3):.2e}"
    q_d1 = metrics.get("scenario_Q.D_1x.event")
    if q_d1 is not None:
        message += f" qD1={float(q_d1):.2e}"
    activation_max = metrics.get("scenario_repair_activation.max")
    if activation_max is not None and math.isfinite(float(activation_max)):
        payload["act_max"] = f"{float(activation_max):.2e}"
        message += f" actMax={float(activation_max):.2e}"
    repair_max = metrics.get("scenario_I_A.max")
    if repair_max is not None and math.isfinite(float(repair_max)):
        payload["I_A_max"] = f"{float(repair_max):.2e}"
        message += f" Imax={float(repair_max):.2e}"
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(payload)
    if metrics.get("new_best"):
        message += " best=*"
    print(message, flush=True)


def _should_update_train_postfix(step: int, log_every: int) -> bool:
    """Refresh cheap train-only progress between expensive validation passes."""

    interval = max(1, min(10, int(log_every) // 10 if int(log_every) > 1 else 1))
    return int(step) == 1 or int(step) % interval == 0


def _report_train_postfix(
    progress,
    mat: torch.Tensor,
    loss: torch.Tensor,
    *,
    step: int,
    stop_hits: int,
    enabled: bool,
    stage: str | None = None,
) -> None:
    if not enabled or not hasattr(progress, "set_postfix"):
        return
    with torch.no_grad():
        train_rms = float(torch.sqrt(mat.detach().pow(2).mean()).cpu())
        loss_value = float(loss.detach().cpu())
    payload = {
        "train_rms": f"{train_rms:.2e}",
        "loss": f"{loss_value:.2e}",
        "stop": int(stop_hits),
    }
    if stage:
        payload["stage"] = str(stage)
    progress.set_postfix(payload)


def _announce_training(
    *,
    kind: str,
    total: int,
    train_cfg: TrainConfig,
    qmc_cfg: QMCConfig,
    log_every: int,
) -> None:
    if not train_cfg.show_progress:
        return
    optimal_bits = ""
    if kind in {"discretion", "commitment"}:
        optimal_bits = (
            f", feasibility_pretrain={int(train_cfg.optimal_feasibility_pretrain_steps)}, "
            f"full_warmup={int(train_cfg.optimal_full_weight_warmup_steps)}, "
            f"stat_w={float(train_cfg.optimal_stationarity_loss_weight):g}, "
            f"bellman_w={float(train_cfg.optimal_bellman_loss_weight):g}, "
            f"promise_w={float(train_cfg.optimal_promise_loss_weight):g}"
        )
    print(
        f"Starting {kind}: total={int(total)}, batch_size={int(train_cfg.batch_size)}, "
        f"sim_batch_size={int(train_cfg.sim_batch_size)}, episode_length={int(train_cfg.episode_length)}, "
        f"updates_per_episode={int(train_cfg.episode_updates_per_episode)}, "
        f"broad_share={float(train_cfg.episode_broad_share):.2f}, "
        f"qmc_train={int(qmc_cfg.n_train)}, qmc_val={int(qmc_cfg.n_val)}, "
        f"stop_val_states={int(train_cfg.stop_val_states)}, log_every={int(log_every)}, "
        f"device={train_cfg.device}, dtype={train_cfg.dtype}{optimal_bits}",
        flush=True,
    )


def _top_residual_summary(residuals: Dict[str, torch.Tensor], *, limit: int = 3) -> str:
    with torch.no_grad():
        items = []
        for name, value in residuals.items():
            rms = torch.sqrt(value.detach().pow(2).mean())
            items.append((float(rms.cpu()), name))
    items.sort(reverse=True)
    return ", ".join(f"{name}:{rms:.2e}" for rms, name in items[: int(limit)])


def _optimal_training_stage(step: int, train_cfg: TrainConfig) -> str:
    pretrain = max(0, int(train_cfg.optimal_feasibility_pretrain_steps))
    return "feas" if int(step) <= pretrain else "full"


def _optimal_full_weight(step: int | None, train_cfg: TrainConfig) -> float:
    if step is None:
        return 1.0
    pretrain = max(0, int(train_cfg.optimal_feasibility_pretrain_steps))
    warmup = max(1, int(train_cfg.optimal_full_weight_warmup_steps))
    return min(1.0, max(0.0, (int(step) - pretrain) / warmup))


def _nonfinite_residual_summary(residuals: Dict[str, torch.Tensor], *, limit: int = 6) -> str:
    bad: list[str] = []
    with torch.no_grad():
        for name, value in residuals.items():
            finite = torch.isfinite(value.detach())
            if bool(finite.all().cpu()):
                continue
            total = int(value.numel())
            count = int((~finite).sum().detach().cpu())
            bad.append(f"{name}:{count}/{total}")
            if len(bad) >= int(limit):
                break
    return ", ".join(bad) if bad else "none"


def _optimal_objective_matrix(
    residuals: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    *,
    stage: str,
    step: int | None = None,
) -> torch.Tensor:
    """Return the loss matrix used for optimal-policy updates.

    Raw validation residuals are left untouched.  The feasibility stage fits
    private implementability first; the full stage then down-weights FOC-style
    residuals so they do not swamp the Calvo/resource/Q feasibility block.
    """

    pieces: list[torch.Tensor] = []
    full_weight = _optimal_full_weight(step, train_cfg) if stage == "full" else 0.0
    for name, value in residuals.items():
        if stage == "feas" and not name.startswith("priv_"):
            continue
        weight = float(train_cfg.optimal_private_loss_weight)
        if name == "bellman":
            weight = full_weight * float(train_cfg.optimal_bellman_loss_weight)
        elif name.startswith("stat_"):
            weight = full_weight * float(train_cfg.optimal_stationarity_loss_weight)
        elif name.startswith("promise_"):
            weight = full_weight * float(train_cfg.optimal_promise_loss_weight)
        pieces.append(value * weight)
    if not pieces:
        return stack_residuals(residuals)
    return torch.stack(pieces, dim=-1)


def _optimal_training_residuals_for_stage(
    z: torch.Tensor,
    raw: torch.Tensor,
    net: MLP,
    nodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    fb_epsilon: float,
    stage: str,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    key = kind.lower()
    if stage == "feas":
        out = _decode_optimal_for_kind(raw, key, params=params)
        private, drv = private_residuals_free(
            z,
            out,
            net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=fb_epsilon,
            commitment=key == "commitment",
        )
        return {f"priv_{name}": value for name, value in private.items()}, drv
    if key == "discretion":
        return discretion_residuals(
            z,
            raw,
            net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=fb_epsilon,
        )
    return commitment_residuals(
        z,
        raw,
        net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
    )


def _copy_state_dict_to_cpu(net: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def _metric_if_finite(metrics: dict[str, float], name: str) -> float | None:
    if name not in metrics:
        return None
    value = float(metrics[name])
    return value if math.isfinite(value) else None


def _checkpoint_selection_score(metrics: dict[str, float], cfg: TrainConfig) -> tuple[float, str]:
    val_rms = _metric_if_finite(metrics, "val_rms")
    if val_rms is None:
        return float("inf"), "unavailable"
    score = val_rms
    parts = ["val_rms"]
    q_rms = _metric_if_finite(metrics, "scenario_Q.rms")
    q_weight = float(cfg.best_scenario_q_weight)
    if q_rms is not None and q_weight != 0.0:
        score += q_weight * q_rms
        parts.append(f"{q_weight:g}*scenario_Q.rms")
    calm_rms = _metric_if_finite(metrics, "calm_anchor.rms")
    calm_weight = float(cfg.best_calm_anchor_weight)
    if calm_rms is not None and calm_weight != 0.0:
        score += calm_weight * calm_rms
        parts.append(f"{calm_weight:g}*calm_anchor.rms")
    calm_resid_rms = _metric_if_finite(metrics, "calm_residual.rms")
    calm_resid_weight = float(cfg.best_calm_residual_weight)
    if calm_resid_rms is not None and calm_resid_weight != 0.0:
        score += calm_resid_weight * calm_resid_rms
        parts.append(f"{calm_resid_weight:g}*calm_residual.rms")
    return score, "min_" + "_plus_".join(parts)


def _maybe_save_best_checkpoint(
    net: nn.Module,
    log: TrainLog,
    metrics: dict[str, float],
    step: int,
    cfg: TrainConfig,
    *,
    extra: Dict[str, object] | None = None,
) -> Path | None:
    if cfg.checkpoint_dir is None:
        return None
    directory = Path(cfg.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{str(cfg.checkpoint_name)}_best.pt"
    scalar_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))
    }
    payload = {
        "state_dict": net.state_dict(),
        "metadata": _cpu_detached(
            {
                "step": int(step),
                "selection_score": log.best_selection_score,
                "selection_criterion": log.best_selection_criterion,
                "metrics": scalar_metrics,
                "extra": extra or {},
            }
        ),
    }
    torch.save(payload, path)
    return path


def _maybe_update_best_state(
    net: nn.Module,
    log: TrainLog,
    metrics: dict[str, float],
    step: int,
    best_state: dict[str, torch.Tensor] | None,
    cfg: TrainConfig,
    *,
    extra: Dict[str, object] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Track the best checkpoint using validation plus targeted scenario diagnostics."""

    if "val_rms" not in metrics:
        return best_state
    selection_score, selection_criterion = _checkpoint_selection_score(metrics, cfg)
    metrics["selection_score"] = selection_score
    val_rms = float(metrics["val_rms"])
    val_max = float(metrics.get("val_max_abs", float("inf")))
    train_rms = float(metrics.get("train_rms", float("nan")))
    is_better = log.best_selection_score is None
    if log.best_selection_score is not None:
        tol = 1e-12
        is_better = selection_score < float(log.best_selection_score) - tol or (
            abs(selection_score - float(log.best_selection_score)) <= tol
            and (
                val_rms < float(log.best_val_rms) - tol
                if log.best_val_rms is not None
                else True
            )
        ) or (
            abs(selection_score - float(log.best_selection_score)) <= tol
            and log.best_val_rms is not None
            and abs(val_rms - float(log.best_val_rms)) <= tol
            and (log.best_val_max_abs is None or val_max < float(log.best_val_max_abs))
        )
    if not is_better:
        metrics["new_best"] = False
        return best_state

    log.best_step = int(step)
    log.best_selection_score = selection_score
    log.best_selection_criterion = selection_criterion
    log.best_val_rms = val_rms
    log.best_val_max_abs = val_max
    log.best_train_rms = train_rms
    log.best_scenario_q_rms = _metric_if_finite(metrics, "scenario_Q.rms")
    log.best_calm_anchor_rms = _metric_if_finite(metrics, "calm_anchor.rms")
    log.best_calm_residual_rms = _metric_if_finite(metrics, "calm_residual.rms")
    metrics["new_best"] = True
    best_path = _maybe_save_best_checkpoint(net, log, metrics, step, cfg, extra=extra)
    if best_path is not None:
        metrics["best_checkpoint"] = str(best_path)
    return _copy_state_dict_to_cpu(net)


def _restore_best_state(net: nn.Module, best_state: dict[str, torch.Tensor] | None) -> None:
    if best_state is not None:
        net.load_state_dict(best_state)


def _validation_nodes(qmc_cfg: QMCConfig, *, device: str, dtype: torch.dtype, seed_offset: int) -> QMCNodes:
    cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + int(seed_offset))
    return make_qmc_nodes(cfg.n_train, cfg=cfg, device=device, dtype=dtype)


_RULE_SCENARIO_LABELS = ("no_event", "D_1x", "D_3x", "D_1x_X_lag", "D_3x_X_lag")
_RULE_SCENARIO_OFFSETS = {
    "pre_event": -1,
    "event": 0,
    "event_plus_1": 1,
    "event_plus_4": 4,
}
_RULE_SCENARIO_POINTS = (
    ("no_event", "event"),
    ("D_1x", "event"),
    ("D_1x", "event_plus_1"),
    ("D_3x", "event"),
    ("D_3x", "event_plus_1"),
    ("D_3x", "event_plus_4"),
)


def _normal_rule_state(batch_size: int, *, params: BaselineParams, device: str, dtype: torch.dtype) -> torch.Tensor:
    D = torch.zeros(batch_size, device=device, dtype=dtype)
    X = torch.zeros_like(D)
    ell_D = torch.full_like(D, float(params.log_bar_lambda_D))
    ell_X = torch.full_like(D, float(params.log_bar_lambda_X))
    log_Z = torch.full_like(D, float(-0.5 * params.sigma_z**2))
    A = torch.zeros_like(D)
    log_Delta = torch.zeros_like(D)
    return torch.stack([D, X, ell_D, ell_X, log_Z, A, log_Delta], dim=-1)


def _rule_scenario_additions(
    *,
    t: int,
    pulse: int,
    relief_lag: int,
    params: BaselineParams,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    specs = {
        "no_event": {},
        "D_1x": {pulse: (float(params.mark_D), 0.0)},
        "D_3x": {pulse: (3.0 * float(params.mark_D), 0.0)},
        "D_1x_X_lag": {
            pulse: (float(params.mark_D), 0.0),
            pulse + int(relief_lag): (0.0, float(params.mark_X)),
        },
        "D_3x_X_lag": {
            pulse: (3.0 * float(params.mark_D), 0.0),
            pulse + int(relief_lag): (0.0, float(params.mark_X)),
        },
    }
    add_D = []
    add_X = []
    for label in _RULE_SCENARIO_LABELS:
        d, x = specs[label].get(int(t), (0.0, 0.0))
        add_D.append(float(d))
        add_X.append(float(x))
    return (
        torch.tensor(add_D, device=device, dtype=dtype),
        torch.tensor(add_X, device=device, dtype=dtype),
    )


def _deterministic_rule_step(
    z: torch.Tensor,
    *,
    A_next: torch.Tensor,
    Delta_next: torch.Tensor,
    add_D: torch.Tensor,
    add_X: torch.Tensor,
    params: BaselineParams,
) -> torch.Tensor:
    st = unpack_rule_state(z)
    D_next = (1.0 - float(params.delta_D)) * st.D + add_D
    X_next = (1.0 - float(params.delta_X)) * st.X + add_X
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
    return torch.stack([D_next, X_next, ell_D_next, ell_X_next, log_Z_next, A_next, log_Delta_next], dim=-1)


def _rule_training_scenario_states(
    net: MLP,
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, list[str]]:
    """Deterministic no-event/crisis states used to discipline value-function learning."""

    device = train_cfg.device
    dtype = train_cfg.dtype
    burnin = max(1, int(train_cfg.rule_scenario_burnin))
    horizon = max(5, int(train_cfg.rule_scenario_horizon))
    total = burnin + horizon + 1
    z = _normal_rule_state(len(_RULE_SCENARIO_LABELS), params=params, device=device, dtype=dtype)
    states = [z]
    with torch.no_grad():
        for t in range(1, total):
            st = unpack_rule_state(z)
            uses_natural_y_ref = abs(float(params.phi_y)) > 1e-14
            if uses_natural_y_ref or policy.lower() == "ba":
                out_n = natural_benchmark_outputs(
                    z[..., :6],
                    natural_net,
                    params=params,
                    need_rate=policy.lower() == "ba",
                )
            else:
                y_ref = torch.full_like(z[..., 0], float(params.steady_state_output))
                out_n = {"Y_n": y_ref, "R_n_real": torch.full_like(y_ref, float(params.bar_R))}
            out = decode_rule_outputs(
                net(z),
                RULE_OUTPUT_NAMES,
                params=params,
                y_ref=out_n["Y_n"] if uses_natural_y_ref else None,
            )
            drv = derive_rule(st, out, params, Y_n=out_n["Y_n"], R_n=out_n["R_n_real"], policy=policy)
            add_D, add_X = _rule_scenario_additions(
                t=t,
                pulse=burnin,
                relief_lag=4,
                params=params,
                device=z.device,
                dtype=z.dtype,
            )
            z = _deterministic_rule_step(
                z,
                A_next=drv["A_next"],
                Delta_next=drv["Delta"],
                add_D=add_D,
                add_X=add_X,
                params=params,
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


def _rule_scenario_residuals(
    net: MLP,
    natural_net: MLP,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list[str]]:
    z, names = _rule_training_scenario_states(net, natural_net, policy=policy, params=params, train_cfg=train_cfg)
    res, drv = rule_residuals(
        z,
        net(z),
        net,
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        policy=policy,
    )
    return res, drv, names


def _rule_calm_anchor_loss(
    net: MLP,
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    return _calm_anchor_loss_from_terms(
        _rule_calm_anchor_terms(net, natural_net, policy=policy, params=params, train_cfg=train_cfg)
    )


def _pricing_sum_targets(y_target: torch.Tensor, params: BaselineParams) -> tuple[torch.Tensor, torch.Tensor]:
    denom = max(1.0 - float(params.theta) * float(params.beta), 1e-8)
    f_target = y_target / denom
    mc_target = (float(params.epsilon) - 1.0) / float(params.epsilon)
    s_target = mc_target * f_target
    return s_target, f_target


def _calm_anchor_matrix(terms: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([term.reshape(-1) for term in terms], dim=-1)


def _calm_anchor_loss_from_terms(terms: list[torch.Tensor]) -> torch.Tensor:
    return _calm_anchor_matrix(terms).pow(2).mean()


def _calm_anchor_diagnostics_from_terms(terms: list[torch.Tensor]) -> Dict[str, float]:
    mat = _calm_anchor_matrix(terms).detach()
    return {
        "calm_anchor.rms": float(torch.sqrt(mat.pow(2).mean()).cpu()),
        "calm_anchor.max_abs": float(mat.abs().max().cpu()),
    }


def _residual_matrix_diagnostics(prefix: str, mat: torch.Tensor) -> Dict[str, float]:
    mat = mat.detach()
    return {
        f"{prefix}.rms": float(torch.sqrt(mat.pow(2).mean()).cpu()),
        f"{prefix}.max_abs": float(mat.abs().max().cpu()),
    }


def _rule_calm_residuals(
    net: MLP,
    natural_net: MLP,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    z = _normal_rule_state(1, params=params, device=train_cfg.device, dtype=train_cfg.dtype)
    return rule_residuals(
        z,
        net(z),
        net,
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        policy=policy,
    )


def _rule_calm_residual_loss(
    net: MLP,
    natural_net: MLP,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> torch.Tensor:
    res, _ = _rule_calm_residuals(
        net,
        natural_net,
        nodes,
        policy=policy,
        params=params,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=fb_epsilon,
    )
    return residual_loss(stack_residuals(res), loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)


def _rule_calm_residual_diagnostics(
    net: MLP,
    natural_net: MLP,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> Dict[str, float]:
    res, _ = _rule_calm_residuals(
        net,
        natural_net,
        nodes,
        policy=policy,
        params=params,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=fb_epsilon,
    )
    return _residual_matrix_diagnostics("calm_residual", stack_residuals(res))


def _rule_calm_anchor_terms(
    net: MLP,
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> list[torch.Tensor]:
    z = _normal_rule_state(1, params=params, device=train_cfg.device, dtype=train_cfg.dtype)
    uses_natural_y_ref = abs(float(params.phi_y)) > 1e-14
    if uses_natural_y_ref or policy.lower() == "ba":
        out_n = natural_benchmark_outputs(
            z[..., :6],
            natural_net,
            params=params,
            need_rate=policy.lower() == "ba",
        )
    else:
        targets = steady_decode_targets(params)
        C_n = torch.full_like(z[..., 0], float(targets["C"]))
        Y_n = torch.full_like(z[..., 0], float(targets["Y"]))
        out_n = {"C_n": C_n, "Y_n": Y_n, "R_n_real": torch.full_like(Y_n, float(params.bar_R))}
    out = decode_rule_outputs(
        net(z),
        RULE_OUTPUT_NAMES,
        params=params,
        y_ref=out_n["Y_n"] if uses_natural_y_ref else None,
    )
    drv = derive_rule(
        unpack_rule_state(z),
        out,
        params,
        Y_n=out_n["Y_n"],
        R_n=out_n["R_n_real"],
        policy=policy,
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


def _rule_calm_anchor_diagnostics(
    net: MLP,
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    return _calm_anchor_diagnostics_from_terms(
        _rule_calm_anchor_terms(net, natural_net, policy=policy, params=params, train_cfg=train_cfg)
    )


def _rule_scenario_q_diagnostics(res: Dict[str, torch.Tensor], names: list[str]) -> Dict[str, float]:
    return _scenario_q_diagnostics(res, names, q_key="Q")


def _should_apply_scenario_loss(step: int | None, train_cfg: TrainConfig) -> bool:
    interval = max(1, int(train_cfg.rule_scenario_loss_interval))
    if step is None:
        return True
    step_i = int(step)
    return step_i == 1 or step_i % interval == 0


def _rule_auxiliary_training_loss(
    net: MLP,
    natural_net: MLP,
    nodes: QMCNodes,
    *,
    policy: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
    step: int | None = None,
) -> torch.Tensor:
    pieces = []
    q_weight = float(train_cfg.rule_scenario_q_weight)
    if q_weight > 0.0 and _should_apply_scenario_loss(step, train_cfg):
        scenario_res, _, _ = _rule_scenario_residuals(
            net,
            natural_net,
            nodes,
            policy=policy,
            params=params,
            qmc_cfg=qmc_cfg,
            train_cfg=train_cfg,
            fb_epsilon=fb_epsilon,
        )
        q_resid = scenario_res["Q"].reshape(-1, 1)
        pieces.append(q_weight * residual_loss(q_resid, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta))
    calm_weight = float(train_cfg.rule_calm_anchor_weight)
    if calm_weight > 0.0:
        pieces.append(calm_weight * _rule_calm_anchor_loss(net, natural_net, policy=policy, params=params, train_cfg=train_cfg))
    calm_resid_weight = float(train_cfg.rule_calm_residual_weight)
    if calm_resid_weight > 0.0:
        pieces.append(
            calm_resid_weight
            * _rule_calm_residual_loss(
                net,
                natural_net,
                nodes,
                policy=policy,
                params=params,
                qmc_cfg=qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=fb_epsilon,
            )
        )
    if not pieces:
        return torch.zeros((), device=train_cfg.device, dtype=train_cfg.dtype)
    return torch.stack(pieces).sum()


def _decode_optimal_for_kind(raw: torch.Tensor, kind: str, *, params: BaselineParams | None = None) -> Dict[str, torch.Tensor]:
    key = kind.lower()
    if key == "discretion":
        return decode_discretion(raw, params=params)
    if key == "commitment":
        return decode_commitment(raw, params=params)
    raise ValueError("kind must be 'discretion' or 'commitment'.")


def _normal_optimal_state(
    batch_size: int,
    *,
    kind: str,
    params: BaselineParams,
    device: str,
    dtype: torch.dtype,
    promise_init_scale: float,
) -> torch.Tensor:
    z_phys = _normal_rule_state(batch_size, params=params, device=device, dtype=dtype)
    if kind.lower() != "commitment":
        return z_phys
    mean, _ = commitment_promise_init_tensors(device=device, dtype=dtype, scale=promise_init_scale)
    promises = mean[None, :].expand(batch_size, len(COMMITMENT_PROMISE_NAMES))
    return torch.cat([z_phys, promises], dim=-1)


def _optimal_training_scenario_states(
    net: MLP,
    nodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, list[str]]:
    """Deterministic no-event/crisis states for optimal-policy value diagnostics."""

    key = kind.lower()
    device = train_cfg.device
    dtype = train_cfg.dtype
    burnin = max(1, int(train_cfg.rule_scenario_burnin))
    horizon = max(5, int(train_cfg.rule_scenario_horizon))
    total = burnin + horizon + 1
    z = _normal_optimal_state(
        len(_RULE_SCENARIO_LABELS),
        kind=key,
        params=params,
        device=device,
        dtype=dtype,
        promise_init_scale=train_cfg.promise_init_scale,
    )
    states = [z]
    with torch.no_grad():
        for t in range(1, total):
            out = _decode_optimal_for_kind(net(z), key, params=params)
            _, drv = private_residuals_free(
                z,
                out,
                net,
                nodes,
                params=params,
                qmc_cfg=qmc_cfg,
                fb_epsilon=train_cfg.fb_epsilon_final,
                commitment=key == "commitment",
            )
            add_D, add_X = _rule_scenario_additions(
                t=t,
                pulse=burnin,
                relief_lag=4,
                params=params,
                device=z.device,
                dtype=z.dtype,
            )
            z_phys = _deterministic_rule_step(
                z[..., :7],
                A_next=drv["A_next"],
                Delta_next=drv["Delta"],
                add_D=add_D,
                add_X=add_X,
                params=params,
            )
            if key == "commitment":
                promises = torch.stack([out[name] for name in COMMITMENT_PROMISE_NAMES], dim=-1)
                z = torch.cat([z_phys, promises], dim=-1)
            else:
                z = z_phys
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


def _optimal_scenario_residuals(
    net: MLP,
    nodes: QMCNodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list[str]]:
    key = kind.lower()
    z, names = _optimal_training_scenario_states(
        net,
        nodes,
        kind=key,
        params=params,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
    )
    out = _decode_optimal_for_kind(net(z), key, params=params)
    res, drv = private_residuals_free(
        z,
        out,
        net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=key == "commitment",
    )
    return res, drv, names


def _scenario_q_diagnostics(res: Dict[str, torch.Tensor], names: list[str], *, q_key: str) -> Dict[str, float]:
    if q_key not in res:
        return {}
    q = res[q_key].detach().reshape(-1)
    diag: Dict[str, float] = {
        "scenario_Q.rms": float(torch.sqrt(q.pow(2).mean()).cpu()),
        "scenario_Q.max_abs": float(q.abs().max().cpu()),
    }
    wanted = {f"{label}.{tag}" for label, tag in _RULE_SCENARIO_POINTS}
    for i, name in enumerate(names):
        if name in wanted and i < q.numel():
            diag[f"scenario_Q.{name}"] = float(q[i].abs().cpu())
    return diag


def _scenario_mechanism_diagnostics(
    drv: Dict[str, torch.Tensor],
    names: list[str],
    *,
    params: BaselineParams,
) -> Dict[str, float]:
    if not adaptation_enabled(params) or "Q_A" not in drv:
        return {}
    q = drv["Q_A"].detach().reshape(-1)
    diag: Dict[str, float] = {
        "scenario_Q_A.mean": float(q.mean().cpu()),
        "scenario_Q_A.min": float(q.min().cpu()),
        "scenario_Q_A.max": float(q.max().cpu()),
    }
    if {"Omega_A", "p_a"}.issubset(drv):
        threshold = (
            drv["Omega_A"].detach().reshape(-1)
            * drv["p_a"].detach().reshape(-1)
            * float(params.psi_A)
        ).clamp_min(1e-12)
        activation = q / threshold
        diag.update(
            {
                "scenario_repair_threshold.mean": float(threshold.mean().cpu()),
                "scenario_repair_activation.mean": float(activation.mean().cpu()),
                "scenario_repair_activation.min": float(activation.min().cpu()),
                "scenario_repair_activation.max": float(activation.max().cpu()),
            }
        )
    else:
        threshold = None
        activation = None
    if "I_A" in drv:
        repair = drv["I_A"].detach().reshape(-1)
        diag.update(
            {
                "scenario_I_A.mean": float(repair.mean().cpu()),
                "scenario_I_A.max": float(repair.max().cpu()),
                "scenario_repair_positive.freq": float((repair > 1e-5).to(repair.dtype).mean().cpu()),
            }
        )
    if {"M_zero_rent", "mbar"}.issubset(drv):
        pressure = drv["M_zero_rent"].detach().reshape(-1) / torch.clamp(
            drv["mbar"].detach().reshape(-1),
            min=1e-12,
        )
        diag.update(
            {
                "scenario_cap_pressure.mean": float(pressure.mean().cpu()),
                "scenario_cap_pressure.max": float(pressure.max().cpu()),
            }
        )
    wanted = {f"{label}.{tag}" for label, tag in _RULE_SCENARIO_POINTS}
    for i, name in enumerate(names):
        if name not in wanted or i >= q.numel():
            continue
        diag[f"scenario_Q_A.{name}"] = float(q[i].cpu())
        if activation is not None and i < activation.numel():
            diag[f"scenario_repair_activation.{name}"] = float(activation[i].cpu())
        if "I_A" in drv:
            repair = drv["I_A"].detach().reshape(-1)
            if i < repair.numel():
                diag[f"scenario_I_A.{name}"] = float(repair[i].cpu())
    return diag


def _optimal_calm_anchor_loss(
    net: MLP,
    *,
    kind: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    return _calm_anchor_loss_from_terms(
        _optimal_calm_anchor_terms(net, kind=kind, params=params, train_cfg=train_cfg)
    )


def _optimal_calm_anchor_terms(
    net: MLP,
    *,
    kind: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> list[torch.Tensor]:
    z = _normal_optimal_state(
        1,
        kind=kind,
        params=params,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
        promise_init_scale=train_cfg.promise_init_scale,
    )
    out = _decode_optimal_for_kind(net(z), kind, params=params)
    drv = derive_free(unpack_rule_state(z[..., :7]), out, params, R=torch.full_like(out["C"], float(params.bar_R)))
    pressure = drv["M_zero_rent"] / torch.clamp(drv["mbar"], min=1e-12)
    target_pressure = torch.full_like(pressure, 1.0 / (1.0 + float(params.normal_capacity_slack)))
    repair_scale = max(float(params.repair_capacity), 1e-6)
    y_target = torch.full_like(out["Y"], float(params.steady_state_output))
    s_target, f_target = _pricing_sum_targets(y_target, params)
    return [
        torch.log(torch.clamp(out["Y"] / float(params.steady_state_output), min=1e-12)),
        torch.log(torch.clamp(out["Pi"] / float(params.bar_pi), min=1e-12)),
        torch.log(torch.clamp(out["S_p"] / torch.clamp(s_target, min=1e-12), min=1e-12)),
        torch.log(torch.clamp(out["F_p"] / torch.clamp(f_target, min=1e-12), min=1e-12)),
        pressure - target_pressure,
        drv["chi"] / torch.clamp(drv["pm"], min=1e-12),
        drv["I_A"] / repair_scale,
    ]


def _optimal_calm_anchor_diagnostics(
    net: MLP,
    *,
    kind: str,
    params: BaselineParams,
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    return _calm_anchor_diagnostics_from_terms(
        _optimal_calm_anchor_terms(net, kind=kind, params=params, train_cfg=train_cfg)
    )


def _optimal_calm_residuals(
    net: MLP,
    nodes: QMCNodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    key = kind.lower()
    z = _normal_optimal_state(
        1,
        kind=key,
        params=params,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
        promise_init_scale=train_cfg.promise_init_scale,
    )
    out = _decode_optimal_for_kind(net(z), key, params=params)
    return private_residuals_free(
        z,
        out,
        net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=fb_epsilon,
        commitment=key == "commitment",
    )


def _optimal_calm_residual_loss(
    net: MLP,
    nodes: QMCNodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> torch.Tensor:
    res, _ = _optimal_calm_residuals(
        net,
        nodes,
        kind=kind,
        params=params,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=fb_epsilon,
    )
    return residual_loss(stack_residuals(res), loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)


def _optimal_calm_residual_diagnostics(
    net: MLP,
    nodes: QMCNodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
) -> Dict[str, float]:
    res, _ = _optimal_calm_residuals(
        net,
        nodes,
        kind=kind,
        params=params,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=fb_epsilon,
    )
    return _residual_matrix_diagnostics("calm_residual", stack_residuals(res))


def _optimal_auxiliary_training_loss(
    net: MLP,
    nodes: QMCNodes,
    *,
    kind: str,
    params: BaselineParams,
    qmc_cfg: QMCConfig,
    train_cfg: TrainConfig,
    fb_epsilon: float,
    step: int | None = None,
) -> torch.Tensor:
    pieces = []
    q_weight = float(train_cfg.rule_scenario_q_weight)
    if q_weight > 0.0 and _should_apply_scenario_loss(step, train_cfg):
        scenario_res, _, _ = _optimal_scenario_residuals(
            net,
            nodes,
            kind=kind,
            params=params,
            qmc_cfg=qmc_cfg,
            train_cfg=train_cfg,
            fb_epsilon=fb_epsilon,
        )
        q_resid = scenario_res["Q"].reshape(-1, 1)
        pieces.append(q_weight * residual_loss(q_resid, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta))
    calm_weight = float(train_cfg.rule_calm_anchor_weight)
    if calm_weight > 0.0:
        pieces.append(calm_weight * _optimal_calm_anchor_loss(net, kind=kind, params=params, train_cfg=train_cfg))
    calm_resid_weight = float(train_cfg.rule_calm_residual_weight)
    if calm_resid_weight > 0.0:
        pieces.append(
            calm_resid_weight
            * _optimal_calm_residual_loss(
                net,
                nodes,
                kind=kind,
                params=params,
                qmc_cfg=qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=fb_epsilon,
            )
        )
    if not pieces:
        return torch.zeros((), device=train_cfg.device, dtype=train_cfg.dtype)
    return torch.stack(pieces).sum()


def residual_diagnostics(residuals: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Return equation-level residual diagnostics.

    DEQN is only as credible as its residual checks.  These diagnostics make it
    visible which equilibrium block is currently limiting the approximation.
    """

    diag: Dict[str, float] = {}
    with torch.no_grad():
        mats = []
        for name, value in residuals.items():
            v = value.detach()
            mats.append(v.reshape(-1))
            diag[f"{name}.rms"] = float(torch.sqrt(v.pow(2).mean()).cpu())
            diag[f"{name}.mean_abs"] = float(v.abs().mean().cpu())
            diag[f"{name}.max_abs"] = float(v.abs().max().cpu())
        if mats:
            mat = torch.cat(mats)
            diag["overall.rms"] = float(torch.sqrt(mat.pow(2).mean()).cpu())
            diag["overall.max_abs"] = float(mat.abs().max().cpu())
    return diag


def _add_tensor_diagnostics(diag: Dict[str, float], prefix: str, value: torch.Tensor) -> None:
    v = value.detach().reshape(-1)
    diag[f"{prefix}.rms"] = float(torch.sqrt(v.pow(2).mean()).cpu())
    diag[f"{prefix}.mean_abs"] = float(v.abs().mean().cpu())
    diag[f"{prefix}.max_abs"] = float(v.abs().max().cpu())
    diag[f"{prefix}.min"] = float(v.min().cpu())
    diag[f"{prefix}.max"] = float(v.max().cpu())


def exact_condition_diagnostics(data: Dict[str, torch.Tensor], params: BaselineParams, *, natural: bool = False) -> Dict[str, float]:
    """Diagnostics for original, un-smoothed complementarity conditions.

    Training can use normalized and smoothed residuals for conditioning.  These
    checks report the original gaps and products, so the saved eval JSON reveals
    whether smoothing creates economically meaningful complementarity leakage.
    """

    diag: Dict[str, float] = {}
    with torch.no_grad():
        chi_name = "chi_n" if natural else "chi"
        if chi_name in data and "mbar" in data and "M" in data:
            chi = data[chi_name]
            cap_gap = data["mbar"] - data["M"]
            cap_gap_rel = cap_gap / torch.clamp(data["mbar"], min=1e-12)
            pressure_source = data.get("M_zero_rent", data["M"])
            cap_pressure = pressure_source / torch.clamp(data["mbar"], min=1e-12)
            _add_tensor_diagnostics(diag, "exact_cap_gap", cap_gap)
            _add_tensor_diagnostics(diag, "exact_cap_gap_rel", cap_gap_rel)
            _add_tensor_diagnostics(diag, "exact_cap_pressure_ratio", cap_pressure)
            if "M_zero_rent" in data and "M_at_rent" in data:
                cap_target = torch.where(data["M_zero_rent"] > data["mbar"], data["mbar"], data["M_zero_rent"])
                cap_solve_error = data["M_at_rent"] - cap_target
                _add_tensor_diagnostics(diag, "exact_cap_solve_error", cap_solve_error)
                _add_tensor_diagnostics(
                    diag,
                    "exact_cap_solve_error_rel",
                    cap_solve_error / torch.clamp(data["mbar"], min=1e-12),
                )
            _add_tensor_diagnostics(diag, "exact_cap_product", chi * cap_gap)
            if "pm" in data:
                cap_rent_scaled = chi / torch.clamp(data["pm"], min=1e-12)
                _add_tensor_diagnostics(diag, "exact_cap_product_scaled", cap_rent_scaled * cap_gap_rel)
            _add_tensor_diagnostics(diag, "exact_cap_chi_negative", torch.relu(-chi))
            _add_tensor_diagnostics(diag, "exact_cap_gap_negative", torch.relu(-cap_gap))

        if "euler_rate_residual" in data:
            _add_tensor_diagnostics(diag, "exact_euler_rate_residual", data["euler_rate_residual"])

        if not natural and adaptation_enabled(params) and {"I_A", "Q_A", "Omega_A", "p_a"}.issubset(data):
            I = data.get("I_A_effective", data["I_A"])
            repair_gap = data["Omega_A"] * data["p_a"] * psi_prime(I, params) - data["Q_A"]
            repair_gap_scaled = repair_gap / torch.clamp(data["Omega_A"] * data["p_a"], min=1e-12)
            repair_activation = data["Q_A"] / torch.clamp(
                data["Omega_A"] * data["p_a"] * float(params.psi_A), min=1e-12
            )
            eta = 1.0 / torch.clamp(data["Omega_A"] * data["p_a"] * float(params.phi_A), min=1e-12)
            projected = torch.clamp(I - eta * repair_gap, min=0.0, max=float(params.repair_capacity))
            repair_proj = I - projected
            lower = (I <= 1e-6).to(I.dtype)
            upper = (I >= float(params.repair_capacity) - 1e-6).to(I.dtype)
            interior = 1.0 - torch.clamp(lower + upper, max=1.0)
            _add_tensor_diagnostics(diag, "exact_repair_gap", repair_gap)
            _add_tensor_diagnostics(diag, "exact_repair_gap_scaled", repair_gap_scaled)
            _add_tensor_diagnostics(diag, "exact_repair_activation_ratio", repair_activation)
            _add_tensor_diagnostics(diag, "exact_repair_projection", repair_proj)
            _add_tensor_diagnostics(diag, "exact_repair_lower_violation", lower * torch.relu(-repair_gap_scaled))
            _add_tensor_diagnostics(diag, "exact_repair_interior_violation", interior * repair_gap_scaled.abs())
            _add_tensor_diagnostics(diag, "exact_repair_upper_violation", upper * torch.relu(repair_gap_scaled))
            _add_tensor_diagnostics(diag, "exact_repair_I_negative", torch.relu(-I))
            _add_tensor_diagnostics(diag, "exact_repair_capacity_excess", torch.relu(I - float(params.repair_capacity)))
    return diag


def save_checkpoint(
    path: str | Path,
    net: nn.Module,
    *,
    metadata: Dict[str, object] | None = None,
) -> None:
    """Save a model checkpoint without imposing a training-workflow format."""

    payload = {"state_dict": net.state_dict(), "metadata": metadata or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _adapt_legacy_policy_output_state_dict(state_dict: Dict[str, torch.Tensor], net: nn.Module) -> Dict[str, torch.Tensor]:
    """Map recent reduced-output checkpoints into the current independent-S_p heads."""

    target = net.state_dict()
    weight_keys = [key for key, value in target.items() if key.endswith(".weight") and value.ndim == 2]
    if not weight_keys:
        return state_dict
    weight_key = weight_keys[-1]
    bias_key = weight_key[:-6] + "bias"
    if weight_key not in state_dict or bias_key not in state_dict or bias_key not in target:
        return state_dict

    src_w = state_dict[weight_key]
    src_b = state_dict[bias_key]
    tgt_w = target[weight_key]
    tgt_b = target[bias_key]
    if src_w.ndim != 2 or src_w.shape[1] != tgt_w.shape[1] or src_b.ndim != 1:
        return state_dict
    if src_w.shape[0] == tgt_w.shape[0] and src_b.shape[0] == tgt_b.shape[0]:
        return state_dict

    target_rows = int(tgt_w.shape[0])
    source_rows = int(src_w.shape[0])
    mapping: list[tuple[int, int]] | None = None
    if target_rows == len(RULE_OUTPUT_NAMES) and source_rows == target_rows - 1:
        # Reduced Taylor head: C,Y,Pi,Q_A,F_p -> C,Y,Pi,Q_A,S_p,F_p.
        mapping = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)]
    elif target_rows == len(DISCRETION_OUTPUT_NAMES) and source_rows == target_rows - 2:
        # Reduced discretion head: no S_p control and no mu_price_index.
        mapping = [
            (0, 0),  # C
            (1, 1),  # Y
            (2, 2),  # Pi
            (3, 3),  # Q_A
            (4, 5),  # F_p
            (5, 6),  # V
            (6, 7),  # mu_resource
            (7, 9),  # mu_calvo_S
            (8, 10),  # mu_calvo_F
            (9, 11),  # mu_Q
        ]
    elif target_rows == len(COMMITMENT_OUTPUT_NAMES) and source_rows == target_rows - 2:
        # Reduced commitment head: no S_p control and no mu_price_index.
        mapping = [
            (0, 0),  # C
            (1, 1),  # Y
            (2, 2),  # Pi
            (3, 3),  # Q_A
            (4, 5),  # F_p
            (5, 6),  # mu_resource
            (6, 8),  # mu_calvo_S
            (7, 9),  # mu_calvo_F
            (8, 10),  # mu_Q
            (9, 11),  # promise_S
            (10, 12),  # promise_F
            (11, 13),  # promise_Q
        ]
    if mapping is None:
        return state_dict

    new_w = tgt_w.detach().clone()
    new_b = tgt_b.detach().clone()
    for src_idx, tgt_idx in mapping:
        new_w[tgt_idx].copy_(src_w[src_idx].to(device=new_w.device, dtype=new_w.dtype))
        new_b[tgt_idx].copy_(src_b[src_idx].to(device=new_b.device, dtype=new_b.dtype))
    migrated = dict(state_dict)
    migrated[weight_key] = new_w
    migrated[bias_key] = new_b
    return migrated


def load_model_state_dict(net: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load a checkpoint state dict, including known architecture migrations."""

    net.load_state_dict(_adapt_legacy_policy_output_state_dict(state_dict, net))


def _cpu_detached(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_detached(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_detached(v) for v in value)
    return value


def _maybe_save_training_state(
    *,
    step: int,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    extra: Dict[str, object] | None = None,
) -> None:
    if cfg.checkpoint_dir is None or int(cfg.checkpoint_every) <= 0:
        return
    if int(step) % int(cfg.checkpoint_every) != 0:
        return
    directory = Path(cfg.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = str(cfg.checkpoint_name)
    path = directory / f"{name}_step_{int(step):08d}.pt"
    payload = {
        "step": int(step),
        "state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": _cpu_detached(extra or {}),
    }
    torch.save(payload, path)

    keep = int(cfg.checkpoint_keep)
    if keep > 0:
        checkpoints = sorted(directory.glob(f"{name}_step_*.pt"))
        for old in checkpoints[:-keep]:
            old.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, net: nn.Module, *, map_location: str | torch.device = "cpu") -> Dict[str, object]:
    """Load a checkpoint into an already constructed network."""

    payload = torch.load(Path(path), map_location=map_location)
    load_model_state_dict(net, payload["state_dict"])
    metadata = payload.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def train_natural(
    *,
    params: BaselineParams = BaselineParams(),
    net_cfg: NetworkConfig = NetworkConfig(),
    qmc_cfg: QMCConfig = QMCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    steps: int | None = None,
    log_every: int = 100,
) -> tuple[MLP, TrainLog]:
    """Train the auxiliary flexible-price benchmark network."""

    device = train_cfg.device
    dtype = train_cfg.dtype
    net = make_natural_net(net_cfg, device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=float(train_cfg.lr))
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=device, dtype=dtype)
    val_nodes = _validation_nodes(qmc_cfg, device=device, dtype=dtype, seed_offset=10_001)
    val_qmc_cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + 10_001)
    val_z = sample_rule_states(
        train_cfg.stop_val_states,
        params=params,
        device=device,
        dtype=dtype,
    )
    val_z_n = natural_from_rule_states(val_z)
    n_steps = int(train_cfg.steps if steps is None else steps)
    log = TrainLog()
    stop_hits = 0
    best_state = None

    _announce_training(kind="natural", total=n_steps, train_cfg=train_cfg, qmc_cfg=qmc_cfg, log_every=log_every)
    progress = _progress_range(n_steps, desc="natural", enabled=train_cfg.show_progress)
    for step in progress:
        z = sample_rule_states(train_cfg.batch_size, params=params, device=device, dtype=dtype)
        z_n = natural_from_rule_states(z)
        raw = net(z_n)
        res, _ = natural_residuals(
            z_n,
            raw,
            net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=train_cfg.fb_epsilon_start,
        )
        mat = stack_residuals(res)
        loss = residual_loss(mat, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        should_validate = step == 1 or step % int(log_every) == 0 or step == n_steps
        if should_validate:
            with torch.no_grad():
                val_res, _ = natural_residuals(
                    val_z_n,
                    net(val_z_n),
                    net,
                    val_nodes,
                    params=params,
                    qmc_cfg=val_qmc_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                )
                val_mat = stack_residuals(val_res).detach()
            metrics = _log_metrics(step, mat.detach(), log, loss.detach(), val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            best_state = _maybe_update_best_state(
                net,
                log,
                metrics,
                step,
                best_state,
                train_cfg,
                extra={"kind": "natural"},
            )
            _maybe_save_training_state(
                step=step,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "natural"},
            )
            if _passes_stop_criteria(val_mat, train_cfg, step, metrics):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, step, train_cfg)
                    _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
                    break
            else:
                stop_hits = 0
            _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
        elif _should_update_train_postfix(step, log_every):
            _report_train_postfix(
                progress,
                mat.detach(),
                loss.detach(),
                step=step,
                stop_hits=stop_hits,
                enabled=train_cfg.show_progress,
            )
    _restore_best_state(net, best_state)
    return net, log


def train_rule(
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams = BaselineParams(),
    net_cfg: NetworkConfig = NetworkConfig(),
    qmc_cfg: QMCConfig = QMCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    steps: int | None = None,
    log_every: int = 100,
) -> tuple[MLP, TrainLog]:
    """Train a rule-based Taylor DEQN policy network."""

    if policy.lower() not in {"fixed", "ba", "bottleneck", "repair_aware"}:
        raise ValueError("policy must be 'fixed', 'ba', 'bottleneck', or 'repair_aware'.")
    freeze(natural_net)
    device = train_cfg.device
    dtype = train_cfg.dtype
    net = make_rule_net(net_cfg, device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=float(train_cfg.lr))
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=device, dtype=dtype)
    val_nodes = _validation_nodes(qmc_cfg, device=device, dtype=dtype, seed_offset=10_101)
    val_qmc_cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + 10_101)
    val_z = sample_rule_states(
        train_cfg.stop_val_states,
        params=params,
        device=device,
        dtype=dtype,
    )
    n_steps = int(train_cfg.steps if steps is None else steps)
    log = TrainLog()
    stop_hits = 0
    best_state = None

    _announce_training(kind=f"rule-{policy.lower()}", total=n_steps, train_cfg=train_cfg, qmc_cfg=qmc_cfg, log_every=log_every)
    progress = _progress_range(n_steps, desc=f"rule-{policy.lower()}", enabled=train_cfg.show_progress)
    for step in progress:
        z = sample_rule_states(train_cfg.batch_size, params=params, device=device, dtype=dtype)
        raw = net(z)
        res, _ = rule_residuals(
            z,
            raw,
            net,
            natural_net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=train_cfg.fb_epsilon_start,
            policy=policy,
        )
        mat, loss = _rule_residual_loss(res, train_cfg)
        loss = loss + _rule_auxiliary_training_loss(
            net,
            natural_net,
            nodes,
            policy=policy,
            params=params,
            qmc_cfg=qmc_cfg,
            train_cfg=train_cfg,
            fb_epsilon=train_cfg.fb_epsilon_start,
            step=step,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        should_validate = step == 1 or step % int(log_every) == 0 or step == n_steps
        if should_validate:
            with torch.no_grad():
                val_res, _ = rule_residuals(
                    val_z,
                    net(val_z),
                    net,
                    natural_net,
                    val_nodes,
                    params=params,
                    qmc_cfg=val_qmc_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                    policy=policy,
                )
                val_mat = stack_residuals(val_res).detach()
                scenario_val_res, scenario_val_drv, scenario_names = _rule_scenario_residuals(
                    net,
                    natural_net,
                    val_nodes,
                    policy=policy,
                    params=params,
                    qmc_cfg=val_qmc_cfg,
                    train_cfg=train_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                )
            metrics = _log_metrics(step, mat.detach(), log, loss.detach(), val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            scenario_diag = _rule_scenario_q_diagnostics(scenario_val_res, scenario_names)
            mechanism_diag = _scenario_mechanism_diagnostics(scenario_val_drv, scenario_names, params=params)
            metrics.update(scenario_diag)
            metrics.update(mechanism_diag)
            calm_diag = _rule_calm_anchor_diagnostics(
                net,
                natural_net,
                policy=policy,
                params=params,
                train_cfg=train_cfg,
            )
            metrics.update(calm_diag)
            calm_resid_diag = _rule_calm_residual_diagnostics(
                net,
                natural_net,
                val_nodes,
                policy=policy,
                params=params,
                qmc_cfg=val_qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=train_cfg.fb_epsilon_final,
            )
            metrics.update(calm_resid_diag)
            best_state = _maybe_update_best_state(
                net,
                log,
                metrics,
                step,
                best_state,
                train_cfg,
                extra={"kind": "rule", "policy": policy.lower()},
            )
            log.extra_metrics.append(
                {
                    "step": float(step),
                    **scenario_diag,
                    **mechanism_diag,
                    **calm_diag,
                    **calm_resid_diag,
                    "selection_score": float(metrics.get("selection_score", float("nan"))),
                }
            )
            _maybe_save_training_state(
                step=step,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "rule", "policy": policy.lower()},
            )
            if _passes_stop_criteria(val_mat, train_cfg, step, metrics):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, step, train_cfg)
                    _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
                    break
            else:
                stop_hits = 0
            _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
        elif _should_update_train_postfix(step, log_every):
            _report_train_postfix(
                progress,
                mat.detach(),
                loss.detach(),
                step=step,
                stop_hits=stop_hits,
                enabled=train_cfg.show_progress,
            )
    _restore_best_state(net, best_state)
    return net, log


def train_rule_episode(
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams = BaselineParams(),
    net_cfg: NetworkConfig = NetworkConfig(),
    qmc_cfg: QMCConfig = QMCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    episodes: int | None = None,
    log_every: int = 10,
) -> tuple[MLP, TrainLog]:
    """Train a rule network on simulated DEQN episodes.

    This is the closest PyTorch analogue to the Keras author code: each
    episode first simulates state trajectories under the current network, then
    residual minimization is applied to mini-batches from those simulated
    states.  Expectations inside residuals are still computed with fixed QMC
    nodes.
    """

    if policy.lower() not in {"fixed", "ba", "bottleneck", "repair_aware"}:
        raise ValueError("policy must be 'fixed', 'ba', 'bottleneck', or 'repair_aware'.")
    freeze(natural_net)
    device = train_cfg.device
    dtype = train_cfg.dtype
    net = make_rule_net(net_cfg, device=device, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=float(train_cfg.lr))
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=device, dtype=dtype)
    val_nodes = _validation_nodes(qmc_cfg, device=device, dtype=dtype, seed_offset=10_201)
    val_qmc_cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + 10_201)
    val_z = sample_rule_states(
        train_cfg.stop_val_states,
        params=params,
        device=device,
        dtype=dtype,
    )
    n_episodes = int(train_cfg.steps if episodes is None else episodes)
    log = TrainLog()
    stop_hits = 0
    best_state = None

    current_state = sample_rule_states(
        train_cfg.sim_batch_size,
        params=params,
        device=device,
        dtype=dtype,
    )
    _announce_training(
        kind=f"rule-{policy.lower()}-episode",
        total=n_episodes,
        train_cfg=train_cfg,
        qmc_cfg=qmc_cfg,
        log_every=log_every,
    )
    progress = _progress_range(n_episodes, desc=f"rule-{policy.lower()}", enabled=train_cfg.show_progress)
    for episode in progress:
        state_episode = simulate_rule_episode(
            current_state,
            net,
            natural_net,
            policy=policy,
            params=params,
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
                pieces.append(sample_rule_states(broad_n, params=params, device=device, dtype=dtype))
            z = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
            raw = net(z)
            res, _ = rule_residuals(
                z,
                raw,
                net,
                natural_net,
                nodes,
                params=params,
                qmc_cfg=qmc_cfg,
                fb_epsilon=train_cfg.fb_epsilon_start,
                policy=policy,
            )
            mat, loss = _rule_residual_loss(res, train_cfg)
            loss = loss + _rule_auxiliary_training_loss(
                net,
                natural_net,
                nodes,
                policy=policy,
                params=params,
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
        should_validate = episode == 1 or episode % int(log_every) == 0 or episode == n_episodes
        if last_mat is not None and last_loss is not None and should_validate:
            with torch.no_grad():
                val_res, _ = rule_residuals(
                    val_z,
                    net(val_z),
                    net,
                    natural_net,
                    val_nodes,
                    params=params,
                    qmc_cfg=val_qmc_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                    policy=policy,
                )
                val_mat = stack_residuals(val_res).detach()
                scenario_val_res, scenario_val_drv, scenario_names = _rule_scenario_residuals(
                    net,
                    natural_net,
                    val_nodes,
                    policy=policy,
                    params=params,
                    qmc_cfg=val_qmc_cfg,
                    train_cfg=train_cfg,
                    fb_epsilon=train_cfg.fb_epsilon_final,
                )
            metrics = _log_metrics(episode, last_mat, log, last_loss, val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            scenario_diag = _rule_scenario_q_diagnostics(scenario_val_res, scenario_names)
            mechanism_diag = _scenario_mechanism_diagnostics(scenario_val_drv, scenario_names, params=params)
            metrics.update(scenario_diag)
            metrics.update(mechanism_diag)
            calm_diag = _rule_calm_anchor_diagnostics(
                net,
                natural_net,
                policy=policy,
                params=params,
                train_cfg=train_cfg,
            )
            metrics.update(calm_diag)
            calm_resid_diag = _rule_calm_residual_diagnostics(
                net,
                natural_net,
                val_nodes,
                policy=policy,
                params=params,
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
                extra={"kind": "rule", "policy": policy.lower(), "current_state": current_state},
            )
            log.extra_metrics.append(
                {
                    "step": float(episode),
                    **scenario_diag,
                    **mechanism_diag,
                    **calm_diag,
                    **calm_resid_diag,
                    "selection_score": float(metrics.get("selection_score", float("nan"))),
                }
            )
            _maybe_save_training_state(
                step=episode,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "rule", "policy": policy.lower(), "current_state": current_state},
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
            _report_progress(progress, metrics, step=episode, total=n_episodes, stop_hits=stop_hits, enabled=train_cfg.show_progress)
        elif (
            last_mat is not None
            and last_loss is not None
            and _should_update_train_postfix(episode, log_every)
        ):
            _report_train_postfix(
                progress,
                last_mat,
                last_loss,
                step=episode,
                stop_hits=stop_hits,
                enabled=train_cfg.show_progress,
            )
    _restore_best_state(net, best_state)
    return net, log


def _initial_optimal_states(
    n: int,
    *,
    kind: str,
    params: BaselineParams,
    device: str,
    dtype: torch.dtype,
    promise_init_scale: float = 1.0,
) -> torch.Tensor:
    z = sample_rule_states(n, params=params, device=device, dtype=dtype)
    if kind == "commitment":
        mean, std = commitment_promise_init_tensors(device=z.device, dtype=z.dtype, scale=promise_init_scale)
        promises = mean[None, :] + std[None, :] * torch.randn(
            (z.shape[0], len(COMMITMENT_PROMISE_NAMES)),
            device=z.device,
            dtype=z.dtype,
        )
        z = torch.cat([z, promises], dim=-1)
    return z


def train_optimal_episode(
    *,
    kind: str,
    params: BaselineParams = BaselineParams(),
    net_cfg: NetworkConfig = NetworkConfig(),
    qmc_cfg: QMCConfig = QMCConfig(),
    train_cfg: TrainConfig = TrainConfig(),
    episodes: int | None = None,
    log_every: int = 10,
) -> tuple[MLP, TrainLog]:
    key = kind.lower()
    if key == "discretion":
        net = make_discretion_net(net_cfg, device=train_cfg.device, dtype=train_cfg.dtype)
        residual_fn = discretion_residuals
    elif key == "commitment":
        net = make_commitment_net(net_cfg, device=train_cfg.device, dtype=train_cfg.dtype)
        initialize_commitment_promises(net, scale=train_cfg.promise_init_scale)
        residual_fn = commitment_residuals
    else:
        raise ValueError("kind must be 'discretion' or 'commitment'.")

    opt = torch.optim.Adam(net.parameters(), lr=float(train_cfg.lr))
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=train_cfg.device, dtype=train_cfg.dtype)
    val_nodes = _validation_nodes(qmc_cfg, device=train_cfg.device, dtype=train_cfg.dtype, seed_offset=10_301)
    val_qmc_cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + 10_301)
    n_episodes = int(train_cfg.steps if episodes is None else episodes)
    log = TrainLog()
    stop_hits = 0
    best_state = None
    current_state = _initial_optimal_states(
        train_cfg.sim_batch_size,
        kind=key,
        params=params,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
        promise_init_scale=train_cfg.promise_init_scale,
    )
    val_state = _initial_optimal_states(
        train_cfg.stop_val_states,
        kind=key,
        params=params,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
        promise_init_scale=train_cfg.promise_init_scale,
    )

    _announce_training(kind=key, total=n_episodes, train_cfg=train_cfg, qmc_cfg=qmc_cfg, log_every=log_every)
    progress = _progress_range(n_episodes, desc=key, enabled=train_cfg.show_progress)
    for episode in progress:
        state_episode = simulate_optimal_episode(
            current_state,
            net,
            kind=key,
            params=params,
            length=train_cfg.episode_length,
            nodes=nodes,
            qmc_cfg=qmc_cfg,
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
            stage = _optimal_training_stage(episode, train_cfg)
            pieces = []
            if episode_n > 0:
                idx = torch.randint(flat_states.shape[0], (episode_n,), device=flat_states.device)
                pieces.append(flat_states[idx])
            if broad_n > 0:
                pieces.append(
                    _initial_optimal_states(
                        broad_n,
                        kind=key,
                        params=params,
                        device=train_cfg.device,
                        dtype=train_cfg.dtype,
                        promise_init_scale=train_cfg.promise_init_scale,
                    )
                )
            z = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
            raw = net(z)
            res, _ = _optimal_training_residuals_for_stage(
                z,
                raw,
                net,
                nodes,
                kind=key,
                params=params,
                qmc_cfg=qmc_cfg,
                fb_epsilon=train_cfg.fb_epsilon_start,
                stage=stage,
            )
            if not all(bool(torch.isfinite(value).all().detach().cpu()) for value in res.values()):
                bad = _nonfinite_residual_summary(res)
                raise FloatingPointError(
                    f"Non-finite {key} residuals at episode {episode} stage={stage}: {bad}"
                )
            obj_mat = _optimal_objective_matrix(res, train_cfg, stage=stage, step=episode)
            loss = residual_loss(obj_mat, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)
            loss = loss + _optimal_auxiliary_training_loss(
                net,
                nodes,
                kind=key,
                params=params,
                qmc_cfg=qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=train_cfg.fb_epsilon_start,
                step=episode,
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                bad = _nonfinite_residual_summary(res)
                raise FloatingPointError(
                    f"Non-finite {key} loss at episode {episode} stage={stage}: {bad}"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            if not bool(torch.isfinite(grad_norm).detach().cpu()):
                opt.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite {key} gradient at episode {episode} stage={stage}: "
                    f"grad_norm={float(grad_norm.detach().cpu())}"
                )
            opt.step()
            last_mat = obj_mat.detach()
            last_loss = loss.detach()
        should_validate = episode == 1 or episode % int(log_every) == 0 or episode == n_episodes
        if last_mat is not None and last_loss is not None and should_validate:
            val_raw = net(val_state)
            val_res, _ = residual_fn(
                val_state,
                val_raw,
                net,
                val_nodes,
                params=params,
                qmc_cfg=val_qmc_cfg,
                fb_epsilon=train_cfg.fb_epsilon_final,
            )
            val_mat = stack_residuals(val_res).detach()
            val_top = _top_residual_summary(val_res)
            del val_raw, val_res
            scenario_val_res, scenario_val_drv, scenario_names = _optimal_scenario_residuals(
                net,
                val_nodes,
                kind=key,
                params=params,
                qmc_cfg=val_qmc_cfg,
                train_cfg=train_cfg,
                fb_epsilon=train_cfg.fb_epsilon_final,
            )
            metrics = _log_metrics(episode, last_mat, log, last_loss, val_mat)
            metrics["stage"] = _optimal_training_stage(episode, train_cfg)
            metrics["full_weight"] = _optimal_full_weight(episode, train_cfg) if metrics["stage"] == "full" else 0.0
            metrics["val_top"] = val_top
            scenario_diag = _scenario_q_diagnostics(scenario_val_res, scenario_names, q_key="Q")
            mechanism_diag = _scenario_mechanism_diagnostics(scenario_val_drv, scenario_names, params=params)
            del scenario_val_res, scenario_val_drv, scenario_names
            metrics.update(scenario_diag)
            metrics.update(mechanism_diag)
            calm_diag = _optimal_calm_anchor_diagnostics(
                net,
                kind=key,
                params=params,
                train_cfg=train_cfg,
            )
            metrics.update(calm_diag)
            calm_resid_diag = _optimal_calm_residual_diagnostics(
                net,
                val_nodes,
                kind=key,
                params=params,
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
                extra={"kind": key, "current_state": current_state},
            )
            log.extra_metrics.append(
                {
                    "step": float(episode),
                    "stage": metrics["stage"],
                    "full_weight": metrics["full_weight"],
                    **scenario_diag,
                    **mechanism_diag,
                    **calm_diag,
                    **calm_resid_diag,
                    "selection_score": float(metrics.get("selection_score", float("nan"))),
                }
            )
            _maybe_save_training_state(
                step=episode,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": key, "current_state": current_state},
            )
            if _passes_stop_criteria(val_mat, train_cfg, episode, metrics):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, episode, train_cfg)
                    _report_progress(progress, metrics, step=episode, total=n_episodes, stop_hits=stop_hits, enabled=train_cfg.show_progress)
                    break
            else:
                stop_hits = 0
            _report_progress(progress, metrics, step=episode, total=n_episodes, stop_hits=stop_hits, enabled=train_cfg.show_progress)
            if str(train_cfg.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif (
            last_mat is not None
            and last_loss is not None
            and _should_update_train_postfix(episode, log_every)
        ):
            _report_train_postfix(
                progress,
                last_mat,
                last_loss,
                step=episode,
                stop_hits=stop_hits,
                enabled=train_cfg.show_progress,
                stage=_optimal_training_stage(episode, train_cfg),
            )
    _restore_best_state(net, best_state)
    return net, log


def evaluate_natural(
    net: MLP,
    *,
    params: BaselineParams = BaselineParams(),
    qmc_cfg: QMCConfig = QMCConfig(n_train=4096),
    train_cfg: TrainConfig = TrainConfig(),
    n_states: int = 4096,
) -> Dict[str, float]:
    device = train_cfg.device
    dtype = train_cfg.dtype
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=device, dtype=dtype)
    z = sample_rule_states(n_states, params=params, device=device, dtype=dtype, seed=1234)
    z_n = natural_from_rule_states(z)
    with torch.no_grad():
        res, drv = natural_residuals(
            z_n,
            net(z_n),
            net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=train_cfg.fb_epsilon_final,
        )
        mat = stack_residuals(res)
        return {
            "loss": float(mat.pow(2).mean().cpu()),
            "rms": float(torch.sqrt(mat.pow(2).mean()).cpu()),
            "max_abs": float(mat.abs().max().cpu()),
            **residual_diagnostics(res),
            **exact_condition_diagnostics(drv, params, natural=True),
        }


def evaluate_rule(
    net: MLP,
    natural_net: MLP,
    *,
    policy: str,
    params: BaselineParams = BaselineParams(),
    qmc_cfg: QMCConfig = QMCConfig(n_train=4096),
    train_cfg: TrainConfig = TrainConfig(),
    n_states: int = 4096,
) -> Dict[str, float]:
    device = train_cfg.device
    dtype = train_cfg.dtype
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=device, dtype=dtype)
    z = sample_rule_states(n_states, params=params, device=device, dtype=dtype, seed=4321)
    with torch.no_grad():
        res, drv = rule_residuals(
            z,
            net(z),
            net,
            natural_net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=train_cfg.fb_epsilon_final,
            policy=policy,
        )
        mat = stack_residuals(res)
        scenario_res, scenario_drv, scenario_names = _rule_scenario_residuals(
            net,
            natural_net,
            nodes,
            policy=policy,
            params=params,
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
            **_rule_scenario_q_diagnostics(scenario_res, scenario_names),
            **_scenario_mechanism_diagnostics(scenario_drv, scenario_names, params=params),
        }


def evaluate_optimal(
    net: MLP,
    *,
    kind: str,
    params: BaselineParams = BaselineParams(),
    qmc_cfg: QMCConfig = QMCConfig(n_train=4096),
    train_cfg: TrainConfig = TrainConfig(),
    n_states: int = 4096,
) -> Dict[str, float]:
    key = kind.lower()
    if key == "discretion":
        residual_fn = discretion_residuals
    elif key == "commitment":
        residual_fn = commitment_residuals
    else:
        raise ValueError("kind must be 'discretion' or 'commitment'.")

    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=train_cfg.device, dtype=train_cfg.dtype)
    z = _initial_optimal_states(
        n_states,
        kind=key,
        params=params,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
        promise_init_scale=train_cfg.promise_init_scale,
    )
    raw = net(z)
    res, drv = residual_fn(
        z,
        raw,
        net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=train_cfg.fb_epsilon_final,
    )
    mat = stack_residuals(res)
    scenario_res, scenario_drv, scenario_names = _optimal_scenario_residuals(
        net,
        nodes,
        kind=key,
        params=params,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        fb_epsilon=train_cfg.fb_epsilon_final,
    )
    return {
        "loss": float(mat.pow(2).mean().detach().cpu()),
        "rms": float(torch.sqrt(mat.pow(2).mean()).detach().cpu()),
        "max_abs": float(mat.abs().max().detach().cpu()),
        **residual_diagnostics(res),
        **exact_condition_diagnostics(drv, params),
        **_scenario_q_diagnostics(scenario_res, scenario_names, q_key="Q"),
        **_scenario_mechanism_diagnostics(scenario_drv, scenario_names, params=params),
    }
