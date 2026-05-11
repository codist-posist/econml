from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn

from .config import (
    BaselineParams,
    COMMITMENT_PROMISE_INIT_MEAN,
    COMMITMENT_PROMISE_INIT_STD,
    COMMITMENT_OUTPUT_NAMES,
    DISCRETION_OUTPUT_NAMES,
    NATURAL_OUTPUT_NAMES,
    RULE_OUTPUT_NAMES,
    NetworkConfig,
    QMCConfig,
    TrainConfig,
)
from .networks import MLP
from .qmc import make_qmc_nodes
from .residuals import natural_residuals, rule_residuals, stack_residuals
from .sampling import natural_from_rule_states, sample_rule_states
from .episode import simulate_rule_episode
from .economics import adaptation_enabled, psi_prime
from .optimal import commitment_residuals, discretion_residuals, simulate_optimal_episode


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


def make_natural_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    """Build the auxiliary flexible-price benchmark network."""

    return MLP(6, len(NATURAL_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def make_rule_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    """Build a rule-based DEQN network."""

    return MLP(7, len(RULE_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def make_discretion_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    return MLP(7, len(DISCRETION_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


def make_commitment_net(net_cfg: NetworkConfig = NetworkConfig(), *, device: str = "cpu", dtype: torch.dtype = torch.float64) -> MLP:
    return MLP(11, len(COMMITMENT_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)


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
    start = len(COMMITMENT_OUTPUT_NAMES) - 4
    mean, _ = commitment_promise_init_tensors(device=last.bias.device, dtype=last.bias.dtype, scale=scale)
    with torch.no_grad():
        last.bias[start : start + 4].copy_(mean)


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


def _passes_stop_criteria(resid: torch.Tensor, cfg: TrainConfig, step: int) -> bool:
    if cfg.target_rms is None and cfg.target_max_abs is None:
        return False
    if step < int(cfg.min_steps_before_stop):
        return False
    with torch.no_grad():
        rms = float(torch.sqrt(resid.pow(2).mean()).detach().cpu())
        max_abs = float(resid.abs().max().detach().cpu())
    rms_ok = True if cfg.target_rms is None else rms <= float(cfg.target_rms)
    max_ok = True if cfg.target_max_abs is None else max_abs <= float(cfg.target_max_abs)
    return rms_ok and max_ok


def _mark_stopped(log: TrainLog, step: int, cfg: TrainConfig) -> None:
    log.stopped_early = True
    log.stop_reason = (
        f"validation residual criteria satisfied for {int(cfg.early_stop_patience)} consecutive checks "
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
    payload = {
        "train_rms": f"{metrics['train_rms']:.2e}",
        "val_rms": f"{metrics.get('val_rms', float('nan')):.2e}",
        "val_max": f"{metrics.get('val_max_abs', float('nan')):.2e}",
        "stop": int(stop_hits),
    }
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(payload)
    message = (
        f"[{step}/{total}] train_rms={payload['train_rms']} "
        f"val_rms={payload['val_rms']} val_max={payload['val_max']} stop_hits={payload['stop']}"
    )
    val_top = metrics.get("val_top")
    if val_top:
        message += f" top_val={val_top}"
    if metrics.get("new_best"):
        message += " best=*"
    print(message, flush=True)


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
    print(
        f"Starting {kind}: total={int(total)}, batch_size={int(train_cfg.batch_size)}, "
        f"sim_batch_size={int(train_cfg.sim_batch_size)}, episode_length={int(train_cfg.episode_length)}, "
        f"updates_per_episode={int(train_cfg.episode_updates_per_episode)}, "
        f"broad_share={float(train_cfg.episode_broad_share):.2f}, "
        f"qmc_train={int(qmc_cfg.n_train)}, qmc_val={int(qmc_cfg.n_val)}, "
        f"stop_val_states={int(train_cfg.stop_val_states)}, log_every={int(log_every)}, "
        f"device={train_cfg.device}, dtype={train_cfg.dtype}",
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


def _copy_state_dict_to_cpu(net: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def _maybe_update_best_state(
    net: nn.Module,
    log: TrainLog,
    metrics: dict[str, float],
    step: int,
    best_state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    """Track the best validation checkpoint by val_rms, breaking ties by val_max."""

    if "val_rms" not in metrics:
        return best_state
    val_rms = float(metrics["val_rms"])
    val_max = float(metrics.get("val_max_abs", float("inf")))
    train_rms = float(metrics.get("train_rms", float("nan")))
    is_better = log.best_val_rms is None
    if log.best_val_rms is not None:
        tol = 1e-12
        is_better = val_rms < float(log.best_val_rms) - tol or (
            abs(val_rms - float(log.best_val_rms)) <= tol
            and (log.best_val_max_abs is None or val_max < float(log.best_val_max_abs))
        )
    if not is_better:
        metrics["new_best"] = False
        return best_state

    log.best_step = int(step)
    log.best_val_rms = val_rms
    log.best_val_max_abs = val_max
    log.best_train_rms = train_rms
    metrics["new_best"] = True
    return _copy_state_dict_to_cpu(net)


def _restore_best_state(net: nn.Module, best_state: dict[str, torch.Tensor] | None) -> None:
    if best_state is not None:
        net.load_state_dict(best_state)


def _validation_nodes(qmc_cfg: QMCConfig, *, device: str, dtype: torch.dtype, seed_offset: int) -> QMCNodes:
    cfg = replace(qmc_cfg, n_train=qmc_cfg.n_val, seed=int(qmc_cfg.seed) + int(seed_offset))
    return make_qmc_nodes(cfg.n_train, cfg=cfg, device=device, dtype=dtype)


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
    net.load_state_dict(payload["state_dict"])
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
        if step == 1 or step % int(log_every) == 0 or step == n_steps:
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
            best_state = _maybe_update_best_state(net, log, metrics, step, best_state)
            _maybe_save_training_state(
                step=step,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "natural"},
            )
            if _passes_stop_criteria(val_mat, train_cfg, step):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, step, train_cfg)
                    _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
                    break
            else:
                stop_hits = 0
            _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
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
    """Train a fixed-Taylor or bottleneck-adjusted Taylor DEQN policy network."""

    if policy.lower() not in {"fixed", "ba"}:
        raise ValueError("policy must be 'fixed' or 'ba'.")
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
        mat = stack_residuals(res)
        loss = residual_loss(mat, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        if step == 1 or step % int(log_every) == 0 or step == n_steps:
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
            metrics = _log_metrics(step, mat.detach(), log, loss.detach(), val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            best_state = _maybe_update_best_state(net, log, metrics, step, best_state)
            _maybe_save_training_state(
                step=step,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "rule", "policy": policy.lower()},
            )
            if _passes_stop_criteria(val_mat, train_cfg, step):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, step, train_cfg)
                    _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
                    break
            else:
                stop_hits = 0
            _report_progress(progress, metrics, step=step, total=n_steps, stop_hits=stop_hits, enabled=train_cfg.show_progress)
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

    if policy.lower() not in {"fixed", "ba"}:
        raise ValueError("policy must be 'fixed' or 'ba'.")
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
            mat = stack_residuals(res)
            loss = residual_loss(mat, loss=train_cfg.loss, huber_delta=train_cfg.huber_delta)
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
            metrics = _log_metrics(episode, last_mat, log, last_loss, val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            best_state = _maybe_update_best_state(net, log, metrics, episode, best_state)
            _maybe_save_training_state(
                step=episode,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": "rule", "policy": policy.lower(), "current_state": current_state},
            )
            if _passes_stop_criteria(val_mat, train_cfg, episode):
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
        promises = mean[None, :] + std[None, :] * torch.randn((z.shape[0], 4), device=z.device, dtype=z.dtype)
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
            res, _ = residual_fn(
                z,
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
            last_mat = mat.detach()
            last_loss = loss.detach()
        if last_mat is not None and last_loss is not None and (
            episode == 1 or episode % int(log_every) == 0 or episode == n_episodes
        ):
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
            metrics = _log_metrics(episode, last_mat, log, last_loss, val_mat)
            metrics["val_top"] = _top_residual_summary(val_res)
            best_state = _maybe_update_best_state(net, log, metrics, episode, best_state)
            _maybe_save_training_state(
                step=episode,
                net=net,
                optimizer=opt,
                cfg=train_cfg,
                extra={"kind": key, "current_state": current_state},
            )
            if _passes_stop_criteria(val_mat, train_cfg, episode):
                stop_hits += 1
                if stop_hits >= int(train_cfg.early_stop_patience):
                    _mark_stopped(log, episode, train_cfg)
                    _report_progress(progress, metrics, step=episode, total=n_episodes, stop_hits=stop_hits, enabled=train_cfg.show_progress)
                    break
            else:
                stop_hits = 0
            _report_progress(progress, metrics, step=episode, total=n_episodes, stop_hits=stop_hits, enabled=train_cfg.show_progress)
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
        return {
            "loss": float(mat.pow(2).mean().cpu()),
            "rms": float(torch.sqrt(mat.pow(2).mean()).cpu()),
            "max_abs": float(mat.abs().max().cpu()),
            **residual_diagnostics(res),
            **exact_condition_diagnostics(drv, params),
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
    return {
        "loss": float(mat.pow(2).mean().detach().cpu()),
        "rms": float(torch.sqrt(mat.pow(2).mean()).detach().cpu()),
        "max_abs": float(mat.abs().max().detach().cpu()),
        **residual_diagnostics(res),
        **exact_condition_diagnostics(drv, params),
    }
