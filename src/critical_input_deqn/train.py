from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn

from .config import (
    BaselineParams,
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
from .optimal import commitment_residuals, discretion_residuals, simulate_optimal_episode


@dataclass
class TrainLog:
    steps: list[int]
    losses: list[float]
    max_abs: list[float]


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


def _log_metrics(step: int, resid: torch.Tensor, log: TrainLog, objective: torch.Tensor | None = None) -> None:
    with torch.no_grad():
        log.steps.append(int(step))
        logged_loss = resid.pow(2).mean() if objective is None else objective
        log.losses.append(float(logged_loss.detach().cpu()))
        log.max_abs.append(float(resid.abs().max().detach().cpu()))


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
    n_steps = int(train_cfg.steps if steps is None else steps)
    log = TrainLog([], [], [])

    for step in range(1, n_steps + 1):
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
            _log_metrics(step, mat, log, loss)
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
    n_steps = int(train_cfg.steps if steps is None else steps)
    log = TrainLog([], [], [])

    for step in range(1, n_steps + 1):
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
            _log_metrics(step, mat, log, loss)
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
    n_episodes = int(train_cfg.steps if episodes is None else episodes)
    log = TrainLog([], [], [])

    current_state = sample_rule_states(
        train_cfg.sim_batch_size,
        params=params,
        device=device,
        dtype=dtype,
    )
    for episode in range(1, n_episodes + 1):
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
        order = torch.randperm(flat_states.shape[0], device=flat_states.device)
        last_mat = None
        for start in range(0, flat_states.shape[0], int(train_cfg.batch_size)):
            idx = order[start : start + int(train_cfg.batch_size)]
            if idx.numel() == 0:
                continue
            z = flat_states[idx]
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
        if last_mat is not None and (episode == 1 or episode % int(log_every) == 0 or episode == n_episodes):
            _log_metrics(episode, last_mat, log, loss)
    return net, log


def _initial_optimal_states(
    n: int,
    *,
    kind: str,
    params: BaselineParams,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = sample_rule_states(n, params=params, device=device, dtype=dtype)
    if kind == "commitment":
        z = torch.cat([z, torch.zeros((z.shape[0], 4), device=z.device, dtype=z.dtype)], dim=-1)
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
        residual_fn = commitment_residuals
    else:
        raise ValueError("kind must be 'discretion' or 'commitment'.")

    opt = torch.optim.Adam(net.parameters(), lr=float(train_cfg.lr))
    nodes = make_qmc_nodes(qmc_cfg.n_train, cfg=qmc_cfg, device=train_cfg.device, dtype=train_cfg.dtype)
    n_episodes = int(train_cfg.steps if episodes is None else episodes)
    log = TrainLog([], [], [])
    current_state = _initial_optimal_states(
        train_cfg.sim_batch_size,
        kind=key,
        params=params,
        device=train_cfg.device,
        dtype=train_cfg.dtype,
    )

    for episode in range(1, n_episodes + 1):
        state_episode = simulate_optimal_episode(
            current_state,
            net,
            kind=key,
            params=params,
            length=train_cfg.episode_length,
        )
        current_state = state_episode[-1].detach()
        flat_states = state_episode.reshape(-1, state_episode.shape[-1]).detach()
        order = torch.randperm(flat_states.shape[0], device=flat_states.device)
        last_mat = None
        last_loss = None
        for start in range(0, flat_states.shape[0], int(train_cfg.batch_size)):
            idx = order[start : start + int(train_cfg.batch_size)]
            if idx.numel() == 0:
                continue
            z = flat_states[idx]
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
            _log_metrics(episode, last_mat, log, last_loss)
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
        res, _ = natural_residuals(
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
        res, _ = rule_residuals(
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
    )
    raw = net(z)
    res, _ = residual_fn(
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
    }
