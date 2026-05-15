from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from .config import (
    COMMITMENT_PROMISE_INIT_MEAN,
    COMMITMENT_PROMISE_INIT_STD,
    NetworkConfig,
    QMCConfig,
    TrainConfig,
    stop_profile,
)
from .experiments import resolve_params
from .train import evaluate_optimal, save_checkpoint, train_optimal_episode


def _dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("--dtype must be float64 or float32.")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _selection_metadata(log) -> dict[str, object]:
    return {
        "criterion": log.best_selection_criterion or "min_val_rms_then_val_max_abs",
        "best_step": log.best_step,
        "best_selection_score": log.best_selection_score,
        "best_val_rms": log.best_val_rms,
        "best_val_max_abs": log.best_val_max_abs,
        "best_train_rms": log.best_train_rms,
        "best_scenario_q_rms": log.best_scenario_q_rms,
        "best_q_nobubble_rms": log.best_q_nobubble_rms,
        "best_calm_anchor_rms": log.best_calm_anchor_rms,
        "best_calm_residual_rms": log.best_calm_residual_rms,
    }


def _resolved_stop(args: argparse.Namespace) -> dict[str, float | int | None]:
    defaults = {} if args.no_auto_stop else stop_profile(args.kind)
    return {
        "target_rms": args.target_rms if args.target_rms is not None else defaults.get("target_rms"),
        "target_max_abs": args.target_max_abs if args.target_max_abs is not None else defaults.get("target_max_abs"),
        "early_stop_patience": (
            args.early_stop_patience
            if args.early_stop_patience is not None
            else defaults.get("early_stop_patience", 5)
        ),
        "min_steps_before_stop": (
            args.min_steps_before_stop
            if args.min_steps_before_stop is not None
            else defaults.get("min_steps_before_stop", 0)
        ),
    }


def _argument_supplied(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _apply_commitment_safe_defaults(args: argparse.Namespace) -> None:
    if args.kind != "commitment":
        return
    if not _argument_supplied("--lr"):
        args.lr = 5e-5
    if not _argument_supplied("--feasibility-pretrain-steps"):
        args.feasibility_pretrain_steps = 1_500
    if not _argument_supplied("--full-weight-warmup-steps"):
        args.full_weight_warmup_steps = 2_500


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optimal-policy DEQN networks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("discretion", "commitment"), required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--full-batch-size",
        type=int,
        default=256,
        help=(
            "Row microbatch size for full optimal-policy FOC/envelope residuals. "
            "The outer batch-size still controls sampling; this only limits CUDA memory after feasibility pretraining."
        ),
    )
    parser.add_argument("--sim-batch-size", type=int, default=512)
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument(
        "--episode-updates-per-episode",
        type=int,
        default=2,
        help="Number of gradient mini-batch updates after each simulated episode.",
    )
    parser.add_argument(
        "--episode-broad-share",
        type=float,
        default=0.50,
        help="Share of each episode-training mini-batch drawn from broad sampled states.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", default="huber", choices=("huber", "mse"))
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--target-rms", type=float, default=None)
    parser.add_argument("--target-max-abs", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--min-steps-before-stop", type=int, default=None)
    parser.add_argument("--no-auto-stop", action="store_true", help="Disable policy-specific validation stopping defaults.")
    parser.add_argument("--stop-val-states", type=int, default=512)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument(
        "--scenario-q-weight",
        type=float,
        default=25.0,
        help="Extra weight on private Q_A recursion residuals at deterministic no-event/crisis scenario states.",
    )
    parser.add_argument(
        "--calm-anchor-weight",
        type=float,
        default=5.0,
        help="Extra weight on the calm steady-branch anchor for optimal-policy training.",
    )
    parser.add_argument(
        "--calm-residual-weight",
        type=float,
        default=5.0,
        help="Extra weight on full private residuals at the calm state for optimal-policy training.",
    )
    parser.add_argument("--scenario-burnin", type=int, default=5)
    parser.add_argument("--scenario-horizon", type=int, default=10)
    parser.add_argument(
        "--scenario-loss-interval",
        type=int,
        default=25,
        help="Apply the expensive scenario-Q training loss every K optimal-policy episodes.",
    )
    parser.add_argument("--best-scenario-q-weight", type=float, default=1.0)
    parser.add_argument(
        "--q-nobubble-weight",
        type=float,
        default=0.0,
        help=(
            "Experimental extra weight on finite-horizon no-bubble Q_A present-value residuals "
            "at scenario states. Default is zero: report qPV diagnostics without training on them."
        ),
    )
    parser.add_argument(
        "--q-nobubble-horizon",
        type=int,
        default=12,
        help="Finite path horizon for the no-bubble Q_A present-value residual.",
    )
    parser.add_argument(
        "--q-nobubble-paths",
        type=int,
        default=64,
        help="QMC paths used by the no-bubble Q_A present-value residual.",
    )
    parser.add_argument("--best-q-nobubble-weight", type=float, default=0.0)
    parser.add_argument("--best-calm-anchor-weight", type=float, default=1.0)
    parser.add_argument("--best-calm-residual-weight", type=float, default=1.0)
    parser.add_argument("--target-scenario-q-rms", type=float, default=1e-2)
    parser.add_argument(
        "--feasibility-pretrain-steps",
        type=int,
        default=1_000,
        help="Initial optimal-policy episodes that train only private feasibility residuals.",
    )
    parser.add_argument(
        "--full-weight-warmup-steps",
        type=int,
        default=1_000,
        help="Episodes over which stationarity/promise residual weights ramp in after feasibility pretraining.",
    )
    parser.add_argument(
        "--private-loss-weight",
        type=float,
        default=1.0,
        help="Training-objective weight on private implementability residuals.",
    )
    parser.add_argument(
        "--bellman-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Legacy weight on an explicit Bellman residual. The current optimal-policy "
            "system is FOC/envelope based and has no discretion V residual by default."
        ),
    )
    parser.add_argument(
        "--stationarity-loss-weight",
        type=float,
        default=0.10,
        help="Training-objective weight on optimal-policy stationarity residuals after feasibility pretraining.",
    )
    parser.add_argument(
        "--envelope-loss-weight",
        type=float,
        default=0.10,
        help="Training-objective weight on explicit discretion envelope-costate residuals after feasibility pretraining.",
    )
    parser.add_argument(
        "--promise-loss-weight",
        type=float,
        default=1.0,
        help="Training-objective weight on commitment promise residuals after feasibility pretraining.",
    )
    parser.add_argument(
        "--promise-init-scale",
        type=float,
        default=1.0,
        help="Scale factor on the author-style commitment promise initialization standard deviations.",
    )
    parser.add_argument("--qmc-train", type=int, default=256)
    parser.add_argument("--qmc-val", type=int, default=512)
    parser.add_argument("--n-val-states", type=int, default=1024)
    parser.add_argument("--hidden-width", type=int, default=192)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    _apply_commitment_safe_defaults(args)

    params, experiment_meta = resolve_params(args.experiment, args.params_json)
    dtype = _dtype(args.dtype)
    net_cfg = NetworkConfig(hidden_width=args.hidden_width, hidden_depth=args.hidden_depth)
    qmc_cfg = QMCConfig(n_train=args.qmc_train, n_val=args.qmc_val, seed=args.seed)
    stop = _resolved_stop(args)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        optimal_full_batch_size=args.full_batch_size,
        sim_batch_size=args.sim_batch_size,
        episode_length=args.episode_length,
        episode_updates_per_episode=args.episode_updates_per_episode,
        episode_broad_share=args.episode_broad_share,
        lr=args.lr,
        steps=args.steps,
        loss=args.loss,
        huber_delta=args.huber_delta,
        target_rms=stop["target_rms"],
        target_max_abs=stop["target_max_abs"],
        early_stop_patience=int(stop["early_stop_patience"]),
        min_steps_before_stop=int(stop["min_steps_before_stop"]),
        stop_val_states=args.stop_val_states,
        show_progress=not args.no_progress,
        promise_init_scale=args.promise_init_scale,
        checkpoint_dir=None if args.no_checkpoints else str(args.output_dir / "checkpoints"),
        checkpoint_name=args.kind,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        rule_scenario_q_weight=args.scenario_q_weight,
        rule_calm_anchor_weight=args.calm_anchor_weight,
        rule_calm_residual_weight=args.calm_residual_weight,
        rule_scenario_burnin=args.scenario_burnin,
        rule_scenario_horizon=args.scenario_horizon,
        rule_scenario_loss_interval=args.scenario_loss_interval,
        best_scenario_q_weight=args.best_scenario_q_weight,
        optimal_q_nobubble_weight=args.q_nobubble_weight,
        optimal_q_nobubble_horizon=args.q_nobubble_horizon,
        optimal_q_nobubble_paths=args.q_nobubble_paths,
        best_q_nobubble_weight=args.best_q_nobubble_weight,
        best_calm_anchor_weight=args.best_calm_anchor_weight,
        best_calm_residual_weight=args.best_calm_residual_weight,
        target_scenario_q_rms=args.target_scenario_q_rms,
        optimal_feasibility_pretrain_steps=args.feasibility_pretrain_steps,
        optimal_private_loss_weight=args.private_loss_weight,
        optimal_bellman_loss_weight=args.bellman_loss_weight,
        optimal_stationarity_loss_weight=args.stationarity_loss_weight,
        optimal_envelope_loss_weight=args.envelope_loss_weight,
        optimal_promise_loss_weight=args.promise_loss_weight,
        optimal_full_weight_warmup_steps=args.full_weight_warmup_steps,
        dtype=dtype,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "kind": args.kind,
        "experiment": experiment_meta,
        "params": experiment_meta["params"],
        "network": asdict(net_cfg),
        "qmc": asdict(qmc_cfg),
        "train": {
            "batch_size": args.batch_size,
            "full_batch_size": args.full_batch_size,
            "sim_batch_size": args.sim_batch_size,
            "episode_length": args.episode_length,
            "episode_updates_per_episode": args.episode_updates_per_episode,
            "episode_broad_share": args.episode_broad_share,
            "lr": args.lr,
            "steps": args.steps,
            "loss": args.loss,
            "huber_delta": args.huber_delta,
            "stop": stop,
            "target_rms_arg": args.target_rms,
            "target_max_abs_arg": args.target_max_abs,
            "early_stop_patience_arg": args.early_stop_patience,
            "min_steps_before_stop_arg": args.min_steps_before_stop,
            "no_auto_stop": args.no_auto_stop,
            "stop_val_states": args.stop_val_states,
            "show_progress": not args.no_progress,
            "promise_init_scale": args.promise_init_scale,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_keep": args.checkpoint_keep,
            "no_checkpoints": args.no_checkpoints,
            "scenario_q_weight": args.scenario_q_weight,
            "calm_anchor_weight": args.calm_anchor_weight,
            "calm_residual_weight": args.calm_residual_weight,
            "scenario_burnin": args.scenario_burnin,
            "scenario_horizon": args.scenario_horizon,
            "scenario_loss_interval": args.scenario_loss_interval,
            "best_scenario_q_weight": args.best_scenario_q_weight,
            "q_nobubble_weight": args.q_nobubble_weight,
            "q_nobubble_horizon": args.q_nobubble_horizon,
            "q_nobubble_paths": args.q_nobubble_paths,
            "best_q_nobubble_weight": args.best_q_nobubble_weight,
            "best_calm_anchor_weight": args.best_calm_anchor_weight,
            "best_calm_residual_weight": args.best_calm_residual_weight,
            "target_scenario_q_rms": args.target_scenario_q_rms,
            "feasibility_pretrain_steps": args.feasibility_pretrain_steps,
            "full_weight_warmup_steps": args.full_weight_warmup_steps,
            "private_loss_weight": args.private_loss_weight,
            "bellman_loss_weight": args.bellman_loss_weight,
            "stationarity_loss_weight": args.stationarity_loss_weight,
            "envelope_loss_weight": args.envelope_loss_weight,
            "promise_loss_weight": args.promise_loss_weight,
            "commitment_promise_init_mean": COMMITMENT_PROMISE_INIT_MEAN,
            "commitment_promise_init_std": COMMITMENT_PROMISE_INIT_STD,
            "dtype": args.dtype,
            "device": args.device,
        },
    }
    _write_json(args.output_dir / "run_config.json", run_config)
    print(
        f"Configured run_optimal: output_dir={args.output_dir}, kind={args.kind}, "
        f"device={args.device}, dtype={args.dtype}, steps={args.steps}, "
        f"batch_size={args.batch_size}, full_batch_size={args.full_batch_size}, "
        f"qmc_train={args.qmc_train}, qmc_val={args.qmc_val}, "
        f"updates_per_episode={args.episode_updates_per_episode}, broad_share={args.episode_broad_share}, "
        f"feasibility_pretrain={args.feasibility_pretrain_steps}, "
        f"full_warmup={args.full_weight_warmup_steps}, "
        f"stat_w={args.stationarity_loss_weight:g}, env_w={args.envelope_loss_weight:g}, "
        f"legacy_bellman_w={args.bellman_loss_weight:g}, "
        f"q_nobubble_w={args.q_nobubble_weight:g}",
        flush=True,
    )
    net, log = train_optimal_episode(
        kind=args.kind,
        params=params,
        net_cfg=net_cfg,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        episodes=args.steps,
        log_every=args.log_every,
    )
    save_checkpoint(
        args.output_dir / f"{args.kind}.pt",
        net,
        metadata={"kind": args.kind, "config": run_config, "selection": _selection_metadata(log)},
    )
    _write_json(args.output_dir / f"{args.kind}_train_log.json", asdict(log))
    eval_metrics = evaluate_optimal(
        net,
        kind=args.kind,
        params=params,
        qmc_cfg=QMCConfig(n_train=args.qmc_val, seed=args.seed + 303),
        train_cfg=train_cfg,
        n_states=args.n_val_states,
    )
    _write_json(args.output_dir / f"{args.kind}_eval.json", eval_metrics)


if __name__ == "__main__":
    main()
