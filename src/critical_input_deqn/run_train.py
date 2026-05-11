from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import NetworkConfig, QMCConfig, TrainConfig, stop_profile
from .experiments import resolve_params
from .train import (
    evaluate_natural,
    evaluate_rule,
    load_checkpoint,
    make_natural_net,
    save_checkpoint,
    train_natural,
    train_rule,
    train_rule_episode,
)


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
        "criterion": "min_val_rms_then_val_max_abs",
        "best_step": log.best_step,
        "best_val_rms": log.best_val_rms,
        "best_val_max_abs": log.best_val_max_abs,
        "best_train_rms": log.best_train_rms,
    }


def _policies(raw: str) -> list[str]:
    policies = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = {"fixed", "ba"}
    bad = sorted(set(policies) - valid)
    if bad:
        raise ValueError(f"Unknown policy names: {bad}. Use fixed, ba, or fixed,ba.")
    return policies


def _resolved_stop(args: argparse.Namespace, kind: str) -> dict[str, float | int | None]:
    defaults = {} if args.no_auto_stop else stop_profile(kind)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the critical-input baseline DEQN.")
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--policies", default="fixed,ba", help="Comma-separated list: fixed,ba")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--natural-checkpoint", type=Path, default=None)
    parser.add_argument("--natural-steps", type=int, default=20_000)
    parser.add_argument("--rule-steps", type=int, default=8_000)
    parser.add_argument("--rule-trainer", default="episode", choices=("episode", "iid"))
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
    parser.add_argument("--batch-size", type=int, default=2048)
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
    parser.add_argument(
        "--rule-scenario-q-weight",
        type=float,
        default=25.0,
        help="Extra weight on Q_A recursion residuals at deterministic no-event/crisis scenario states.",
    )
    parser.add_argument(
        "--rule-calm-anchor-weight",
        type=float,
        default=5.0,
        help="Extra weight on the calm steady-branch anchor for rule-policy training.",
    )
    parser.add_argument("--rule-scenario-burnin", type=int, default=5)
    parser.add_argument("--rule-scenario-horizon", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    params, experiment_meta = resolve_params(args.experiment, args.params_json)
    dtype = _dtype(args.dtype)
    net_cfg = NetworkConfig(hidden_width=args.hidden_width, hidden_depth=args.hidden_depth)
    qmc_cfg = QMCConfig(n_train=args.qmc_train, n_val=args.qmc_val, seed=args.seed)
    natural_stop = _resolved_stop(args, "natural")
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.natural_steps,
        loss=args.loss,
        huber_delta=args.huber_delta,
        target_rms=natural_stop["target_rms"],
        target_max_abs=natural_stop["target_max_abs"],
        early_stop_patience=int(natural_stop["early_stop_patience"]),
        min_steps_before_stop=int(natural_stop["min_steps_before_stop"]),
        stop_val_states=args.stop_val_states,
        show_progress=not args.no_progress,
        checkpoint_dir=None if args.no_checkpoints else str(args.output_dir / "checkpoints"),
        checkpoint_name="natural",
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        dtype=dtype,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment": experiment_meta,
        "params": experiment_meta["params"],
        "network": asdict(net_cfg),
        "qmc": asdict(qmc_cfg),
        "train": {
            "batch_size": args.batch_size,
            "sim_batch_size": args.sim_batch_size,
            "episode_length": args.episode_length,
            "episode_updates_per_episode": args.episode_updates_per_episode,
            "episode_broad_share": args.episode_broad_share,
            "rule_scenario_q_weight": args.rule_scenario_q_weight,
            "rule_calm_anchor_weight": args.rule_calm_anchor_weight,
            "rule_scenario_burnin": args.rule_scenario_burnin,
            "rule_scenario_horizon": args.rule_scenario_horizon,
            "lr": args.lr,
            "natural_steps": args.natural_steps,
            "rule_steps": args.rule_steps,
            "rule_trainer": args.rule_trainer,
            "loss": args.loss,
            "huber_delta": args.huber_delta,
            "natural_stop": natural_stop,
            "target_rms_arg": args.target_rms,
            "target_max_abs_arg": args.target_max_abs,
            "early_stop_patience_arg": args.early_stop_patience,
            "min_steps_before_stop_arg": args.min_steps_before_stop,
            "no_auto_stop": args.no_auto_stop,
            "stop_val_states": args.stop_val_states,
            "show_progress": not args.no_progress,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_keep": args.checkpoint_keep,
            "no_checkpoints": args.no_checkpoints,
            "dtype": args.dtype,
            "device": args.device,
        },
        "policies": _policies(args.policies),
        "policy_stop": {policy: _resolved_stop(args, policy) for policy in _policies(args.policies)},
    }
    _write_json(args.output_dir / "run_config.json", run_config)
    print(
        f"Configured run_train: output_dir={args.output_dir}, policies={_policies(args.policies)}, "
        f"device={args.device}, dtype={args.dtype}, natural_steps={args.natural_steps}, "
        f"rule_steps={args.rule_steps}, qmc_train={args.qmc_train}, qmc_val={args.qmc_val}, "
        f"updates_per_episode={args.episode_updates_per_episode}, broad_share={args.episode_broad_share}",
        flush=True,
    )

    if args.natural_checkpoint is None:
        print("Training auxiliary flexible-price benchmark network.", flush=True)
        natural_net, natural_log = train_natural(
            net_cfg=net_cfg,
            qmc_cfg=qmc_cfg,
            train_cfg=train_cfg,
            params=params,
            log_every=args.log_every,
        )
        save_checkpoint(
            args.output_dir / "natural.pt",
            natural_net,
            metadata={"kind": "natural", "config": run_config, "selection": _selection_metadata(natural_log)},
        )
        _write_json(args.output_dir / "natural_train_log.json", asdict(natural_log))
    else:
        print(f"Loading auxiliary flexible-price benchmark network from {args.natural_checkpoint}.", flush=True)
        natural_net = make_natural_net(net_cfg, device=args.device, dtype=dtype)
        load_checkpoint(args.natural_checkpoint, natural_net, map_location=args.device)

    print("Evaluating auxiliary flexible-price benchmark network.", flush=True)
    natural_eval = evaluate_natural(
        natural_net,
        params=params,
        qmc_cfg=QMCConfig(n_train=args.qmc_val, seed=args.seed + 101),
        train_cfg=train_cfg,
        n_states=args.n_val_states,
    )
    _write_json(args.output_dir / "natural_eval.json", natural_eval)

    for policy in _policies(args.policies):
        print(f"Training rule-based policy network: {policy}.", flush=True)
        rule_stop = _resolved_stop(args, policy)
        rule_cfg = TrainConfig(
            batch_size=args.batch_size,
            sim_batch_size=args.sim_batch_size,
            episode_length=args.episode_length,
            episode_updates_per_episode=args.episode_updates_per_episode,
            episode_broad_share=args.episode_broad_share,
            lr=args.lr,
            steps=args.rule_steps,
            loss=args.loss,
            huber_delta=args.huber_delta,
            target_rms=rule_stop["target_rms"],
            target_max_abs=rule_stop["target_max_abs"],
            early_stop_patience=int(rule_stop["early_stop_patience"]),
            min_steps_before_stop=int(rule_stop["min_steps_before_stop"]),
            stop_val_states=args.stop_val_states,
            show_progress=not args.no_progress,
            checkpoint_dir=None if args.no_checkpoints else str(args.output_dir / "checkpoints"),
            checkpoint_name=policy,
            checkpoint_every=args.checkpoint_every,
            checkpoint_keep=args.checkpoint_keep,
            dtype=dtype,
            device=args.device,
            rule_scenario_q_weight=args.rule_scenario_q_weight,
            rule_calm_anchor_weight=args.rule_calm_anchor_weight,
            rule_scenario_burnin=args.rule_scenario_burnin,
            rule_scenario_horizon=args.rule_scenario_horizon,
        )
        if args.rule_trainer == "episode":
            rule_net, rule_log = train_rule_episode(
                natural_net,
                policy=policy,
                params=params,
                net_cfg=net_cfg,
                qmc_cfg=qmc_cfg,
                train_cfg=rule_cfg,
                episodes=args.rule_steps,
                log_every=args.log_every,
            )
        else:
            rule_net, rule_log = train_rule(
                natural_net,
                policy=policy,
                params=params,
                net_cfg=net_cfg,
                qmc_cfg=qmc_cfg,
                train_cfg=rule_cfg,
                log_every=args.log_every,
            )
        save_checkpoint(
            args.output_dir / f"{policy}.pt",
            rule_net,
            metadata={
                "kind": "rule",
                "policy": policy,
                "config": run_config,
                "selection": _selection_metadata(rule_log),
            },
        )
        _write_json(args.output_dir / f"{policy}_train_log.json", asdict(rule_log))
        rule_eval = evaluate_rule(
            rule_net,
            natural_net,
            policy=policy,
            params=params,
            qmc_cfg=QMCConfig(n_train=args.qmc_val, seed=args.seed + 202),
            train_cfg=rule_cfg,
            n_states=args.n_val_states,
        )
        _write_json(args.output_dir / f"{policy}_eval.json", rule_eval)


if __name__ == "__main__":
    main()
