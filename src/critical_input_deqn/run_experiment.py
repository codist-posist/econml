from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .experiments import EXPERIMENTS, experiment_root


def _base_cmd() -> list[str]:
    return [sys.executable, "-u", "-m"]


def _append_common_training_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd += [
        "--experiment",
        args.experiment,
        "--qmc-train",
        str(args.qmc_train),
        "--qmc-val",
        str(args.qmc_val),
        "--n-val-states",
        str(args.n_val_states),
        "--stop-val-states",
        str(args.stop_val_states),
        "--hidden-width",
        str(args.hidden_width),
        "--hidden-depth",
        str(args.hidden_depth),
        "--batch-size",
        str(args.batch_size),
        "--sim-batch-size",
        str(args.sim_batch_size),
        "--episode-length",
        str(args.episode_length),
        "--episode-updates-per-episode",
        str(args.episode_updates_per_episode),
        "--episode-broad-share",
        str(args.episode_broad_share),
        "--lr",
        str(args.lr),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--checkpoint-keep",
        str(args.checkpoint_keep),
        "--log-every",
        str(args.log_every),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.no_progress:
        cmd.append("--no-progress")
    if args.no_auto_stop:
        cmd.append("--no-auto-stop")
    if args.params_json is not None:
        cmd += ["--params-json", str(args.params_json)]
    return cmd


def _append_rule_training_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd = _append_common_training_args(cmd, args)
    cmd += [
        "--rule-scenario-q-weight",
        str(args.rule_scenario_q_weight),
        "--rule-calm-anchor-weight",
        str(args.rule_calm_anchor_weight),
        "--rule-calm-residual-weight",
        str(args.rule_calm_residual_weight),
        "--rule-scenario-burnin",
        str(args.rule_scenario_burnin),
        "--rule-scenario-horizon",
        str(args.rule_scenario_horizon),
        "--rule-scenario-loss-interval",
        str(args.rule_scenario_loss_interval),
        "--best-scenario-q-weight",
        str(args.best_scenario_q_weight),
        "--best-calm-anchor-weight",
        str(args.best_calm_anchor_weight),
        "--best-calm-residual-weight",
        str(args.best_calm_residual_weight),
        "--target-scenario-q-rms",
        str(args.target_scenario_q_rms),
    ]
    return cmd


def _append_optimal_training_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd = _append_common_training_args(cmd, args)
    cmd += [
        "--scenario-q-weight",
        str(args.rule_scenario_q_weight),
        "--calm-anchor-weight",
        str(args.rule_calm_anchor_weight),
        "--calm-residual-weight",
        str(args.rule_calm_residual_weight),
        "--scenario-burnin",
        str(args.rule_scenario_burnin),
        "--scenario-horizon",
        str(args.rule_scenario_horizon),
        "--scenario-loss-interval",
        str(args.rule_scenario_loss_interval),
        "--best-scenario-q-weight",
        str(args.best_scenario_q_weight),
        "--best-calm-anchor-weight",
        str(args.best_calm_anchor_weight),
        "--best-calm-residual-weight",
        str(args.best_calm_residual_weight),
        "--target-scenario-q-rms",
        str(args.target_scenario_q_rms),
        "--feasibility-pretrain-steps",
        str(args.optimal_feasibility_pretrain_steps),
        "--private-loss-weight",
        str(args.optimal_private_loss_weight),
        "--bellman-loss-weight",
        str(args.optimal_bellman_loss_weight),
        "--stationarity-loss-weight",
        str(args.optimal_stationarity_loss_weight),
        "--promise-loss-weight",
        str(args.optimal_promise_loss_weight),
    ]
    return cmd


def _natural_checkpoint(root: Path) -> Path:
    best = root / "natural" / "checkpoints" / "natural_best.pt"
    final = root / "natural" / "natural.pt"
    return best if best.exists() else final


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    root = experiment_root(args.base_root, args.experiment)
    cmds: list[list[str]] = []

    if args.stage in {"natural", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_train",
            "--output-dir",
            str(root / "natural"),
            "--policies",
            "",
            "--natural-steps",
            str(args.natural_steps),
        ]
        cmds.append(_append_common_training_args(cmd, args))

    if args.stage in {"rules", "fixed", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_train",
            "--output-dir",
            str(root / "fixed_taylor"),
            "--natural-checkpoint",
            str(_natural_checkpoint(root)),
            "--policies",
            "fixed",
            "--rule-steps",
            str(args.rule_steps),
        ]
        cmds.append(_append_rule_training_args(cmd, args))

    if args.stage in {"rules", "ba", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_train",
            "--output-dir",
            str(root / "modified_taylor"),
            "--natural-checkpoint",
            str(_natural_checkpoint(root)),
            "--policies",
            "ba",
            "--rule-steps",
            str(args.rule_steps),
        ]
        cmds.append(_append_rule_training_args(cmd, args))

    if args.stage in {"rules", "bottleneck", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_train",
            "--output-dir",
            str(root / "bottleneck_taylor"),
            "--natural-checkpoint",
            str(_natural_checkpoint(root)),
            "--policies",
            "bottleneck",
            "--rule-steps",
            str(args.rule_steps),
        ]
        cmds.append(_append_rule_training_args(cmd, args))

    if args.stage in {"rules", "repair_aware", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_train",
            "--output-dir",
            str(root / "repair_aware_taylor"),
            "--natural-checkpoint",
            str(_natural_checkpoint(root)),
            "--policies",
            "repair_aware",
            "--rule-steps",
            str(args.rule_steps),
        ]
        cmds.append(_append_rule_training_args(cmd, args))

    if args.stage in {"discretion", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_optimal",
            "--output-dir",
            str(root / "discretion"),
            "--kind",
            "discretion",
            "--steps",
            str(args.optimal_steps),
        ]
        cmds.append(_append_optimal_training_args(cmd, args))

    if args.stage in {"commitment", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.run_optimal",
            "--output-dir",
            str(root / "commitment"),
            "--kind",
            "commitment",
            "--steps",
            str(args.optimal_steps),
            "--promise-init-scale",
            str(args.promise_init_scale),
        ]
        cmds.append(_append_optimal_training_args(cmd, args))

    if args.stage in {"postprocess", "all"}:
        cmd = _base_cmd() + [
            "src.critical_input_deqn.postprocess",
            "--artifact-root",
            str(root),
            "--output-dir",
            str(root / "postprocess"),
            "--experiment",
            args.experiment,
            "--length",
            str(args.postprocess_length),
            "--batch-size",
            str(args.postprocess_batch_size),
            "--ir-burnin",
            str(args.ir_burnin),
            "--ir-horizon",
            str(args.ir_horizon),
            "--ir-presteps",
            str(args.ir_presteps),
            "--ir-relief-lag",
            str(args.ir_relief_lag),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
        ]
        if args.params_json is not None:
            cmd += ["--params-json", str(args.params_json)]
        cmds.append(cmd)
    return cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one critical-input DEQN experiment in a structured artifact folder.")
    parser.add_argument("--base-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--experiment", default="baseline", choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "natural",
            "fixed",
            "ba",
            "bottleneck",
            "repair_aware",
            "rules",
            "discretion",
            "commitment",
            "postprocess",
            "all",
        ),
    )
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--natural-steps", type=int, default=20_000)
    parser.add_argument("--rule-steps", type=int, default=8_000)
    parser.add_argument("--optimal-steps", type=int, default=8_000)
    parser.add_argument("--qmc-train", type=int, default=256)
    parser.add_argument("--qmc-val", type=int, default=512)
    parser.add_argument("--n-val-states", type=int, default=1024)
    parser.add_argument("--stop-val-states", type=int, default=512)
    parser.add_argument("--hidden-width", type=int, default=192)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sim-batch-size", type=int, default=512)
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument("--episode-updates-per-episode", type=int, default=2)
    parser.add_argument("--episode-broad-share", type=float, default=0.50)
    parser.add_argument("--rule-scenario-q-weight", type=float, default=25.0)
    parser.add_argument("--rule-calm-anchor-weight", type=float, default=5.0)
    parser.add_argument("--rule-calm-residual-weight", type=float, default=5.0)
    parser.add_argument("--rule-scenario-burnin", type=int, default=5)
    parser.add_argument("--rule-scenario-horizon", type=int, default=10)
    parser.add_argument("--rule-scenario-loss-interval", type=int, default=25)
    parser.add_argument("--best-scenario-q-weight", type=float, default=1.0)
    parser.add_argument("--best-calm-anchor-weight", type=float, default=1.0)
    parser.add_argument("--best-calm-residual-weight", type=float, default=1.0)
    parser.add_argument("--target-scenario-q-rms", type=float, default=1e-2)
    parser.add_argument("--optimal-feasibility-pretrain-steps", type=int, default=1_000)
    parser.add_argument("--optimal-private-loss-weight", type=float, default=1.0)
    parser.add_argument("--optimal-bellman-loss-weight", type=float, default=0.25)
    parser.add_argument("--optimal-stationarity-loss-weight", type=float, default=0.10)
    parser.add_argument("--optimal-promise-loss-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--promise-init-scale", type=float, default=1.0)
    parser.add_argument("--postprocess-length", type=int, default=2000)
    parser.add_argument("--postprocess-batch-size", type=int, default=64)
    parser.add_argument("--ir-burnin", type=int, default=400)
    parser.add_argument("--ir-horizon", type=int, default=200)
    parser.add_argument("--ir-presteps", type=int, default=5)
    parser.add_argument("--ir-relief-lag", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-auto-stop", action="store_true")
    args = parser.parse_args()

    commands = build_commands(args)
    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
