from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import NetworkConfig, QMCConfig, TrainConfig
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optimal-policy DEQN networks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("discretion", "commitment"), required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sim-batch-size", type=int, default=1024)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", default="huber", choices=("huber", "mse"))
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--qmc-train", type=int, default=512)
    parser.add_argument("--qmc-val", type=int, default=4096)
    parser.add_argument("--n-val-states", type=int, default=4096)
    parser.add_argument("--hidden-width", type=int, default=192)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    dtype = _dtype(args.dtype)
    net_cfg = NetworkConfig(hidden_width=args.hidden_width, hidden_depth=args.hidden_depth)
    qmc_cfg = QMCConfig(n_train=args.qmc_train, n_val=args.qmc_val, seed=args.seed)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        sim_batch_size=args.sim_batch_size,
        episode_length=args.episode_length,
        lr=args.lr,
        steps=args.steps,
        loss=args.loss,
        huber_delta=args.huber_delta,
        dtype=dtype,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "kind": args.kind,
        "network": asdict(net_cfg),
        "qmc": asdict(qmc_cfg),
        "train": {
            "batch_size": args.batch_size,
            "sim_batch_size": args.sim_batch_size,
            "episode_length": args.episode_length,
            "lr": args.lr,
            "steps": args.steps,
            "loss": args.loss,
            "huber_delta": args.huber_delta,
            "dtype": args.dtype,
            "device": args.device,
        },
    }
    _write_json(args.output_dir / "run_config.json", run_config)
    net, log = train_optimal_episode(
        kind=args.kind,
        net_cfg=net_cfg,
        qmc_cfg=qmc_cfg,
        train_cfg=train_cfg,
        episodes=args.steps,
        log_every=args.log_every,
    )
    save_checkpoint(args.output_dir / f"{args.kind}.pt", net, metadata=run_config)
    _write_json(args.output_dir / f"{args.kind}_train_log.json", asdict(log))
    eval_metrics = evaluate_optimal(
        net,
        kind=args.kind,
        qmc_cfg=QMCConfig(n_train=args.qmc_val, seed=args.seed + 303),
        train_cfg=train_cfg,
        n_states=args.n_val_states,
    )
    _write_json(args.output_dir / f"{args.kind}_eval.json", eval_metrics)


if __name__ == "__main__":
    main()
