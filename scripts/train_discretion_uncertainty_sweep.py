#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Allow running from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import ModelParams, PhaseConfig, TrainConfig
from src.deqn import PolicyNetwork, Trainer, simulate_paths
from src.io_utils import (
    ensure_dir,
    make_run_dir,
    pack_config,
    save_csv,
    save_json,
    save_npz,
    save_run_metadata,
    save_selected_run,
    save_torch,
)
from src.metrics import residual_quality


def _parse_grid(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if len(vals) != 4:
        raise ValueError(f"Expected exactly 4 grid values, got {len(vals)} from: {text}")
    if any(v <= 0.0 for v in vals):
        raise ValueError(f"All grid values must be > 0, got: {vals}")
    return vals


def _variant_params(base: ModelParams, variant: str, bad_multiplier: float, normal_multiplier: float) -> ModelParams:
    if variant == "A":
        return base.with_sigma_tau_regime(
            bad_multiplier=float(bad_multiplier),
            normal_multiplier=float(normal_multiplier),
        )
    if variant == "B":
        return base.with_all_sigma_regime(
            bad_multiplier=float(bad_multiplier),
            normal_multiplier=float(normal_multiplier),
        )
    raise ValueError(f"Unknown variant: {variant}")


def _scenario_slug(variant: str, bad_multiplier: float) -> str:
    bm = f"{bad_multiplier:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"variant_{variant}_bm_{bm}"


def _to_native_dict(d: Dict) -> Dict:
    out: Dict = {}
    for k, v in d.items():
        if isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _train_one(
    *,
    params: ModelParams,
    cfg: TrainConfig,
    artifacts_root: str,
    scenario: str,
    variant: str,
    bad_multiplier: float,
    seed: int,
    sim_T: int,
    sim_burn_in: int,
    sim_init_B: int,
    sim_thin: int,
    show_progress: bool,
) -> Dict:
    policy = "discretion"
    d_in, d_out = 5, 11
    run_dir = make_run_dir(artifacts_root, policy, tag=f"{cfg.mode}_{scenario}", seed=seed)

    cfg = TrainConfig.mid_discretion_fast(
        seed=seed,
        run_dir=run_dir,
        artifacts_root=artifacts_root,
        phase1=cfg.phase1,
        phase2=cfg.phase2,
        n_path=cfg.n_path,
        n_paths_per_step=cfg.n_paths_per_step,
        val_size=cfg.val_size,
        val_every=cfg.val_every,
        log_every=cfg.log_every,
    )

    save_run_metadata(
        run_dir,
        pack_config(
            params,
            cfg,
            extra={
                "policy": policy,
                "scenario": scenario,
                "variant": variant,
                "bad_multiplier": float(bad_multiplier),
            },
        ),
    )

    net = PolicyNetwork(d_in, d_out, hidden=cfg.hidden_layers, activation=cfg.activation)
    trainer = Trainer(params=params, cfg=cfg, policy=policy, net=net)

    losses = trainer.train(commitment_sss=None, n_path=cfg.n_path, n_paths_per_step=cfg.n_paths_per_step)

    save_torch(os.path.join(run_dir, "weights.pt"), trainer.net.state_dict())
    save_csv(
        os.path.join(run_dir, "train_log.csv"),
        pd.DataFrame({"iter": np.arange(len(losses)), "loss": losses}),
    )

    # Validation residual quality
    with torch.enable_grad():
        x_val = trainer.simulate_initial_state(int(cfg.val_size), commitment_sss=None)
        val_burn = int(getattr(cfg, "val_burn_in", 200))
        for _ in range(val_burn):
            x_val = trainer._step_state(x_val)
        resid = trainer._residuals(x_val).detach().cpu().numpy()
    q = residual_quality(resid, tol=getattr(cfg, "report_tol", 1e-3))
    save_json(os.path.join(run_dir, "train_quality.json"), _to_native_dict(q))
    save_selected_run(artifacts_root, policy, run_dir)

    # Ergodic simulation
    x0 = trainer.simulate_initial_state(int(sim_init_B), commitment_sss=None)
    sim = simulate_paths(
        params=params,
        policy=policy,
        net=trainer.net,
        T=int(sim_T),
        burn_in=int(sim_burn_in),
        x0=x0,
        rbar_by_regime=None,
        compute_implied_i=True,
        gh_n=3,
        thin=int(sim_thin),
        show_progress=bool(show_progress),
    )
    save_npz(os.path.join(run_dir, "sim_paths.npz"), **sim)

    return {
        "scenario": scenario,
        "run_dir": run_dir,
        "final_loss": float(losses[-1]) if len(losses) else float("nan"),
        "best_loss": float(np.min(losses)) if len(losses) else float("nan"),
        "train_rms": float(q.get("rms", np.nan)),
        "train_max_abs": float(q.get("max_abs", np.nan)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Retrain discretion for critique scenarios with 4-point bad_multiplier grid. "
            "Variant A: sigma_tau regime-dependent. Variant B: all sigmas regime-dependent."
        )
    )
    ap.add_argument("--artifacts_root", default=os.path.join(_ROOT, "artifacts", "critique_discretion"))
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normal_multiplier", type=float, default=1.0)
    ap.add_argument("--bad_grid", default="1.0,1.25,1.5,2.0")
    ap.add_argument(
        "--variants",
        default="A,B",
        help="Comma-separated subset of A,B. Default: A,B",
    )
    ap.add_argument("--n_path", type=int, default=32)
    ap.add_argument("--phase1_steps", type=int, default=4500)
    ap.add_argument("--phase1_batch", type=int, default=96)
    ap.add_argument("--phase2_steps", type=int, default=700)
    ap.add_argument("--phase2_batch", type=int, default=64)
    ap.add_argument("--sim_T", type=int, default=12000)
    ap.add_argument("--sim_burn_in", type=int, default=1200)
    ap.add_argument("--sim_init_B", type=int, default=384)
    ap.add_argument("--sim_thin", type=int, default=10)
    ap.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show training and simulation progress bars.",
    )
    args = ap.parse_args()

    ensure_dir(args.artifacts_root)
    grid = _parse_grid(args.bad_grid)
    variants = [v.strip().upper() for v in str(args.variants).split(",") if v.strip()]
    for v in variants:
        if v not in {"A", "B"}:
            raise ValueError(f"--variants accepts only A,B. Got: {variants}")

    base = ModelParams(device=args.device, dtype=torch.float32).to_torch()
    cfg_fast_default = TrainConfig.mid_discretion_fast(seed=int(args.seed))
    cfg_template = TrainConfig.mid_discretion_fast(
        seed=int(args.seed),
        n_path=int(args.n_path),
        n_paths_per_step=1,
        phase1=PhaseConfig(
            steps=int(args.phase1_steps),
            lr=float(cfg_fast_default.phase1.lr),
            batch_size=int(args.phase1_batch),
            gh_n_train=int(cfg_fast_default.phase1.gh_n_train),
            use_float64=bool(cfg_fast_default.phase1.use_float64),
            eps_stop=cfg_fast_default.phase1.eps_stop,
        ),
        phase2=PhaseConfig(
            steps=int(args.phase2_steps),
            lr=float(cfg_fast_default.phase2.lr),
            batch_size=int(args.phase2_batch),
            gh_n_train=int(cfg_fast_default.phase2.gh_n_train),
            use_float64=bool(cfg_fast_default.phase2.use_float64),
            eps_stop=cfg_fast_default.phase2.eps_stop,
        ),
    )

    rows: List[Dict] = []
    for variant in variants:
        for bad_multiplier in grid:
            scenario = _scenario_slug(variant, float(bad_multiplier))
            scenario_root = os.path.join(args.artifacts_root, scenario)
            ensure_dir(scenario_root)

            params = _variant_params(
                base=base,
                variant=variant,
                bad_multiplier=float(bad_multiplier),
                normal_multiplier=float(args.normal_multiplier),
            )

            print(f"\n=== Train discretion: {scenario} ===")
            row = _train_one(
                params=params,
                cfg=cfg_template,
                artifacts_root=scenario_root,
                scenario=scenario,
                variant=variant,
                bad_multiplier=float(bad_multiplier),
                seed=int(args.seed),
                sim_T=int(args.sim_T),
                sim_burn_in=int(args.sim_burn_in),
                sim_init_B=int(args.sim_init_B),
                sim_thin=int(args.sim_thin),
                show_progress=bool(args.show_progress),
            )
            row["variant"] = variant
            row["bad_multiplier"] = float(bad_multiplier)
            rows.append(row)
            print(
                f"[done] {scenario} | best_loss={row['best_loss']:.3e} | "
                f"train_rms={row['train_rms']:.3e} | run={row['run_dir']}"
            )

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.artifacts_root, "discretion_sweep_summary.csv")
    out.to_csv(out_path, index=False)
    print("\nSaved summary:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
