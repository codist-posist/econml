"""Train all policy variants with the true mid preset and save artifacts.

This script is a notebook-free equivalent of:
  - notebooks/10_train_taylor.ipynb
  - notebooks/11_train_mod_taylor.ipynb
  - notebooks/12_train_discretion.ipynb
  - notebooks/13_train_commitment.ipynb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Allow running from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import ModelParams, TrainConfig, set_seeds
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
from src.steady_states import export_rbar_tensor, solve_commitment_sss_from_policy, solve_flexprice_sss
from src.table2_builder import _simulate_flex_prices_for_table2


DIMS: Dict[str, tuple[int, int]] = {
    "taylor": (5, 8),
    "mod_taylor": (5, 8),
    "discretion": (5, 11),
    "commitment": (7, 13),
}


def _parse_policies(raw: str) -> List[str]:
    out: List[str] = []
    for p in raw.split(","):
        name = p.strip()
        if not name:
            continue
        if name not in DIMS:
            raise ValueError(f"Unknown policy '{name}'. Expected one of: {sorted(DIMS)}")
        out.append(name)
    if not out:
        raise ValueError("No policies provided")
    return out


def _build_train_quality(trainer: Trainer, cfg: TrainConfig) -> dict:
    val_size = int(getattr(cfg, "val_size", 0))
    if val_size <= 0:
        return {"skipped": True, "reason": "val_size<=0"}
    # Discretion residuals use autograd (dF/dDelta, dG/dDelta); inference/no_grad would break them.
    ctx = torch.enable_grad() if trainer.policy == "discretion" else torch.inference_mode()
    with ctx:
        x_val = trainer.simulate_initial_state(val_size, commitment_sss=None)
        val_burn = int(getattr(cfg, "val_burn_in", 200))
        for _ in range(val_burn):
            x_val = trainer._step_state(x_val)
        resid = trainer._residuals(x_val).detach().cpu().numpy()
    return residual_quality(resid, tol=float(getattr(cfg, "report_tol", 1e-3)))


def train_one(policy: str, args: argparse.Namespace) -> str:
    set_seeds(int(args.seed))
    params = ModelParams(device=args.device, dtype=torch.float32).to_torch()
    cfg = TrainConfig.mid(seed=int(args.seed), artifacts_root=str(args.artifacts_root))

    run_dir = make_run_dir(str(args.artifacts_root), policy, tag=cfg.mode, seed=cfg.seed)
    save_run_metadata(run_dir, pack_config(params, cfg, extra={"policy": policy}))

    rbar = None
    if policy == "mod_taylor":
        flex = solve_flexprice_sss(params)
        rbar = export_rbar_tensor(params, flex)
        save_json(
            os.path.join(run_dir, "flex_ref.json"),
            {"by_regime": flex.by_regime, "rbar_by_regime": rbar.detach().cpu().tolist()},
        )

    d_in, d_out = DIMS[policy]
    net = PolicyNetwork(d_in, d_out, hidden=cfg.hidden_layers, activation=cfg.activation)
    trainer = Trainer(
        params=params,
        cfg=cfg,
        policy=policy,
        net=net,
        rbar_by_regime=(rbar if policy == "mod_taylor" else None),
    )

    t0 = time.time()
    losses = trainer.train(
        commitment_sss=None,
        n_path=cfg.n_path,
        n_paths_per_step=cfg.n_paths_per_step,
    )
    train_seconds = float(time.time() - t0)

    save_torch(os.path.join(run_dir, "weights.pt"), trainer.net.state_dict())
    save_csv(
        os.path.join(run_dir, "train_log.csv"),
        pd.DataFrame({"iter": np.arange(len(losses), dtype=np.int64), "loss": np.asarray(losses, dtype=np.float64)}),
    )
    save_json(os.path.join(run_dir, "train_quality.json"), _build_train_quality(trainer, cfg))
    save_json(
        os.path.join(run_dir, "timing.json"),
        {"policy": policy, "train_seconds": train_seconds, "n_loss_points": int(len(losses))},
    )

    if policy == "commitment":
        comm_sss = solve_commitment_sss_from_policy(params, trainer.net)
        save_json(
            os.path.join(run_dir, "sss_policy_fixed_point.json"),
            {"policy": "commitment", "by_regime": comm_sss.by_regime},
        )
        x0 = trainer.simulate_initial_state(int(args.commitment_b), commitment_sss=comm_sss.by_regime)
        sim = simulate_paths(
            params=params,
            policy="commitment",
            net=trainer.net,
            T=int(args.commitment_t),
            burn_in=int(args.commitment_burn),
            x0=x0,
            compute_implied_i=True,
            gh_n=int(args.commitment_gh),
            thin=int(args.commitment_thin),
            show_progress=bool(args.show_progress),
        )
    else:
        x0 = trainer.simulate_initial_state(int(args.sim_b), commitment_sss=None)
        sim = simulate_paths(
            params=params,
            policy=policy,  # type: ignore[arg-type]
            net=trainer.net,
            T=int(args.sim_t),
            burn_in=int(args.sim_burn),
            x0=x0,
            rbar_by_regime=(rbar if policy == "mod_taylor" else None),
            compute_implied_i=(policy in ("discretion", "commitment")),
            gh_n=int(args.sim_gh),
            thin=int(args.sim_thin),
            show_progress=bool(args.show_progress),
        )

    save_npz(os.path.join(run_dir, "sim_paths.npz"), **sim)
    save_selected_run(str(args.artifacts_root), policy, run_dir)

    print(f"[OK] {policy}: {run_dir}")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Train all policy variants with true mid preset and save artifacts.")
    ap.add_argument("--artifacts_root", default=os.path.join(_ROOT, "artifacts"))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", default="taylor,mod_taylor,discretion,commitment")
    ap.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show simulation progress bars (use --no-show_progress to disable).",
    )

    # Taylor/mod_taylor/discretion simulation defaults (from train notebooks).
    ap.add_argument("--sim_b", type=int, default=512)
    ap.add_argument("--sim_t", type=int, default=20000)
    ap.add_argument("--sim_burn", type=int, default=2000)
    ap.add_argument("--sim_gh", type=int, default=3)
    ap.add_argument("--sim_thin", type=int, default=10)

    # Commitment simulation defaults (from train notebook).
    ap.add_argument("--commitment_b", type=int, default=2048)
    ap.add_argument("--commitment_t", type=int, default=6000)
    ap.add_argument("--commitment_burn", type=int, default=1000)
    ap.add_argument("--commitment_gh", type=int, default=7)
    ap.add_argument("--commitment_thin", type=int, default=1)

    # Optional flex simulation for Table 2 to avoid expensive fallback in build_table2.
    ap.add_argument("--build_flex", action="store_true")
    ap.add_argument("--flex_b", type=int, default=2048)
    ap.add_argument("--flex_t", type=int, default=6000)
    ap.add_argument("--flex_burn", type=int, default=1000)
    ap.add_argument("--flex_gh", type=int, default=3)

    args = ap.parse_args()
    args.artifacts_root = os.path.abspath(os.path.expanduser(str(args.artifacts_root)))
    ensure_dir(args.artifacts_root)

    policies = _parse_policies(str(args.policies))
    print("Policies:", policies)
    print("Artifacts root:", args.artifacts_root)
    print("Preset: mid")

    started = time.time()
    run_dirs: Dict[str, str] = {}
    for policy in policies:
        run_dirs[policy] = train_one(policy, args)

    if bool(args.build_flex):
        params_flex = ModelParams(device=args.device, dtype=torch.float64).to_torch()
        flex_sim = _simulate_flex_prices_for_table2(
            params_flex,
            T=int(args.flex_t),
            burn_in=int(args.flex_burn),
            B=int(args.flex_b),
            gh_n=int(args.flex_gh),
            seed=int(args.seed),
        )
        flex_path = os.path.join(args.artifacts_root, "flex", "sim_paths.npz")
        save_npz(flex_path, **flex_sim)
        print(f"[OK] flex: {flex_path}")

    elapsed = float(time.time() - started)
    print("\nCompleted.")
    print(f"Elapsed seconds: {elapsed:.1f}")
    for k, v in run_dirs.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
