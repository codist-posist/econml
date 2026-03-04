#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams, TrainConfig, PhaseConfig
from src.deqn import PolicyNetwork, Trainer, simulate_paths
from src.io_utils import (
    make_run_dir,
    save_run_metadata,
    pack_config,
    save_torch,
    save_csv,
    save_json,
    save_npz,
    save_selected_run,
)
from src.metrics import residual_quality
from src.steady_states import solve_flexprice_sss, export_rbar_tensor
from src.sss_from_policy import switching_policy_sss_by_regime_from_policy
from src.sanity_checks import fixed_point_check, residuals_check_switching_consistent


DIMS: Dict[str, Tuple[int, int]] = {
    "taylor": (5, 4),
    "taylor_para": (7, 6),
    "mod_taylor": (5, 4),
    "discretion": (5, 5),
    "commitment": (8, 5),
    "taylor_zlb": (5, 4),
    "mod_taylor_zlb": (5, 4),
    "discretion_zlb": (5, 6),
    "commitment_zlb": (10, 7),
}


def _cfg_quick(cfg: TrainConfig, *, n_path: int, n_paths_per_step: int, p1_steps: int, p2_steps: int) -> TrainConfig:
    p1 = replace(
        cfg.phase1,
        steps=int(p1_steps),
        lr=float(cfg.phase1.lr),
        batch_size=min(int(cfg.phase1.batch_size), 16),
        gh_n_train=min(int(cfg.phase1.gh_n_train), 2),
        use_float64=False,
        eps_stop=None,
    )
    p2 = replace(
        cfg.phase2,
        steps=int(p2_steps),
        lr=float(cfg.phase2.lr),
        batch_size=min(int(cfg.phase2.batch_size), 16),
        gh_n_train=min(int(cfg.phase2.gh_n_train), 2),
        use_float64=False,
        eps_stop=None,
    )
    return replace(
        cfg,
        n_path=int(n_path),
        n_paths_per_step=int(n_paths_per_step),
        val_size=min(int(cfg.val_size), 32),
        log_every=1,
        phase1=p1,
        phase2=p2,
    )


def _train_one(
    artifacts_root: str,
    policy: str,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
    n_path: int,
    n_paths_per_step: int,
    p1_steps: int,
    p2_steps: int,
    sim_T: int,
    sim_burn: int,
    sim_B: int,
) -> str:
    if policy not in DIMS:
        raise ValueError(f"Unsupported policy: {policy}")

    params = ModelParams(device=device, dtype=dtype).to_torch()
    cfg_probe = TrainConfig.author_like(policy=policy, seed=seed, n_paths_per_step=n_paths_per_step)
    run_dir = make_run_dir(artifacts_root, policy, tag=f"{cfg_probe.mode}_quick", seed=seed)
    cfg = TrainConfig.author_like(
        policy=policy,
        seed=seed,
        run_dir=run_dir,
        artifacts_root=artifacts_root,
        n_paths_per_step=n_paths_per_step,
    )
    cfg = _cfg_quick(cfg, n_path=n_path, n_paths_per_step=n_paths_per_step, p1_steps=p1_steps, p2_steps=p2_steps)

    save_run_metadata(run_dir, pack_config(params, cfg, extra={"policy": policy, "quick_compatible_run": True}))

    d_in, d_out = DIMS[policy]
    net = PolicyNetwork(
        d_in,
        d_out,
        hidden=cfg.hidden_layers,
        activation=cfg.activation,
        init_mode=getattr(cfg, "init_mode", "default"),
        init_scale=float(getattr(cfg, "init_scale", 0.01)),
        seed=int(cfg.seed),
    )

    rbar_by_regime = None
    if policy in ("mod_taylor", "mod_taylor_zlb"):
        flex = solve_flexprice_sss(params)
        rbar_by_regime = export_rbar_tensor(params, flex)

    trainer = Trainer(
        params=params,
        cfg=cfg,
        policy=policy,  # type: ignore[arg-type]
        net=net,
        rbar_by_regime=rbar_by_regime,
    )

    losses = trainer.train(
        commitment_sss=None,
        n_path=cfg.n_path,
        n_paths_per_step=cfg.n_paths_per_step,
    )
    save_torch(os.path.join(run_dir, "weights.pt"), trainer.net.state_dict())
    save_csv(os.path.join(run_dir, "train_log.csv"), pd.DataFrame({"iter": np.arange(len(losses)), "loss": losses}))

    ctx = torch.enable_grad() if policy in ("discretion", "discretion_zlb") else torch.inference_mode()
    with ctx:
        x_val = trainer.simulate_initial_state(int(cfg.val_size), commitment_sss=None)
        for _ in range(8):
            x_val = trainer._step_state(x_val)
        resid = trainer._residuals(x_val).detach().cpu().numpy()
    q = residual_quality(resid, tol=getattr(cfg, "report_tol", 1e-3))
    save_json(os.path.join(run_dir, "train_quality.json"), q)

    # SSS + sanity
    sss = switching_policy_sss_by_regime_from_policy(params, trainer.net, policy=policy)
    save_json(
        os.path.join(run_dir, "sss_policy_fixed_point.json"),
        {"policy": policy, "by_regime": sss.by_regime},
    )
    fp = fixed_point_check(params, trainer.net, policy=policy, sss_by_regime=sss.by_regime)
    rc = residuals_check_switching_consistent(params, trainer.net, policy=policy, sss_by_regime=sss.by_regime)
    save_json(
        os.path.join(run_dir, "sanity_checks.json"),
        {
            "policy": policy,
            "fixed_regime_one_step_max_abs_state_diff": {int(k): float(v.max_abs_state_diff) for k, v in fp.items()},
            "residual_max_abs": {int(k): float(v.max_abs_residual) for k, v in rc.items()},
            "residuals_by_regime": {int(k): {kk: float(vv) for kk, vv in v.residuals.items()} for k, v in rc.items()},
        },
    )

    # short simulation artifact
    x0 = trainer.simulate_initial_state(int(sim_B), commitment_sss=None)
    sim = simulate_paths(
        params=params,
        policy=policy,  # type: ignore[arg-type]
        net=trainer.net,
        T=int(sim_T),
        burn_in=int(sim_burn),
        x0=x0,
        rbar_by_regime=rbar_by_regime if policy in ("mod_taylor", "mod_taylor_zlb") else None,
        compute_implied_i=True,
        gh_n=3,
        thin=1,
        show_progress=False,
        store_states=True,
    )
    save_npz(os.path.join(run_dir, "sim_paths.npz"), **sim)

    save_selected_run(artifacts_root, policy, run_dir)
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Create quick decoder-compatible runs for current author-mode dimensions.")
    ap.add_argument("--artifacts_root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument(
        "--policies",
        nargs="+",
        default=[
            "taylor",
            "taylor_para",
            "mod_taylor",
            "discretion",
            "commitment",
            "taylor_zlb",
            "mod_taylor_zlb",
            "discretion_zlb",
            "commitment_zlb",
        ],
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_path", type=int, default=2)
    ap.add_argument("--n_paths_per_step", type=int, default=1)
    ap.add_argument("--phase1_steps", type=int, default=2)
    ap.add_argument("--phase2_steps", type=int, default=0)
    ap.add_argument("--sim_T", type=int, default=32)
    ap.add_argument("--sim_burn", type=int, default=8)
    ap.add_argument("--sim_B", type=int, default=8)
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    ok: Dict[str, str] = {}
    fail: Dict[str, str] = {}

    for policy in args.policies:
        try:
            run_dir = _train_one(
                args.artifacts_root,
                policy,
                device=args.device,
                dtype=dtype,
                seed=int(args.seed),
                n_path=int(args.n_path),
                n_paths_per_step=int(args.n_paths_per_step),
                p1_steps=int(args.phase1_steps),
                p2_steps=int(args.phase2_steps),
                sim_T=int(args.sim_T),
                sim_burn=int(args.sim_burn),
                sim_B=int(args.sim_B),
            )
            ok[policy] = run_dir
            print(f"[ok] {policy}: {run_dir}")
        except Exception as e:
            fail[policy] = str(e)
            print(f"[fail] {policy}: {e}")

    print("\n=== SUMMARY ===")
    print(f"ok={len(ok)} fail={len(fail)}")
    for k, v in ok.items():
        print(f" OK  {k}: {v}")
    for k, v in fail.items():
        print(f" FAIL {k}: {v}")

    return 0 if not fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
