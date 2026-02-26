"""Minimal end-to-end smoke test.

Goal: catch "it doesn't run" regressions after refactors.

This is *not* a correctness/replication test. It intentionally runs with tiny
budgets (few optimizer steps, short simulations) so it can finish quickly on CPU.

What it checks:
  - the 4 policy variants can be instantiated and trained (a couple steps)
  - SSS helpers run (including commitment timeless SSS)
  - forward simulations produce finite outputs
  - Table-2 builder can run using the produced artifacts (no full-scale moments)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import numpy as np
import torch

from src.config import ModelParams, TrainConfig, set_seeds
from src.deqn import PolicyNetwork, Trainer, simulate_paths
from src.io_utils import make_run_dir, save_selected_run, save_torch, save_npz
from src.steady_states import (
    solve_flexprice_sss,
    export_rbar_tensor,
    solve_commitment_sss_from_policy,
)
from src.table2_builder import build_table2, _simulate_flex_prices_for_table2


# Input/output dimensions used throughout the notebooks
DIMS: dict[str, tuple[int, int]] = {
    "taylor": (5, 8),
    "mod_taylor": (5, 8),
    "discretion": (5, 11),
    "commitment": (7, 13),
}


def run(policy: str, device: str = "cpu") -> None:
    set_seeds(123)

    params = ModelParams(device=device)
    smoke_artifacts = tempfile.mkdtemp(prefix="deqn_smoke_train_")
    # Ultra-light training config for smoke tests (keep architecture unchanged).
    base = TrainConfig.dev(run_dir=None, artifacts_root=smoke_artifacts)
    cfg = replace(
        base,
        n_path=8,
        n_paths_per_step=1,
        phase1=replace(base.phase1, steps=3, batch_size=16, gh_n_train=2, eps_stop=None),
        phase2=replace(base.phase2, steps=1, batch_size=16, gh_n_train=2, use_float64=False, eps_stop=None),
        log_every=999999,
    )

    d_in, d_out = DIMS[policy]
    net = PolicyNetwork(d_in, d_out, hidden=cfg.hidden_layers, activation=cfg.activation).to(
        device=params.device, dtype=params.dtype
    )

    rbar_by_regime = None
    if policy == "mod_taylor":
        flex = solve_flexprice_sss(params)
        rbar_by_regime = export_rbar_tensor(params, flex).to(device=params.device, dtype=params.dtype)

    trainer = Trainer(
        params=params,
        cfg=cfg,
        policy=policy,  # type: ignore[arg-type]
        net=net,
        rbar_by_regime=rbar_by_regime,
    )

    # Tiny training run (just to ensure training loop executes)
    trainer.train(commitment_sss=None, n_path=cfg.n_path)

    # Commitment timeless SSS is required for paper-style simulations.
    comm_sss = None
    if policy == "commitment":
        # Keep this extremely lightweight: we're only checking that the solver runs.
        comm_sss = solve_commitment_sss_from_policy(params, trainer.net, max_iter=20, tol=1e-4, damping=1.0)

    # Short forward simulation; ensure outputs are finite.
    x0 = trainer.simulate_initial_state(B=32, commitment_sss=(comm_sss.by_regime if comm_sss else None))
    sim = simulate_paths(
        params,
        policy=policy,  # type: ignore[arg-type]
        net=trainer.net,
        T=60,
        burn_in=10,
        x0=x0,
        rbar_by_regime=rbar_by_regime,
        compute_implied_i=True,
        gh_n=2,
        show_progress=False,
        store_states=False,
    )

    for k, v in sim.items():
        if not np.isfinite(v).all():
            raise AssertionError(f"Non-finite values in sim['{k}'] for policy='{policy}'")

    print("ok", policy, "trained+simulated")


def run_pipeline(device: str = "cpu") -> None:
    """Create minimal artifacts and ensure Table-2 builder runs."""
    set_seeds(123)
    params = ModelParams(device=device)
    artifacts_root = tempfile.mkdtemp(prefix="deqn_smoke_artifacts_")

    # Produce a tiny flex sim_paths so build_table2 doesn't run the expensive fallback.
    flex_sim = _simulate_flex_prices_for_table2(params, T=80, burn_in=10, B=128, gh_n=2, seed=123)
    os.makedirs(os.path.join(artifacts_root, "flex"), exist_ok=True)
    save_npz(os.path.join(artifacts_root, "flex", "sim_paths.npz"), **flex_sim)

    for pol in ["taylor", "mod_taylor", "discretion", "commitment"]:
        set_seeds(123)
        base = TrainConfig.dev(run_dir=None, artifacts_root=artifacts_root)
        cfg = replace(
            base,
            run_dir=None,
            n_path=8,
            n_paths_per_step=1,
            phase1=replace(base.phase1, steps=3, batch_size=16, gh_n_train=2, eps_stop=None),
            phase2=replace(base.phase2, steps=1, batch_size=16, gh_n_train=2, use_float64=False, eps_stop=None),
            log_every=999999,
        )

        d_in, d_out = DIMS[pol]
        net = PolicyNetwork(d_in, d_out, hidden=cfg.hidden_layers, activation=cfg.activation).to(
            device=params.device, dtype=params.dtype
        )
        rbar_by_regime = None
        if pol == "mod_taylor":
            flex = solve_flexprice_sss(params)
            rbar_by_regime = export_rbar_tensor(params, flex).to(device=params.device, dtype=params.dtype)

        tr = Trainer(params=params, cfg=cfg, policy=pol, net=net, rbar_by_regime=rbar_by_regime)
        tr.train(commitment_sss=None, n_path=cfg.n_path)

        comm_sss = None
        if pol == "commitment":
            comm_sss = solve_commitment_sss_from_policy(params, tr.net, max_iter=20, tol=1e-4, damping=1.0)

        x0 = tr.simulate_initial_state(B=64, commitment_sss=(comm_sss.by_regime if comm_sss else None))
        sim = simulate_paths(
            params,
            policy=pol,  # type: ignore[arg-type]
            net=tr.net,
            T=80,
            burn_in=10,
            x0=x0,
            rbar_by_regime=rbar_by_regime,
            compute_implied_i=True,
            gh_n=2,
            show_progress=False,
        )

        run_dir = make_run_dir(artifacts_root, pol, tag="smoke", seed=123)
        save_torch(os.path.join(run_dir, "weights.pt"), tr.net.state_dict())
        save_npz(os.path.join(run_dir, "sim_paths.npz"), **sim)
        save_selected_run(artifacts_root, pol, run_dir)

    df = build_table2(artifacts_root, device=str(params.device), include_rules=True)
    if df is None or df.empty:
        raise AssertionError("build_table2 returned empty output")
    print("ok", "table2", "rows", int(df.shape[0]), "cols", int(df.shape[1]))
    print("smoke artifacts:", artifacts_root)


if __name__ == "__main__":
    for pol in ["taylor", "mod_taylor", "discretion", "commitment"]:
        run(pol)
    run_pipeline()
