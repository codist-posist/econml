"""One-step training + short simulation smoke test for all four policies.

This is meant for *quick* consistency checks after code edits:
  - builds minimal networks
  - runs 1 step in each of the 2 phases (total 2 optimizer steps)
  - runs a short simulation with store_states=True and compute_implied_i=True

It is intentionally tiny and should finish in seconds on CPU.
"""

from __future__ import annotations

import os
import sys

# Allow running this script from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.config import ModelParams, TrainConfig, PhaseConfig
from src.deqn import Trainer, PolicyNetwork, simulate_paths
from src.steady_states import solve_flexprice_sss, export_rbar_tensor


def main() -> None:
    params = ModelParams(device="cpu", dtype=torch.float32).to_torch()
    flex = solve_flexprice_sss(params)
    rbar = export_rbar_tensor(params, flex)

    cfg = TrainConfig.dev(
        n_path=5,
        n_paths_per_step=1,
        val_size=0,
        log_every=1,
        phase1=PhaseConfig(steps=1, lr=1e-4, batch_size=8, gh_n_train=2, use_float64=False),
        phase2=PhaseConfig(steps=1, lr=1e-4, batch_size=8, gh_n_train=2, use_float64=False),
    )

    spec = {
        "taylor": (5, 8),
        "mod_taylor": (5, 8),
        "discretion": (5, 11),
        "commitment": (7, 13),
    }

    for pol, (d_in, d_out) in spec.items():
        net = PolicyNetwork(d_in, d_out, hidden=cfg.hidden_layers, activation=cfg.activation)
        tr = Trainer(
            params=params,
            cfg=cfg,
            policy=pol,
            net=net,
            rbar_by_regime=(rbar if pol == "mod_taylor" else None),
        )

        tr.train(n_path=cfg.n_path, n_paths_per_step=cfg.n_paths_per_step)

        x0 = tr.simulate_initial_state(16)
        sim = simulate_paths(
            params,
            pol,
            net,
            T=6,
            burn_in=1,
            x0=x0,
            rbar_by_regime=(rbar if pol == "mod_taylor" else None),
            compute_implied_i=True,
            gh_n=2,
            store_states=True,
        )

        assert "pi" in sim and "Delta" in sim and "s" in sim
        assert "i" in sim, "nominal rate i must be present"
        print(f"[{pol}] OK | pi_mean={sim['pi'].mean():+.3g} | Delta_mean={sim['Delta'].mean():+.3g}")

    print("ALL OK")


if __name__ == "__main__":
    main()
