"""Post-training quality report: equilibrium residuals on a stored simulated path.

Usage (from project root):
  python -m scripts.trajectory_quality_report --artifacts_root ../artifacts --policy discretion

This loads:
  - trained weights from the selected/latest run directory for the policy
  - sim_paths.npz from the same run
and writes:
  - trajectory_residuals.json

The residuals are computed using the exact same expectation operator as training
(Appendix B), but evaluated on states reconstructed from the simulated ergodic path.
"""

from __future__ import annotations

import os
import sys

# Allow running this script from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import os

import torch

from src.config import ModelParams
from src.io_utils import save_json
from src.sanity_checks import trajectory_residuals_check
from src.steady_states import solve_flexprice_sss, export_rbar_tensor
from src.table2_builder import _load_run_dir, _load_net_from_run, _load_sim_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", type=str, required=True)
    ap.add_argument("--policy", type=str, required=True, choices=["taylor", "mod_taylor", "discretion", "commitment"])
    ap.add_argument(
        "--use_selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer selected_runs.json over latest complete run (use --no-use_selected to ignore it).",
    )
    ap.add_argument("--gh_n", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--max_states", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=4096)
    args = ap.parse_args()

    params = ModelParams(device="cpu")
    run_dir = _load_run_dir(args.artifacts_root, args.policy, use_selected=bool(args.use_selected))
    net = _load_net_from_run(run_dir, params, args.policy)  # type: ignore[arg-type]
    sim = _load_sim_paths(run_dir)

    rbar_by_regime: torch.Tensor | None = None
    if args.policy == "mod_taylor":
        flex = solve_flexprice_sss(params)
        rbar_by_regime = export_rbar_tensor(params, flex).to(device=params.device, dtype=params.dtype)

    res = trajectory_residuals_check(
        params,
        net,
        policy=args.policy,  # type: ignore[arg-type]
        sim_paths=sim,
        rbar_by_regime=rbar_by_regime,
        gh_n=int(args.gh_n),
        tol=float(args.tol),
        max_states=int(args.max_states),
        batch_size=int(args.batch_size),
        seed=0,
    )

    out = {
        "policy": str(args.policy),
        "run_dir": str(run_dir),
        "n_states_evaluated": int(res.n_states_evaluated),
        "tol": float(res.tol),
        "metrics": dict(res.metrics),
    }

    save_json(os.path.join(run_dir, "trajectory_residuals.json"), out)
    print("saved", os.path.join(run_dir, "trajectory_residuals.json"))


if __name__ == "__main__":
    main()
