from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import BaselineParams, NetworkConfig, QMCConfig
from .experiments import resolve_params
from .optimal import commitment_residuals, discretion_residuals
from .qmc import make_qmc_nodes
from .residuals import natural_residuals, rule_residuals, stack_residuals
from .sampling import natural_from_rule_states, sample_rule_states
from .train import (
    _initial_optimal_states,
    initialize_commitment_promises,
    make_commitment_net,
    make_discretion_net,
    make_natural_net,
    make_rule_net,
    residual_diagnostics,
)


def _dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be float64 or float32.")


def _check(name: str, residuals: dict[str, torch.Tensor]) -> dict[str, float]:
    mat = stack_residuals(residuals)
    if not torch.isfinite(mat).all():
        bad = torch.isfinite(mat).logical_not().sum().item()
        raise RuntimeError(f"{name} residuals contain {bad} non-finite entries.")
    return residual_diagnostics(residuals)


def run_preflight(
    *,
    n_states: int = 8,
    qmc_nodes: int = 16,
    hidden_width: int = 32,
    hidden_depth: int = 2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    params: BaselineParams | None = None,
) -> dict[str, dict[str, float]]:
    """Check all DEQN residual systems before a long training run.

    This is intentionally not a convergence test.  It only verifies that each
    policy environment can build states, decode network outputs, evaluate QMC
    expectations, apply complementarity residuals, and produce finite residual
    matrices with the current code.
    """

    params = params or BaselineParams()
    net_cfg = NetworkConfig(hidden_width=int(hidden_width), hidden_depth=int(hidden_depth))
    qmc_cfg = QMCConfig(n_train=int(qmc_nodes), n_val=int(qmc_nodes), seed=11)
    nodes = make_qmc_nodes(int(qmc_nodes), cfg=qmc_cfg, device=device, dtype=dtype)
    z = sample_rule_states(int(n_states), params=params, device=device, dtype=dtype, seed=17)
    z_n = natural_from_rule_states(z)

    natural_net = make_natural_net(net_cfg, device=device, dtype=dtype)
    rule_net = make_rule_net(net_cfg, device=device, dtype=dtype)
    discretion_net = make_discretion_net(net_cfg, device=device, dtype=dtype)
    commitment_net = make_commitment_net(net_cfg, device=device, dtype=dtype)
    initialize_commitment_promises(commitment_net)

    out: dict[str, dict[str, float]] = {}

    res_n, _ = natural_residuals(
        z_n,
        natural_net(z_n),
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=1e-6,
    )
    out["natural"] = _check("natural", res_n)

    for policy in ("fixed", "ba", "bottleneck"):
        res_rule, _ = rule_residuals(
            z,
            rule_net(z),
            rule_net,
            natural_net,
            nodes,
            params=params,
            qmc_cfg=qmc_cfg,
            fb_epsilon=1e-6,
            policy=policy,
        )
        out[policy] = _check(policy, res_rule)

    res_disc, _ = discretion_residuals(
        z,
        discretion_net(z),
        discretion_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=1e-6,
    )
    out["discretion"] = _check("discretion", res_disc)

    zc = _initial_optimal_states(
        int(n_states),
        kind="commitment",
        params=params,
        device=device,
        dtype=dtype,
        promise_init_scale=1.0,
    )
    res_com, _ = commitment_residuals(
        zc,
        commitment_net(zc),
        commitment_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=1e-6,
    )
    out["commitment"] = _check("commitment", res_com)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight all critical-input DEQN residual systems.")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--qmc-nodes", type=int, default=16)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    args = parser.parse_args()

    params, experiment_meta = resolve_params(args.experiment, args.params_json)
    diagnostics = run_preflight(
        n_states=args.n_states,
        qmc_nodes=args.qmc_nodes,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        device=args.device,
        dtype=_dtype(args.dtype),
        params=params,
    )
    payload = {
        "experiment": experiment_meta,
        "config": {
            "n_states": args.n_states,
            "qmc_nodes": args.qmc_nodes,
            "hidden_width": args.hidden_width,
            "hidden_depth": args.hidden_depth,
            "device": args.device,
            "dtype": args.dtype,
        },
        "diagnostics": diagnostics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("critical_input_deqn preflight OK")


if __name__ == "__main__":
    main()
