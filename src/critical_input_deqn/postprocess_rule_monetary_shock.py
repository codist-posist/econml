from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .experiments import params_from_metadata, resolve_params
from .monetary_shock import (
    MonetaryShockConfig,
    evaluate_rule_shock_path,
    make_rule_shock_net,
    save_ir_artifacts,
    simulate_rule_monetary_ir_scenarios,
)
from .postprocess import _dtype, _load_payload, _network_config_from_metadata, load_natural


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these checkpoints exists: " + ", ".join(str(p) for p in paths))


def _shock_cfg_from_metadata(metadata: Mapping[str, object], fallback: MonetaryShockConfig) -> MonetaryShockConfig:
    data = metadata.get("shock_config", {})
    if not isinstance(data, Mapping):
        return fallback
    return MonetaryShockConfig(
        rho_R=float(data.get("rho_R", fallback.rho_R)),
        train_shock_std=float(data.get("train_shock_std", fallback.train_shock_std)),
        train_shock_span=float(data.get("train_shock_span", fallback.train_shock_span)),
        small_bp_annualized=float(data.get("small_bp_annualized", fallback.small_bp_annualized)),
        large_bp_annualized=float(data.get("large_bp_annualized", fallback.large_bp_annualized)),
    )


def _load_rule_shock(path: Path, *, device: str, dtype: torch.dtype) -> tuple[torch.nn.Module, dict[str, object]]:
    payload = _load_payload(path, device=device)
    metadata = dict(payload.get("metadata", {}))
    net = make_rule_shock_net(_network_config_from_metadata(metadata), device=device, dtype=dtype)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, metadata


def _peak_ir_summary(outputs: dict[str, np.ndarray], labels: list[str]) -> dict[str, dict[str, float]]:
    variables = ("R", "Pi", "chi", "I_A", "A", "output_gap", "eps_R", "R_shock_multiplier")
    base = "no_shock"
    summary: dict[str, dict[str, float]] = {}
    for label in labels:
        if label == base:
            continue
        row: dict[str, float] = {}
        for variable in variables:
            key = f"{label}__{variable}"
            base_key = f"{base}__{variable}"
            if key not in outputs or base_key not in outputs:
                continue
            diff = np.asarray(outputs[key], dtype=float) - np.asarray(outputs[base_key], dtype=float)
            row[f"{variable}.peak"] = float(np.max(diff))
            row[f"{variable}.trough"] = float(np.min(diff))
            row[f"{variable}.abs_peak"] = float(diff[np.argmax(np.abs(diff))])
        summary[label] = row
    return summary


def _save_flat_ir(
    *,
    policy: str,
    labels: list[str],
    states_np: dict[str, np.ndarray],
    outputs_np: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    save_ir_artifacts(policy=policy, labels=labels, states_np=states_np, outputs_np=outputs_np, out_dir=output_dir)
    flat_defs: dict[str, np.ndarray] = {"labels": np.asarray(labels)}
    for i, label in enumerate(labels):
        for name, arr in outputs_np.items():
            flat_defs[f"{label}__{name}"] = arr[:, i]
    return flat_defs


def run_postprocess_rule_monetary_shock(
    *,
    artifact_root: Path,
    output_dir: Path,
    natural_checkpoint: Path,
    experiment: str,
    params_json: Path | None,
    policies: list[str],
    ir_burnin: int,
    ir_horizon: int,
    ir_presteps: int,
    rho_R_shock: float | None,
    small_bp_annualized: float | None,
    large_bp_annualized: float | None,
    device: str,
    dtype: torch.dtype,
) -> None:
    params, experiment_meta = resolve_params(experiment, params_json)
    natural = load_natural(natural_checkpoint, device=device, dtype=dtype)
    if params_json is None:
        params = params_from_metadata(natural.metadata, fallback=params)
        experiment_meta["params"] = asdict(params)

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for policy in policies:
        path = _first_existing(
            [
                artifact_root / f"{policy}_monetary_shock.pt",
                artifact_root / policy / f"{policy}_monetary_shock.pt",
            ]
        )
        net, metadata = _load_rule_shock(path, device=device, dtype=dtype)
        shock_cfg = _shock_cfg_from_metadata(metadata, MonetaryShockConfig())
        if rho_R_shock is not None:
            shock_cfg = MonetaryShockConfig(
                rho_R=float(rho_R_shock),
                train_shock_std=shock_cfg.train_shock_std,
                train_shock_span=shock_cfg.train_shock_span,
                small_bp_annualized=shock_cfg.small_bp_annualized,
                large_bp_annualized=shock_cfg.large_bp_annualized,
            )
        if small_bp_annualized is not None or large_bp_annualized is not None:
            shock_cfg = MonetaryShockConfig(
                rho_R=shock_cfg.rho_R,
                train_shock_std=shock_cfg.train_shock_std,
                train_shock_span=shock_cfg.train_shock_span,
                small_bp_annualized=(
                    shock_cfg.small_bp_annualized if small_bp_annualized is None else float(small_bp_annualized)
                ),
                large_bp_annualized=(
                    shock_cfg.large_bp_annualized if large_bp_annualized is None else float(large_bp_annualized)
                ),
            )
        labels, states = simulate_rule_monetary_ir_scenarios(
            policy=policy,
            rule_net=net,
            natural_net=natural.net,
            params=params,
            shock_cfg=shock_cfg,
            burnin=ir_burnin,
            horizon=ir_horizon,
            presteps=ir_presteps,
            device=device,
            dtype=dtype,
        )
        states_np, outputs_np = evaluate_rule_shock_path(
            states,
            policy=policy,
            rule_net=net,
            natural_net=natural.net,
            params=params,
        )
        flat_outputs = _save_flat_ir(
            policy=policy,
            labels=labels,
            states_np=states_np,
            outputs_np=outputs_np,
            output_dir=output_dir,
        )
        summaries[policy] = _peak_ir_summary(flat_outputs, labels)
        files.extend(
            [
                f"IR_{policy}_monetary_shock_states.npz",
                f"IR_{policy}_monetary_shock_definitions.npz",
            ]
        )

    manifest = {
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "natural_checkpoint": str(natural_checkpoint),
        "experiment": experiment_meta,
        "params": asdict(params),
        "policies": policies,
        "ir": {
            "burnin": int(ir_burnin),
            "horizon": int(ir_horizon),
            "presteps": int(ir_presteps),
            "scenario_type": "one_time_monetary_policy_shock",
        },
        "files": sorted(set(files)),
    }
    with (output_dir / "monetary_shock_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    with (output_dir / "monetary_shock_peak_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess one-time monetary-shock IRFs for Taylor-rule DEQN networks.")
    parser.add_argument("--artifact-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn/rule_monetary_shock"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--natural-checkpoint", type=Path, default=Path("baseline_artifacts/critical_input_deqn/natural/natural.pt"))
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--policies", default="fixed,ba")
    parser.add_argument("--ir-burnin", type=int, default=200)
    parser.add_argument("--ir-horizon", type=int, default=80)
    parser.add_argument("--ir-presteps", type=int, default=5)
    parser.add_argument("--rho-R-shock", type=float, default=None)
    parser.add_argument("--small-bp-annualized", type=float, default=None)
    parser.add_argument("--large-bp-annualized", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    args = parser.parse_args()

    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
    bad = sorted(set(policies) - {"fixed", "ba", "bottleneck"})
    if bad:
        raise ValueError(f"Unknown policy names: {bad}. Use fixed, ba, bottleneck, or a comma-separated subset.")
    output_dir = args.output_dir or (args.artifact_root / "postprocess")
    run_postprocess_rule_monetary_shock(
        artifact_root=args.artifact_root,
        output_dir=output_dir,
        natural_checkpoint=args.natural_checkpoint,
        experiment=args.experiment,
        params_json=args.params_json,
        policies=policies,
        ir_burnin=args.ir_burnin,
        ir_horizon=args.ir_horizon,
        ir_presteps=args.ir_presteps,
        rho_R_shock=args.rho_R_shock,
        small_bp_annualized=args.small_bp_annualized,
        large_bp_annualized=args.large_bp_annualized,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    print(f"rule monetary-shock artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
