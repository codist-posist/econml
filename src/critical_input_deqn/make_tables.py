from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import BaselineParams
from .experiments import (
    EXPERIMENTS,
    TABLE_EXPERIMENTS,
    experiment_registry_payload,
    experiment_root,
    params_from_dict,
    params_from_overrides,
)
from .optimal import period_utility
import torch


POLICIES = ("fixed", "ba", "discretion", "commitment")
POLICY_LABELS = {
    "fixed": "Fixed Taylor",
    "ba": "Bottleneck-adjusted Taylor",
    "discretion": "Discretion",
    "commitment": "Commitment",
}


def _postprocess_dir(base_root: Path, experiment: str) -> Path:
    return experiment_root(base_root, experiment) / "postprocess"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _params_for_postprocess(root: Path, experiment: str) -> BaselineParams:
    manifest = root / "postprocess_manifest.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        params = payload.get("params")
        if isinstance(params, dict):
            return params_from_dict(params)
    return params_from_overrides(EXPERIMENTS[experiment].overrides)


def _safe(arrays: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    value = arrays.get(name)
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _mean(arr: np.ndarray | None) -> float:
    if arr is None:
        return float("nan")
    return float(np.nanmean(arr))


def _std(arr: np.ndarray | None) -> float:
    if arr is None:
        return float("nan")
    return float(np.nanstd(arr))


def _freq_positive(arr: np.ndarray | None, tol: float = 1e-6) -> float:
    if arr is None:
        return float("nan")
    return float(np.nanmean(np.asarray(arr, dtype=float) > float(tol)))


def _utility_mean(defs: dict[str, np.ndarray], params: BaselineParams) -> float:
    C = _safe(defs, "C")
    N = _safe(defs, "N")
    if C is None or N is None:
        return float("nan")
    C_t = torch.as_tensor(C, dtype=torch.float64)
    N_t = torch.as_tensor(N, dtype=torch.float64)
    return float(period_utility(C_t, N_t, params).mean().cpu())


def policy_moment_row(
    *,
    experiment: str,
    policy: str,
    defs: dict[str, np.ndarray],
    params: BaselineParams,
) -> dict[str, float | str]:
    Pi = _safe(defs, "Pi")
    inflation_ann = None if Pi is None else 400.0 * (Pi - 1.0)
    output_gap = _safe(defs, "output_gap")
    chi = _safe(defs, "chi")
    I_A = _safe(defs, "I_A")
    A = _safe(defs, "A")
    cap_slack = _safe(defs, "cap_slack")
    return {
        "experiment": experiment,
        "policy": policy,
        "policy_label": POLICY_LABELS.get(policy, policy),
        "mean_inflation_ann_pp": _mean(inflation_ann),
        "std_inflation_ann_pp": _std(inflation_ann),
        "mean_output_gap_pct": 100.0 * _mean(output_gap),
        "std_output_gap_pct": 100.0 * _std(output_gap),
        "binding_frequency": _freq_positive(chi),
        "mean_scarcity_rent": _mean(chi),
        "std_scarcity_rent": _std(chi),
        "active_repair_frequency": _freq_positive(I_A),
        "mean_repair_investment": _mean(I_A),
        "std_repair_investment": _std(I_A),
        "mean_adaptation_stock": _mean(A),
        "std_adaptation_stock": _std(A),
        "mean_cap_slack": _mean(cap_slack),
        "mean_utility_flow": _utility_mean(defs, params),
    }


def collect_policy_rows(base_root: Path, experiment: str) -> list[dict[str, float | str]]:
    root = _postprocess_dir(base_root, experiment)
    params = _params_for_postprocess(root, experiment)
    rows: list[dict[str, float | str]] = []
    for policy in POLICIES:
        path = root / f"{policy}_definitions.npz"
        if not path.exists():
            continue
        rows.append(policy_moment_row(experiment=experiment, policy=policy, defs=_load_npz(path), params=params))
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def calibration_rows() -> list[dict[str, str | float]]:
    params = asdict(BaselineParams())
    return [{"parameter": key, "baseline_value": value} for key, value in params.items()]


def experiment_rows(names: Iterable[str]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for name in names:
        spec = EXPERIMENTS[name]
        row: dict[str, str | float] = {
            "experiment": spec.name,
            "group": spec.group,
            "description": spec.description,
        }
        row.update({f"override_{key}": value for key, value in spec.overrides.items()})
        rows.append(row)
    return rows


def make_tables(base_root: Path, output_dir: Path, *, policy_for_decomposition: str = "ba") -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    registry_path = output_dir / "experiment_registry.json"
    _write_json(registry_path, experiment_registry_payload())
    written["registry"] = str(registry_path)

    calibration_path = output_dir / "table_1_calibration.csv"
    _write_csv(calibration_path, calibration_rows())
    written["table_1"] = str(calibration_path)

    baseline_rows = collect_policy_rows(base_root, "baseline")
    table_2 = output_dir / "table_2_ergodic_moments_by_policy.csv"
    _write_csv(table_2, baseline_rows)
    written["table_2"] = str(table_2)

    counter_rows = []
    for experiment in TABLE_EXPERIMENTS["counterfactual"]:
        rows = [row for row in collect_policy_rows(base_root, experiment) if row["policy"] == policy_for_decomposition]
        counter_rows.extend(rows)
    table_3 = output_dir / f"table_3_counterfactual_decomposition_{policy_for_decomposition}.csv"
    _write_csv(table_3, counter_rows)
    written["table_3"] = str(table_3)

    sens_rows = []
    for experiment in TABLE_EXPERIMENTS["sensitivity"]:
        rows = [row for row in collect_policy_rows(base_root, experiment) if row["policy"] == policy_for_decomposition]
        sens_rows.extend(rows)
    table_4 = output_dir / f"table_4_sensitivity_summary_{policy_for_decomposition}.csv"
    _write_csv(table_4, sens_rows)
    written["table_4"] = str(table_4)

    experiment_table = output_dir / "experiment_overrides.csv"
    _write_csv(experiment_table, experiment_rows(EXPERIMENTS.keys()))
    written["experiment_overrides"] = str(experiment_table)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables from critical-input DEQN postprocess artifacts.")
    parser.add_argument("--base-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--policy", default="ba", choices=POLICIES)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.base_root / "tables")
    written = make_tables(args.base_root, output_dir, policy_for_decomposition=args.policy)
    print(json.dumps(written, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
