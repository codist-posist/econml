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


def _unique(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


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
    spec = EXPERIMENTS.get(experiment)
    return params_from_overrides({} if spec is None else spec.overrides)


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


def _utility_parts(defs: dict[str, np.ndarray], params: BaselineParams) -> dict[str, float]:
    C = _safe(defs, "C")
    N = _safe(defs, "N")
    if C is None or N is None:
        return {
            "mean_consumption": float("nan"),
            "std_consumption": float("nan"),
            "mean_labor": float("nan"),
            "std_labor": float("nan"),
            "mean_consumption_utility": float("nan"),
            "mean_labor_disutility": float("nan"),
            "mean_log_consumption": float("nan"),
        }
    sigma = float(params.sigma)
    varphi = float(params.varphi)
    C = np.asarray(C, dtype=float)
    N = np.asarray(N, dtype=float)
    if abs(sigma - 1.0) < 1e-10:
        u_c = np.log(C)
    else:
        u_c = C ** (1.0 - sigma) / (1.0 - sigma)
    labor = N ** (1.0 + varphi) / (1.0 + varphi)
    return {
        "mean_consumption": _mean(C),
        "std_consumption": _std(C),
        "mean_labor": _mean(N),
        "std_labor": _std(N),
        "mean_consumption_utility": _mean(u_c),
        "mean_labor_disutility": _mean(labor),
        "mean_log_consumption": _mean(np.log(C)),
    }


def _flow_cev_pct(row: dict[str, float | str], ref: dict[str, float | str], params: BaselineParams) -> float:
    """Consumption-equivalent flow welfare relative to a reference row.

    The reported number is the permanent percentage change in reference
    consumption, holding reference labor fixed, that matches the row's mean
    period utility.  It is an ergodic flow measure, not a discounted transition
    welfare calculation.
    """

    W = float(row.get("mean_utility_flow", float("nan")))
    ref_labor = float(ref.get("mean_labor_disutility", float("nan")))
    sigma = float(params.sigma)
    if not np.isfinite(W) or not np.isfinite(ref_labor):
        return float("nan")
    target_consumption_utility = W + ref_labor
    if abs(sigma - 1.0) < 1e-10:
        ref_log_c = float(ref.get("mean_log_consumption", float("nan")))
        if not np.isfinite(ref_log_c):
            return float("nan")
        scale = np.exp(target_consumption_utility - ref_log_c)
    else:
        ref_uc = float(ref.get("mean_consumption_utility", float("nan")))
        if not np.isfinite(ref_uc) or abs(ref_uc) < 1e-14:
            return float("nan")
        ratio = target_consumption_utility / ref_uc
        if ratio <= 0.0 or not np.isfinite(ratio):
            return float("nan")
        scale = ratio ** (1.0 / (1.0 - sigma))
    return 100.0 * (float(scale) - 1.0)


def _annotate_cev(
    rows: list[dict[str, float | str]],
    *,
    params: BaselineParams,
    reference_experiment: str,
    reference_policy: str,
) -> list[dict[str, float | str]]:
    ref = next(
        (
            row
            for row in rows
            if row.get("experiment") == reference_experiment and row.get("policy") == reference_policy
        ),
        None,
    )
    for row in rows:
        row["flow_cev_reference"] = f"{reference_experiment}:{reference_policy}"
        row["flow_cev_vs_reference_pct"] = float("nan") if ref is None else _flow_cev_pct(row, ref, params)
    return rows


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
    utility_parts = _utility_parts(defs, params)
    return {
        "experiment": experiment,
        "policy": policy,
        "policy_label": POLICY_LABELS.get(policy, policy),
        "mean_inflation_ann_pp": _mean(inflation_ann),
        "std_inflation_ann_pp": _std(inflation_ann),
        "mean_output_gap_log_pct": 100.0 * _mean(output_gap),
        "std_output_gap_log_pct": 100.0 * _std(output_gap),
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
        **utility_parts,
    }


def collect_policy_rows(base_root: Path, experiment: str) -> list[dict[str, float | str]]:
    root = _postprocess_dir(base_root, experiment)
    return collect_policy_rows_from_postprocess(root, experiment)


def collect_policy_rows_from_postprocess(root: Path, experiment_label: str) -> list[dict[str, float | str]]:
    params = _params_for_postprocess(root, experiment_label)
    rows: list[dict[str, float | str]] = []
    for policy in POLICIES:
        path = root / f"{policy}_definitions.npz"
        if not path.exists():
            continue
        rows.append(policy_moment_row(experiment=experiment_label, policy=policy, defs=_load_npz(path), params=params))
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


def _ir_value(data: dict[str, np.ndarray], scenario: str, variable: str) -> np.ndarray | None:
    value = data.get(f"{scenario}__{variable}")
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _half_life(response: np.ndarray) -> float:
    response = np.asarray(response, dtype=float)
    if response.size == 0:
        return float("nan")
    peak_idx = int(np.nanargmax(response))
    peak = float(response[peak_idx])
    if not np.isfinite(peak) or peak <= 0.0:
        return float("nan")
    threshold = 0.5 * peak
    tail = response[peak_idx:]
    hits = np.where(tail <= threshold)[0]
    if hits.size == 0:
        return float("nan")
    return float(hits[0])


def ir_peak_rows(base_root: Path, experiments: Iterable[str], *, policy: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for experiment in experiments:
        path = _postprocess_dir(base_root, experiment) / f"IR_{policy}_definitions.npz"
        if not path.exists():
            continue
        data = _load_npz(path)
        labels = [str(x) for x in data.get("labels", np.asarray([], dtype=str)).tolist()]
        if "no_event" not in labels:
            continue
        for scenario in labels:
            if scenario == "no_event":
                continue
            Pi = _ir_value(data, scenario, "Pi")
            Pi0 = _ir_value(data, "no_event", "Pi")
            chi = _ir_value(data, scenario, "chi")
            chi0 = _ir_value(data, "no_event", "chi")
            gap = _ir_value(data, scenario, "output_gap")
            gap0 = _ir_value(data, "no_event", "output_gap")
            repair = _ir_value(data, scenario, "I_A")
            repair0 = _ir_value(data, "no_event", "I_A")
            A = _ir_value(data, scenario, "A")
            A0 = _ir_value(data, "no_event", "A")
            R = _ir_value(data, scenario, "R")
            R0 = _ir_value(data, "no_event", "R")

            chi_response = None if chi is None or chi0 is None else chi - chi0
            row: dict[str, float | str] = {
                "experiment": experiment,
                "policy": policy,
                "scenario": scenario,
                "peak_inflation_response_ann_pp": (
                    float(np.nanmax(400.0 * (Pi - Pi0))) if Pi is not None and Pi0 is not None else float("nan")
                ),
                "peak_scarcity_rent_response": (
                    float(np.nanmax(chi_response)) if chi_response is not None else float("nan")
                ),
                "trough_output_gap_response_log_pct": (
                    float(np.nanmin(100.0 * (gap - gap0))) if gap is not None and gap0 is not None else float("nan")
                ),
                "cumulative_repair_response": (
                    float(np.nansum(repair - repair0)) if repair is not None and repair0 is not None else float("nan")
                ),
                "terminal_adaptation_response": (
                    float((A - A0)[-1]) if A is not None and A0 is not None and len(A) else float("nan")
                ),
                "peak_policy_rate_response_ann_pp": (
                    float(np.nanmax(400.0 * (R - R0))) if R is not None and R0 is not None else float("nan")
                ),
                "scarcity_response_half_life_periods": (
                    _half_life(chi_response) if chi_response is not None else float("nan")
                ),
            }
            rows.append(row)
    return rows


def numerical_diagnostic_rows(base_root: Path, experiment: str = "baseline") -> list[dict[str, float | str]]:
    root = experiment_root(base_root, experiment)
    paths = {
        "natural": root / "natural" / "natural_eval.json",
        "fixed": root / "fixed_taylor" / "fixed_eval.json",
        "ba": root / "modified_taylor" / "ba_eval.json",
        "discretion": root / "discretion" / "discretion_eval.json",
        "commitment": root / "commitment" / "commitment_eval.json",
    }
    rows: list[dict[str, float | str]] = []
    for policy, path in paths.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key, value in data.items():
            if isinstance(value, (int, float)):
                rows.append({"experiment": experiment, "policy": policy, "residual": key, "value": float(value)})
    return rows


def numerical_robustness_rows(base_root: Path, *, policy: str) -> list[dict[str, float | str]]:
    root = base_root / "numerical_robustness"
    if not root.exists():
        return []
    rows: list[dict[str, float | str]] = []
    for run_root in sorted(p for p in root.iterdir() if p.is_dir()):
        run_rows = [
            row
            for row in collect_policy_rows_from_postprocess(run_root / "postprocess", f"robustness:{run_root.name}")
            if row["policy"] == policy
        ]
        rows.extend(run_rows)
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
    baseline_rows = _annotate_cev(
        baseline_rows,
        params=BaselineParams(),
        reference_experiment="baseline",
        reference_policy="fixed",
    )
    table_2 = output_dir / "table_2_ergodic_moments_by_policy.csv"
    _write_csv(table_2, baseline_rows)
    written["table_2"] = str(table_2)

    counter_rows = []
    for experiment in TABLE_EXPERIMENTS["counterfactual"]:
        rows = [row for row in collect_policy_rows(base_root, experiment) if row["policy"] == policy_for_decomposition]
        counter_rows.extend(rows)
    counter_rows = _annotate_cev(
        counter_rows,
        params=BaselineParams(),
        reference_experiment="baseline",
        reference_policy=policy_for_decomposition,
    )
    table_3 = output_dir / f"table_3_counterfactual_decomposition_{policy_for_decomposition}.csv"
    _write_csv(table_3, counter_rows)
    written["table_3"] = str(table_3)

    sens_rows = []
    for experiment in TABLE_EXPERIMENTS["sensitivity"]:
        rows = [row for row in collect_policy_rows(base_root, experiment) if row["policy"] == policy_for_decomposition]
        sens_rows.extend(rows)
    sens_rows = _annotate_cev(
        sens_rows,
        params=BaselineParams(),
        reference_experiment="baseline",
        reference_policy=policy_for_decomposition,
    )
    table_4 = output_dir / f"table_4_sensitivity_summary_{policy_for_decomposition}.csv"
    _write_csv(table_4, sens_rows)
    written["table_4"] = str(table_4)

    experiment_table = output_dir / "experiment_overrides.csv"
    _write_csv(experiment_table, experiment_rows(EXPERIMENTS.keys()))
    written["experiment_overrides"] = str(experiment_table)

    peak_path = output_dir / f"table_5_irf_peak_responses_{policy_for_decomposition}.csv"
    peak_experiments = _unique(TABLE_EXPERIMENTS["counterfactual"] + TABLE_EXPERIMENTS["sensitivity"])
    _write_csv(peak_path, ir_peak_rows(base_root, peak_experiments, policy=policy_for_decomposition))
    written["table_5"] = str(peak_path)

    diagnostics_path = output_dir / "table_6_numerical_diagnostics.csv"
    _write_csv(diagnostics_path, numerical_diagnostic_rows(base_root, "baseline"))
    written["table_6"] = str(diagnostics_path)

    robustness_rows = numerical_robustness_rows(base_root, policy=policy_for_decomposition)
    robustness_rows = _annotate_cev(
        robustness_rows,
        params=BaselineParams(),
        reference_experiment="robustness:seed_321",
        reference_policy=policy_for_decomposition,
    )
    robustness_path = output_dir / f"table_7_numerical_robustness_{policy_for_decomposition}.csv"
    _write_csv(robustness_path, robustness_rows)
    written["table_7"] = str(robustness_path)
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
