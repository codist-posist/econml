from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SELECTED_PATH_VARIABLES = [
    "C",
    "Y",
    "Y_n",
    "Y_over_Y_n",
    "output_gap",
    "output_gap_pct_log",
    "Pi",
    "inflation_ann_pct",
    "R",
    "R_n_real",
    "policy_rate_ann_pct",
    "natural_rate_ann_pct",
    "policy_rate_gap_ann_pct",
    "policy_rate_minus_natural_ann_pct",
    "R_over_R_n",
    "R_standard",
    "R_over_R_standard",
    "chi",
    "M",
    "M_zero_rent",
    "cap_pressure_ratio",
    "cap_pressure_policy",
    "bottleneck_scarcity",
    "bottleneck_adjustment",
    "repair_margin_standard",
    "repair_support",
    "repair_adjustment",
    "Q_A",
    "repair_threshold",
    "repair_activation_ratio",
    "Q_over_threshold",
    "Q_over_repair_threshold",
    "I_A",
    "A",
    "resource",
    "price_index",
    "calvo_S",
    "calvo_F",
    "Q",
    "repair_KKT",
    "promise_S",
    "promise_F",
    "promise_Q",
]

VARIABLE_LABELS = {
    "C": "Consumption",
    "Y": "Output",
    "Y_n": "Natural output",
    "Y_over_Y_n": "Output / natural output",
    "output_gap": "Output gap, log",
    "output_gap_pct_log": "Output gap, log pct",
    "Pi": "Gross inflation",
    "inflation_ann_pct": "Inflation, annual pct",
    "R": "Gross policy rate",
    "R_n_real": "Natural real rate",
    "policy_rate_ann_pct": "Policy rate, annual pct",
    "natural_rate_ann_pct": "Natural rate, annual pct",
    "policy_rate_gap_ann_pct": "Policy-natural rate gap, annual log pp",
    "policy_rate_minus_natural_ann_pct": "Policy-natural rate spread, annual pp",
    "R_over_R_n": "Policy rate / natural rate",
    "R_standard": "Standard Taylor gross rate",
    "R_over_R_standard": "Implemented / standard rate",
    "chi": "Scarcity rent",
    "M": "Actual imported input",
    "M_zero_rent": "Desired imports at zero rent",
    "cap_pressure_ratio": "Cap pressure: desired / cap",
    "cap_pressure_policy": "Policy cap-pressure signal",
    "bottleneck_scarcity": "Bottleneck scarcity signal",
    "bottleneck_adjustment": "Bottleneck rate adjustment",
    "repair_margin_standard": "Repair margin before adjustment",
    "repair_support": "Repair-support signal",
    "repair_adjustment": "Repair rate adjustment",
    "Q_A": "Repair value",
    "repair_threshold": "Repair threshold",
    "repair_activation_ratio": "Repair activation: value / threshold",
    "Q_over_threshold": "Repair value / threshold",
    "Q_over_repair_threshold": "Repair value / threshold",
    "I_A": "Repair investment",
    "A": "Adaptation stock",
    "resource": "Resource residual",
    "price_index": "Price-index residual",
    "calvo_S": "Calvo S residual",
    "calvo_F": "Calvo F residual",
    "Q": "Repair-value residual",
    "repair_KKT": "Repair KKT residual",
    "promise_S": "Promise S",
    "promise_F": "Promise F",
    "promise_Q": "Promise Q",
}

POLICY_LABELS = {
    "fixed": "Fixed Taylor",
    "ba": "Natural-rate adjusted",
    "bottleneck": "Bottleneck Taylor",
    "repair_aware": "Repair-aware Taylor",
    "repair_support_aggressive": "Aggressive repair-support",
    "discretion": "Discretion",
    "commitment": "Commitment",
    "unknown": "Unknown policy",
}

VARIANT_LABELS = {
    "baseline_relief": "Baseline relief",
    "strong_relief": "Strong relief",
    "cost_0p015_trained": "Repair cost 0.015",
    "cost_0p020": "Repair cost 0.020",
    "cost_0p030": "Repair cost 0.030",
}

SCENARIO_LABELS = {
    "no_event": "No event",
    "D_1x": "Moderate disruption",
    "D_3x": "Severe disruption",
    "X_1x": "Relief shock",
    "D_1x_X_lag": "Moderate disruption, delayed relief",
    "D_3x_X_lag": "Severe disruption, delayed relief",
}


def pretty_variable(name: str) -> str:
    return VARIABLE_LABELS.get(str(name), str(name).replace("_", " "))


def pretty_policy(name: str) -> str:
    return POLICY_LABELS.get(str(name), str(name).replace("_", " ").title())


def pretty_variant(name: str) -> str:
    return VARIANT_LABELS.get(str(name), str(name).replace("_", " ").title())


def pretty_scenario(name: str) -> str:
    return SCENARIO_LABELS.get(str(name), str(name).replace("_", " "))


def finite_mean(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def finite_min(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else np.nan


def finite_max(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


@dataclass(frozen=True)
class ZipEntry:
    archive: str
    entry: str
    size: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_zip_entries(result_dir: Path) -> Iterable[ZipEntry]:
    for archive in iter_result_archives(result_dir):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                yield ZipEntry(archive.name, info.filename, int(info.file_size))


def iter_result_archives(result_dir: Path) -> list[Path]:
    skip_names = {
        "results_synthesis_outputs.zip",
        "outputs.zip",
    }
    archives = []
    for archive in sorted(result_dir.glob("*.zip")):
        name = archive.name.lower()
        if name in skip_names or "synthesis_outputs" in name:
            continue
        archives.append(archive)
    return archives


def archive_index(result_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = [entry.__dict__ for entry in iter_zip_entries(result_dir)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["suffix"] = df["entry"].map(lambda x: Path(str(x)).suffix.lower())
    df.to_csv(output_dir / "archive_file_index.csv", index=False)
    return df


def read_csv_from_zip(zip_path: Path, entry: str) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(entry) as fh:
                df = pd.read_csv(fh)
        df = df.loc[:, ~df.columns.duplicated()].copy()
    except Exception as exc:
        print(f"Skipping CSV {zip_path.name}:{entry}: {exc}")
        return None
    meta = pd.DataFrame(
        {"source_archive": zip_path.name, "source_entry": entry},
        index=df.index,
    )
    return pd.concat([meta, df.reset_index(drop=True)], axis=1)


def read_json_from_zip(zip_path: Path, entry: str) -> dict | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(entry) as fh:
                return json.loads(fh.read().decode("utf-8"))
    except Exception:
        return None


def collect_csv_tables(result_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    csv_rows = []
    buckets: dict[str, list[pd.DataFrame]] = {
        "steady_references": [],
        "scenario_summaries": [],
        "checkpoint_metrics": [],
        "regime_frequencies": [],
        "active_reference_scans": [],
        "mechanism_tables": [],
        "economic_irfs": [],
        "sss_moments": [],
        "sss_samples": [],
        "stochastic_paths": [],
        "fan_stats": [],
    }
    for zip_path in iter_result_archives(result_dir):
        with zipfile.ZipFile(zip_path) as zf:
            csv_entries = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        for entry in csv_entries:
            csv_rows.append({"archive": zip_path.name, "entry": entry})
            lower = entry.lower()
            df = None
            def load_once() -> pd.DataFrame | None:
                nonlocal df
                if df is None:
                    df = read_csv_from_zip(zip_path, entry)
                return df

            if "steady_reference" in lower or "steady" in lower and "reference" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["steady_references"].append(tab)
            if (
                "summary" in lower
                or "diagnostics" in lower
                or "key_scenario" in lower
                or "scenario_repair" in lower
            ):
                tab = load_once()
                if tab is not None:
                    buckets["scenario_summaries"].append(tab)
            if "checkpoint" in lower or "metrics" in lower or "top_residual" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["checkpoint_metrics"].append(tab)
            if "regime_frequenc" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["regime_frequencies"].append(tab)
            if "conditional_active_repair" in lower or "active_reference" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["active_reference_scans"].append(tab)
            if "mechanism_table" in lower or "decomposition" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["mechanism_tables"].append(tab)
            if "economic_irfs" in lower or "ir_paths_wide" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["economic_irfs"].append(tab)
            if "sss_moments" in lower or "taylor_sss_moments" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["sss_moments"].append(tab)
            if "sss_sample" in lower or "taylor_sss_sample" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["sss_samples"].append(tab)
            if "stochastic_poisson_paths" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["stochastic_paths"].append(tab)
            if "fan_stats" in lower:
                tab = load_once()
                if tab is not None:
                    buckets["fan_stats"].append(tab)

    csv_index = pd.DataFrame(csv_rows)
    csv_index.to_csv(output_dir / "csv_table_index.csv", index=False)

    out: dict[str, pd.DataFrame] = {}
    tables_dir = ensure_dir(output_dir / "tables")
    for name, dfs in buckets.items():
        if not dfs:
            out[name] = pd.DataFrame()
            continue
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        combined.to_csv(tables_dir / f"all_{name}.csv", index=False)
        out[name] = combined
    return out


def infer_policy_from_npz(archive: str, entry: str) -> str:
    text = f"{archive}/{entry}".lower()
    if "aggressive" in text or "repair_support_aggressive" in text:
        return "repair_support_aggressive"
    if "repair_aware" in text:
        return "repair_aware"
    if "bottleneck" in text:
        return "bottleneck"
    if "fixed" in text:
        return "fixed"
    if "ba_" in text or "/ba" in text or "modified" in text:
        return "ba"
    if "discretion" in text:
        return "discretion"
    if "commitment" in text:
        return "commitment"
    return "unknown"


def infer_calibration_from_archive(archive: str, entry: str) -> str:
    text = f"{archive}/{entry}".lower()
    if "strong_relief" in text:
        return "strong_relief"
    if "baseline_relief" in text:
        return "baseline_relief"
    if "no_relief" in text:
        return "no_relief"
    if "interior_repair" in text:
        return "repair_active"
    if "repair_active" in text:
        return "repair_active"
    if "threshold" in text or "cost_0p" in text:
        return "repair_threshold"
    return "baseline_or_saved"


def infer_checkpoint_from_entry(entry: str) -> str:
    text = str(entry).lower()
    for label in ["best", "final", "latest_step", "max_broad_i_a", "max_scenario_i_a"]:
        if label in text:
            return label
    stem = Path(str(entry)).stem.lower()
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if part == "step" and i + 1 < len(parts):
            return f"step_{parts[i + 1]}"
    return "saved"


def load_npz_from_zip(zip_path: Path, entry: str) -> dict[str, np.ndarray] | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(entry) as fh:
                raw = fh.read()
        data = np.load(io.BytesIO(raw), allow_pickle=True)
        return {key: data[key] for key in data.files}
    except Exception as exc:
        print(f"Skipping NPZ {zip_path.name}:{entry}: {exc}")
        return None


def enrich_npz_defs(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(data)
    if "Pi" in out and "inflation_ann_pct" not in out:
        out["inflation_ann_pct"] = (np.asarray(out["Pi"], dtype=float) ** 4 - 1.0) * 100.0
    if "R" in out and "policy_rate_ann_pct" not in out:
        out["policy_rate_ann_pct"] = (np.asarray(out["R"], dtype=float) ** 4 - 1.0) * 100.0
    if "R_n" in out and "R_n_real" not in out:
        out["R_n_real"] = np.asarray(out["R_n"], dtype=float)
    if "R_n_real" in out and "natural_rate_ann_pct" not in out:
        out["natural_rate_ann_pct"] = (np.asarray(out["R_n_real"], dtype=float) ** 4 - 1.0) * 100.0
    if "R" in out and "R_n_real" in out:
        r = np.asarray(out["R"], dtype=float)
        rn = np.asarray(out["R_n_real"], dtype=float)
        if "R_over_R_n" not in out:
            out["R_over_R_n"] = r / np.maximum(rn, 1e-12)
        if "policy_rate_gap_ann_pct" not in out:
            out["policy_rate_gap_ann_pct"] = 400.0 * np.log(np.maximum(r, 1e-12) / np.maximum(rn, 1e-12))
        if (
            "policy_rate_minus_natural_ann_pct" not in out
            and "policy_rate_ann_pct" in out
            and "natural_rate_ann_pct" in out
        ):
            out["policy_rate_minus_natural_ann_pct"] = (
                np.asarray(out["policy_rate_ann_pct"], dtype=float)
                - np.asarray(out["natural_rate_ann_pct"], dtype=float)
            )
    if "Y" in out and "Y_n" in out and "Y_over_Y_n" not in out:
        out["Y_over_Y_n"] = np.asarray(out["Y"], dtype=float) / np.maximum(np.asarray(out["Y_n"], dtype=float), 1e-12)
    if "Y" in out and "Y_n" in out and "output_gap" not in out:
        out["output_gap"] = np.log(np.maximum(np.asarray(out["Y"], dtype=float), 1e-12) / np.maximum(np.asarray(out["Y_n"], dtype=float), 1e-12))
    if "output_gap" in out and "output_gap_pct_log" not in out:
        out["output_gap_pct_log"] = 100.0 * np.asarray(out["output_gap"], dtype=float)
    if "R" in out and "R_standard" in out and "R_over_R_standard" not in out:
        out["R_over_R_standard"] = np.asarray(out["R"], dtype=float) / np.maximum(np.asarray(out["R_standard"], dtype=float), 1e-12)
    if "Q_A" in out and "repair_activation_ratio" in out and "Q_over_repair_threshold" not in out:
        out["Q_over_repair_threshold"] = out["repair_activation_ratio"]
    if "Q_A" in out and "repair_activation_ratio" in out and "Q_over_threshold" not in out:
        out["Q_over_threshold"] = out["repair_activation_ratio"]
    return out


def npz_to_summary_rows(
    *,
    archive: str,
    entry: str,
    data: dict[str, np.ndarray],
    pre_event_index: int | None = None,
) -> tuple[list[dict], list[dict]]:
    data = enrich_npz_defs(data)
    labels = [str(x) for x in np.asarray(data.get("labels", []))]
    if not labels:
        return [], []
    n_time = int(np.asarray(next(v for k, v in data.items() if k != "labels")).shape[0])
    event_idx = int(pre_event_index if pre_event_index is not None else min(5, max(0, n_time - 1)))
    if "D" in data:
        # The deterministic IRFs we generated usually store a few pre-event rows. Detect the first
        # nonzero D jump when possible so summaries remain aligned across old and new notebooks.
        d_arr = np.asarray(data["D"], dtype=float)
        if d_arr.ndim == 2 and "no_event" in labels:
            candidate_cols = [i for i, lab in enumerate(labels) if lab != "no_event"]
            jumps = []
            for i in candidate_cols:
                nz = np.where(np.abs(d_arr[:, i] - d_arr[:, labels.index("no_event")]) > 1e-12)[0]
                if nz.size:
                    jumps.append(int(nz[0]))
            if jumps:
                event_idx = min(jumps)

    policy = infer_policy_from_npz(archive, entry)
    calibration = infer_calibration_from_archive(archive, entry)
    checkpoint = infer_checkpoint_from_entry(entry)
    base_idx = labels.index("no_event") if "no_event" in labels else None

    summary_rows: list[dict] = []
    panel_rows: list[dict] = []
    for var in SELECTED_PATH_VARIABLES:
        if var not in data:
            continue
        arr = np.asarray(data[var], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(labels):
            continue
        base = arr[:, base_idx] if base_idx is not None else np.zeros(arr.shape[0])
        for j, scenario in enumerate(labels):
            series = arr[:, j]
            dev = series - base
            post = series[event_idx:]
            post_dev = dev[event_idx:]
            summary_rows.append(
                {
                    "source_archive": archive,
                    "source_entry": entry,
                    "policy": policy,
                    "calibration": calibration,
                    "checkpoint": checkpoint,
                    "scenario": scenario,
                    "variable": var,
                    "event_index": event_idx,
                    "event": float(series[event_idx]),
                    "event_dev": float(dev[event_idx]),
                    "post_mean": finite_mean(post),
                    "post_min": finite_min(post),
                    "post_max": finite_max(post),
                    "post_mean_dev": finite_mean(post_dev),
                    "post_min_dev": finite_min(post_dev),
                    "post_max_dev": finite_max(post_dev),
                    "terminal": float(series[-1]),
                    "terminal_dev": float(dev[-1]),
                }
            )
            for t, value in enumerate(series):
                panel_rows.append(
                    {
                        "source_archive": archive,
                        "source_entry": entry,
                        "policy": policy,
                        "calibration": calibration,
                        "checkpoint": checkpoint,
                        "scenario": scenario,
                        "variable": var,
                        "t": int(t - event_idx),
                        "value": float(value),
                        "deviation_from_no_event": float(dev[t]),
                    }
                )
    return summary_rows, panel_rows


def collect_npz_ir_paths(result_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog_rows: list[dict] = []
    summary_rows: list[dict] = []
    panel_rows: list[dict] = []
    for zip_path in iter_result_archives(result_dir):
        with zipfile.ZipFile(zip_path) as zf:
            npz_entries = [name for name in zf.namelist() if name.lower().endswith(".npz")]
        for entry in npz_entries:
            data = load_npz_from_zip(zip_path, entry)
            if not data or "labels" not in data:
                continue
            labels = [str(x) for x in np.asarray(data["labels"])]
            keys = sorted(k for k in data.keys() if k != "labels")
            shapes = {k: list(np.asarray(v).shape) for k, v in data.items() if k != "labels"}
            catalog_rows.append(
                {
                    "source_archive": zip_path.name,
                    "source_entry": entry,
                    "policy": infer_policy_from_npz(zip_path.name, entry),
                    "calibration": infer_calibration_from_archive(zip_path.name, entry),
                    "labels": "|".join(labels),
                    "n_keys": len(keys),
                    "keys": "|".join(keys),
                    "shapes_json": json.dumps(shapes),
                }
            )
            s_rows, p_rows = npz_to_summary_rows(archive=zip_path.name, entry=entry, data=data)
            summary_rows.extend(s_rows)
            panel_rows.extend(p_rows)

    catalog = pd.DataFrame(catalog_rows)
    summary = pd.DataFrame(summary_rows)
    panel = pd.DataFrame(panel_rows)
    tables_dir = ensure_dir(output_dir / "tables")
    catalog.to_csv(tables_dir / "ir_npz_catalog.csv", index=False)
    summary.to_csv(tables_dir / "ir_path_summary_long.csv", index=False)
    panel.to_csv(tables_dir / "ir_paths_panel_long.csv", index=False)
    return catalog, summary, panel


def _normalize_wide_ir_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "output_gap_log_pct": "output_gap_pct_log",
        "Q_over_repair_threshold": "Q_over_threshold",
        "cap_pressure": "cap_pressure_ratio",
        "R_n": "R_n_real",
        "natural_R": "R_n_real",
        "natural_rate": "R_n_real",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
        dev_old = f"{old}_dev"
        dev_new = f"{new}_dev"
        if dev_old in out.columns and dev_new not in out.columns:
            out = out.rename(columns={dev_old: dev_new})
    if "inflation_ann_pct" not in out.columns and "Pi" in out.columns:
        out["inflation_ann_pct"] = (pd.to_numeric(out["Pi"], errors="coerce") ** 4 - 1.0) * 100.0
    if "policy_rate_ann_pct" not in out.columns and "R" in out.columns:
        out["policy_rate_ann_pct"] = (pd.to_numeric(out["R"], errors="coerce") ** 4 - 1.0) * 100.0
    if "natural_rate_ann_pct" not in out.columns and "R_n_real" in out.columns:
        out["natural_rate_ann_pct"] = (pd.to_numeric(out["R_n_real"], errors="coerce") ** 4 - 1.0) * 100.0
    if "R" in out.columns and "R_n_real" in out.columns:
        r = pd.to_numeric(out["R"], errors="coerce")
        rn = pd.to_numeric(out["R_n_real"], errors="coerce")
        if "R_over_R_n" not in out.columns:
            out["R_over_R_n"] = r / rn.clip(lower=1e-12)
        if "policy_rate_gap_ann_pct" not in out.columns:
            out["policy_rate_gap_ann_pct"] = 400.0 * np.log(r.clip(lower=1e-12) / rn.clip(lower=1e-12))
        if (
            "policy_rate_minus_natural_ann_pct" not in out.columns
            and "policy_rate_ann_pct" in out.columns
            and "natural_rate_ann_pct" in out.columns
        ):
            out["policy_rate_minus_natural_ann_pct"] = (
                pd.to_numeric(out["policy_rate_ann_pct"], errors="coerce")
                - pd.to_numeric(out["natural_rate_ann_pct"], errors="coerce")
            )
    if "Y" in out.columns and "Y_n" in out.columns and "Y_over_Y_n" not in out.columns:
        out["Y_over_Y_n"] = pd.to_numeric(out["Y"], errors="coerce") / pd.to_numeric(out["Y_n"], errors="coerce").clip(lower=1e-12)
    if "output_gap_pct_log" not in out.columns and "output_gap" in out.columns:
        out["output_gap_pct_log"] = 100.0 * pd.to_numeric(out["output_gap"], errors="coerce")
    if "Q_over_threshold" not in out.columns and "repair_activation_ratio" in out.columns:
        out["Q_over_threshold"] = out["repair_activation_ratio"]
    if "Q_over_repair_threshold" not in out.columns and "Q_over_threshold" in out.columns:
        out["Q_over_repair_threshold"] = out["Q_over_threshold"]
    return out


def collect_csv_ir_paths(tables: dict[str, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    economic = tables.get("economic_irfs", pd.DataFrame())
    tables_dir = ensure_dir(output_dir / "tables")
    if economic.empty or not {"source_archive", "source_entry", "scenario", "t"}.issubset(economic.columns):
        empty = pd.DataFrame()
        empty.to_csv(tables_dir / "csv_ir_path_summary_long.csv", index=False)
        empty.to_csv(tables_dir / "csv_ir_paths_panel_long.csv", index=False)
        return empty, empty

    df = _normalize_wide_ir_columns(economic)
    df = df[df["scenario"].notna() & df["t"].notna()].copy()
    if df.empty:
        empty = pd.DataFrame()
        empty.to_csv(tables_dir / "csv_ir_path_summary_long.csv", index=False)
        empty.to_csv(tables_dir / "csv_ir_paths_panel_long.csv", index=False)
        return empty, empty

    if "checkpoint_label" in df.columns:
        checkpoint = df["checkpoint_label"].astype(str)
    elif "checkpoint" in df.columns:
        checkpoint = df["checkpoint"].astype(str)
    else:
        checkpoint = df["source_entry"].map(infer_checkpoint_from_entry)
    df["checkpoint"] = checkpoint.replace({"nan": "saved", "None": "saved"})
    df["policy"] = [infer_policy_from_npz(a, e) for a, e in zip(df["source_archive"], df["source_entry"])]
    df["calibration"] = [infer_calibration_from_archive(a, e) for a, e in zip(df["source_archive"], df["source_entry"])]
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df[df["t"].notna()].copy()

    meta_cols = ["source_archive", "source_entry", "policy", "calibration", "checkpoint", "scenario", "t"]
    rows: list[pd.DataFrame] = []
    for var in SELECTED_PATH_VARIABLES:
        if var not in df.columns:
            continue
        tmp = df[meta_cols + [var]].copy()
        tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
        if tmp[var].notna().sum() == 0:
            continue
        tmp = tmp.rename(columns={var: "value"})
        dev_col = f"{var}_dev"
        if dev_col in df.columns:
            tmp["deviation_from_no_event"] = pd.to_numeric(df[dev_col], errors="coerce")
        else:
            key = ["source_archive", "source_entry", "checkpoint", "t"]
            base = (
                tmp[tmp["scenario"].astype(str).eq("no_event")][key + ["value"]]
                .drop_duplicates(key)
                .rename(columns={"value": "base_value"})
            )
            tmp = tmp.merge(base, on=key, how="left")
            tmp["deviation_from_no_event"] = tmp["value"] - tmp["base_value"]
            tmp = tmp.drop(columns=["base_value"])
        tmp["variable"] = var
        rows.append(tmp)

    if not rows:
        empty = pd.DataFrame()
        empty.to_csv(tables_dir / "csv_ir_path_summary_long.csv", index=False)
        empty.to_csv(tables_dir / "csv_ir_paths_panel_long.csv", index=False)
        return empty, empty

    panel = pd.concat(rows, ignore_index=True, sort=False)
    panel["t"] = panel["t"].astype(int)

    summary_rows: list[dict] = []
    group_cols = ["source_archive", "source_entry", "policy", "calibration", "checkpoint", "scenario", "variable"]
    for keys, grp in panel.groupby(group_cols, dropna=False, sort=False):
        grp = grp.sort_values("t")
        post = grp[grp["t"] >= 0]
        if post.empty:
            post = grp
        event = post.iloc[(post["t"].abs()).argmin()]
        summary_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "event_index": int(event["t"]),
                "event": float(event["value"]),
                "event_dev": float(event["deviation_from_no_event"]),
                "post_mean": finite_mean(post["value"]),
                "post_min": finite_min(post["value"]),
                "post_max": finite_max(post["value"]),
                "post_mean_dev": finite_mean(post["deviation_from_no_event"]),
                "post_min_dev": finite_min(post["deviation_from_no_event"]),
                "post_max_dev": finite_max(post["deviation_from_no_event"]),
                "terminal": float(grp.iloc[-1]["value"]),
                "terminal_dev": float(grp.iloc[-1]["deviation_from_no_event"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "csv_ir_path_summary_long.csv", index=False)
    panel.to_csv(tables_dir / "csv_ir_paths_panel_long.csv", index=False)
    return summary, panel


def _plot_policy_panel(
    *,
    panel: pd.DataFrame,
    scenario: str,
    plot_vars: list[str],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    scen = panel[(panel["scenario"] == scenario) & (panel["variable"].isin(plot_vars))].copy()
    if scen.empty:
        return
    n = len(plot_vars)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 3.0 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, var in zip(axes, plot_vars):
        sub = scen[scen["variable"] == var]
        if sub.empty:
            ax.axis("off")
            continue
        for policy, grp in sub.groupby("policy", sort=False):
            grp = grp.sort_values("t")
            ax.plot(grp["t"], grp["deviation_from_no_event"], label=pretty_policy(str(policy)), lw=1.9)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_title(pretty_variable(var))
        ax.set_xlabel("Quarters from shock")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_named_panel(
    *,
    panel: pd.DataFrame,
    scenario: str,
    plot_vars: list[str],
    output_path: Path,
    title: str,
    value_col: str = "deviation_from_no_event",
) -> None:
    import matplotlib.pyplot as plt

    scen = panel[(panel["scenario"] == scenario) & (panel["variable"].isin(plot_vars))].copy()
    if scen.empty or "series_label" not in scen.columns:
        return
    n = len(plot_vars)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 3.0 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, var in zip(axes, plot_vars):
        sub = scen[scen["variable"] == var]
        if sub.empty:
            ax.axis("off")
            continue
        for label, grp in sub.groupby("series_label", sort=False):
            grp = grp.sort_values("t")
            ax.plot(grp["t"], grp[value_col], label=str(label), lw=1.9)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_title(pretty_variable(var))
        ax.set_xlabel("Quarters from shock")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ir_comparisons(summary: pd.DataFrame, panel: pd.DataFrame, output_dir: Path) -> None:
    if panel.empty:
        return

    fig_dir = ensure_dir(output_dir / "figures")
    policy_fig_dir = ensure_dir(fig_dir / "policy_comparison")
    plot_vars = [
        "policy_rate_ann_pct",
        "inflation_ann_pct",
        "output_gap_pct_log",
        "chi",
        "cap_pressure_ratio",
        "M_zero_rent",
        "M",
        "Q_A",
        "repair_activation_ratio",
        "I_A",
        "A",
    ]
    scenarios = ["D_1x", "D_3x", "D_3x_X_lag"]

    # Prefer the cleaner repair-active archives when duplicate policies exist. The older four-policy
    # comparison remains available in the panel table, but these figures prioritize mechanism runs.
    preferred_archives = [
        "fixed_taylor_interior_repair_probe.zip",
        "bottleneck_interior_repair_probe.zip",
        "repair_aware_interior_repair_probe.zip",
        "repair_support_aggressive_interior_probe.zip",
        "four_taylor_comparison.zip",
    ]
    order = {name: i for i, name in enumerate(preferred_archives)}
    panel = panel.copy()
    panel["archive_rank"] = panel["source_archive"].map(lambda x: order.get(str(x), 999))

    # Keep the first source per policy/variable/time after ranking.
    panel_ranked = panel.sort_values(["policy", "variable", "scenario", "t", "archive_rank"])
    panel_ranked = panel_ranked.drop_duplicates(["policy", "variable", "scenario", "t"], keep="first")
    taylor_policies = {"fixed", "ba", "bottleneck", "repair_aware", "repair_support_aggressive"}
    taylor_panel = panel_ranked[panel_ranked["policy"].isin(taylor_policies)].copy()

    for scenario in scenarios:
        _plot_policy_panel(
            panel=taylor_panel,
            scenario=scenario,
            plot_vars=plot_vars,
            output_path=policy_fig_dir / f"taylor_policy_comparison_{scenario}.png",
            title=f"Policy comparison: {pretty_scenario(scenario)} (deviation from no-event path)",
        )
        _plot_policy_panel(
            panel=panel_ranked,
            scenario=scenario,
            plot_vars=plot_vars,
            output_path=policy_fig_dir / f"all_saved_policy_diagnostic_{scenario}.png",
            title=f"Diagnostic overlay of all saved paths: {pretty_scenario(scenario)}",
        )

    # A compact heat-style summary for event responses.
    key = summary[
        (summary["scenario"].isin(scenarios))
        & (
            summary["variable"].isin(
                [
                    "policy_rate_ann_pct",
                    "natural_rate_ann_pct",
                    "policy_rate_gap_ann_pct",
                    "policy_rate_minus_natural_ann_pct",
                    "inflation_ann_pct",
                    "Y",
                    "Y_n",
                    "Y_over_Y_n",
                    "output_gap_pct_log",
                    "chi",
                    "cap_pressure_ratio",
                    "I_A",
                    "A",
                ]
            )
        )
    ].copy()
    if not key.empty:
        key = key.sort_values(["scenario", "variable", "policy", "source_archive"])
        key.to_csv(output_dir / "tables" / "key_event_response_summary.csv", index=False)
        key[key["policy"].isin(taylor_policies)].to_csv(output_dir / "tables" / "key_event_response_summary_taylor.csv", index=False)


def plot_optimal_policy_figures(panel: pd.DataFrame, output_dir: Path) -> None:
    if panel.empty:
        return
    fig_dir = ensure_dir(output_dir / "figures" / "optimal_policy")
    plot_vars = [
        "C",
        "Y",
        "output_gap_pct_log",
        "inflation_ann_pct",
        "policy_rate_ann_pct",
        "chi",
        "cap_pressure_ratio",
        "Q_A",
        "Q_over_threshold",
        "I_A",
        "A",
    ]

    baseline = panel[
        panel["source_archive"].isin(["discretion_no_repair_report.zip", "commitment_no_repair_report.zip"])
        & panel["source_entry"].astype(str).str.contains("economic_irfs_long_best", case=False, na=False)
    ].copy()
    if not baseline.empty:
        baseline["series_label"] = baseline["policy"].map(lambda x: f"{pretty_policy(str(x))}: baseline best")
        for scenario in ["D_1x", "D_3x", "D_3x_X_lag"]:
            _plot_named_panel(
                panel=baseline,
                scenario=scenario,
                plot_vars=plot_vars,
                output_path=fig_dir / f"optimal_baseline_no_repair_{scenario}.png",
                title=f"Optimal-policy baseline/no-repair diagnostics: {pretty_scenario(scenario)}",
            )

    explicit = panel[
        (
            panel["source_archive"].eq("discretion_explicit_ia_light.zip")
            & panel["source_entry"].astype(str).str.contains("best_ir_definitions|best_ir_paths_wide", case=False, regex=True, na=False)
        )
        | (
            panel["source_archive"].eq("commitment_multi_checkpoint_report.zip")
            & panel["source_entry"].astype(str).str.contains("best_ir_definitions|best_ir_paths_wide", case=False, regex=True, na=False)
        )
    ].copy()
    if not explicit.empty:
        explicit["series_label"] = explicit["policy"].map(lambda x: f"{pretty_policy(str(x))}: explicit $I_A$")
        for scenario in ["D_1x", "D_3x", "D_3x_X_lag"]:
            _plot_named_panel(
                panel=explicit,
                scenario=scenario,
                plot_vars=plot_vars,
                output_path=fig_dir / f"optimal_explicit_ia_{scenario}.png",
                title=f"Explicit repair-investment controls: {pretty_scenario(scenario)}",
            )

    active = panel[
        panel["source_archive"].isin(
            [
                "discretion_active_margin_probe.zip",
                "discretion_interior_repair_probe.zip",
                "discretion_constrained_active_repair_probe.zip",
                "discretion_active_repair_probe_interrupted_report.zip",
            ]
        )
        & panel["checkpoint"].astype(str).isin(["best", "step_00000500", "final", "saved"])
    ].copy()
    if not active.empty:
        def label_row(row: pd.Series) -> str:
            archive = str(row["source_archive"]).replace(".zip", "")
            checkpoint = str(row.get("checkpoint", "saved"))
            return f"{archive}: {checkpoint}"

        active["series_label"] = active.apply(label_row, axis=1)
        for scenario in ["D_1x", "D_3x", "D_3x_X_lag", "active_ref", "higher_D"]:
            _plot_named_panel(
                panel=active,
                scenario=scenario,
                plot_vars=plot_vars,
                output_path=fig_dir / f"optimal_active_region_diagnostic_{scenario}.png",
                title=f"Active-region optimal-policy diagnostics: {pretty_scenario(scenario)}",
            )


def _plot_wide_path_comparison(
    *,
    df: pd.DataFrame,
    group_col: str,
    group_values: list[str],
    scenario: str,
    title: str,
    output_path: Path,
    plot_vars: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plot_vars = plot_vars or [
        "policy_rate_ann_pct",
        "inflation_ann_pct",
        "output_gap_pct_log",
        "chi",
        "cap_pressure_ratio",
        "M_zero_rent",
        "M",
        "Q_A",
        "Q_over_repair_threshold",
        "I_A",
        "A",
    ]
    sub = df[(df["scenario"] == scenario) & (df[group_col].astype(str).isin([str(v) for v in group_values]))].copy()
    if sub.empty:
        return
    base = df[(df["scenario"] == "no_event") & (df[group_col].astype(str).isin([str(v) for v in group_values]))].copy()
    n = len(plot_vars)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 3.0 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, var in zip(axes, plot_vars):
        if var not in sub.columns:
            ax.axis("off")
            continue
        for value in group_values:
            path = sub[sub[group_col].astype(str) == str(value)].sort_values("t")
            if path.empty:
                continue
            y = path[var].astype(float).to_numpy()
            t = path["t"].astype(float).to_numpy()
            base_path = base[base[group_col].astype(str) == str(value)].sort_values("t")
            if not base_path.empty and var in base_path.columns and len(base_path) == len(path):
                y = y - base_path[var].astype(float).to_numpy()
            label = pretty_variant(value) if group_col == "variant" else pretty_policy(value)
            ax.plot(t, y, label=label, lw=1.9)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_title(pretty_variable(var))
        ax.set_xlabel("Quarters from shock")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_parameter_sensitivities(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    economic = tables.get("economic_irfs", pd.DataFrame())
    if economic.empty:
        return
    fig_dir = ensure_dir(output_dir / "figures")
    relief_dir = ensure_dir(fig_dir / "relief_sensitivity")
    threshold_dir = ensure_dir(fig_dir / "repair_threshold_sensitivity")
    economic = economic.copy()

    # These plots answer: within one policy rule, how does changing the relief process change the
    # same shock path?
    relief = economic[economic["source_archive"].astype(str).eq("relief_crowding_out_tests.zip")].copy()
    required = {"variant", "policy", "scenario", "t"}
    if required.issubset(relief.columns):
        for policy in ["fixed", "bottleneck"]:
            pol = relief[relief["policy"].astype(str).eq(policy)].copy()
            if pol.empty:
                continue
            for scenario in ["D_3x", "D_3x_X_lag"]:
                _plot_wide_path_comparison(
                    df=pol,
                    group_col="variant",
                    group_values=["baseline_relief", "strong_relief"],
                    scenario=scenario,
                    title=f"Relief sensitivity under {pretty_policy(policy)}: {pretty_scenario(scenario)}",
                    output_path=relief_dir / f"relief_sensitivity_{policy}_{scenario}.png",
                )

    # These plots answer: within one policy rule, how does changing repair cost change the same
    # shock path? This is a parameter sensitivity, not a policy comparison.
    threshold = economic[economic["source_archive"].astype(str).eq("repair_threshold_policy_tests.zip")].copy()
    if required.issubset(threshold.columns):
        for policy in ["fixed", "bottleneck", "repair_aware"]:
            pol = threshold[threshold["policy"].astype(str).eq(policy)].copy()
            if pol.empty:
                continue
            variants = [str(v) for v in pol["variant"].dropna().unique()]
            variants = sorted(variants, key=lambda v: {"cost_0p015_trained": 0, "cost_0p020": 1, "cost_0p030": 2}.get(v, 99))
            for scenario in ["D_1x", "D_3x", "D_3x_X_lag"]:
                _plot_wide_path_comparison(
                    df=pol,
                    group_col="variant",
                    group_values=variants,
                    scenario=scenario,
                    title=f"Repair-cost sensitivity under {pretty_policy(policy)}: {pretty_scenario(scenario)}",
                    output_path=threshold_dir / f"repair_cost_sensitivity_{policy}_{scenario}.png",
                )


def _rank_preferred_saved_paths(panel: pd.DataFrame) -> pd.DataFrame:
    preferred_archives = [
        "fixed_taylor_interior_repair_probe.zip",
        "bottleneck_interior_repair_probe.zip",
        "repair_aware_interior_repair_probe.zip",
        "repair_support_aggressive_interior_probe.zip",
        "four_taylor_comparison.zip",
        "discretion_no_repair_report.zip",
        "commitment_no_repair_report.zip",
        "discretion_explicit_ia_light.zip",
        "commitment_multi_checkpoint_report.zip",
    ]
    order = {name: i for i, name in enumerate(preferred_archives)}
    ranked = panel.copy()
    ranked["archive_rank"] = ranked["source_archive"].map(lambda x: order.get(str(x), 999))
    ranked = ranked.sort_values(["policy", "variable", "scenario", "t", "archive_rank"])
    return ranked.drop_duplicates(["policy", "variable", "scenario", "t"], keep="first")


def build_natural_benchmark_tables(panel: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = ensure_dir(output_dir / "tables")
    if panel.empty:
        pd.DataFrame().to_csv(tables_dir / "natural_benchmark_policy_stance_summary.csv", index=False)
        return

    needed = [
        "Y",
        "Y_n",
        "Y_over_Y_n",
        "output_gap_pct_log",
        "R",
        "R_n_real",
        "policy_rate_ann_pct",
        "natural_rate_ann_pct",
        "policy_rate_gap_ann_pct",
        "policy_rate_minus_natural_ann_pct",
        "Pi",
        "inflation_ann_pct",
        "chi",
        "cap_pressure_ratio",
        "I_A",
        "A",
    ]
    sub = panel[panel["variable"].isin(needed)].copy()
    if sub.empty:
        pd.DataFrame().to_csv(tables_dir / "natural_benchmark_policy_stance_summary.csv", index=False)
        return

    index_cols = ["source_archive", "source_entry", "policy", "calibration", "checkpoint", "scenario", "t"]
    value_wide = sub.pivot_table(index=index_cols, columns="variable", values="value", aggfunc="first")
    dev_wide = sub.pivot_table(index=index_cols, columns="variable", values="deviation_from_no_event", aggfunc="first")
    value_wide.columns = [str(c) for c in value_wide.columns]
    dev_wide.columns = [f"d_{c}" for c in dev_wide.columns]
    wide = value_wide.join(dev_wide, how="outer").reset_index()
    if wide.empty:
        pd.DataFrame().to_csv(tables_dir / "natural_benchmark_policy_stance_summary.csv", index=False)
        return

    def series(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    tracked = [
        "Y",
        "Y_n",
        "Y_over_Y_n",
        "output_gap_pct_log",
        "policy_rate_ann_pct",
        "natural_rate_ann_pct",
        "policy_rate_gap_ann_pct",
        "policy_rate_minus_natural_ann_pct",
        "inflation_ann_pct",
        "chi",
        "cap_pressure_ratio",
        "I_A",
        "A",
    ]
    rows: list[dict] = []
    group_cols = ["source_archive", "source_entry", "policy", "calibration", "checkpoint", "scenario"]
    for keys, grp in wide.groupby(group_cols, dropna=False, sort=False):
        grp = grp.sort_values("t")
        post = grp[grp["t"] >= 0].copy()
        if post.empty:
            post = grp.copy()
        event = post.iloc[(post["t"].abs()).argmin()]
        row = {
            **dict(zip(group_cols, keys)),
            "n_post": int(len(post)),
            "event_t": int(event["t"]),
            "has_Y_n": bool(series(post, "Y_n").notna().any()),
            "has_R_n_real": bool(series(post, "R_n_real").notna().any()),
        }
        for var in tracked:
            row[f"{var}.event"] = float(pd.to_numeric(pd.Series([event.get(var, np.nan)]), errors="coerce").iloc[0])
            row[f"{var}.post_mean"] = finite_mean(series(post, var))
            row[f"{var}.post_min"] = finite_min(series(post, var))
            row[f"{var}.post_max"] = finite_max(series(post, var))
            dvar = f"d_{var}"
            row[f"{var}.event_dev"] = float(pd.to_numeric(pd.Series([event.get(dvar, np.nan)]), errors="coerce").iloc[0])
            row[f"{var}.post_mean_dev"] = finite_mean(series(post, dvar))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "natural_benchmark_policy_stance_summary.csv", index=False)
    if not summary.empty:
        summary[
            summary["source_archive"].isin(
                [
                    "fixed_taylor_interior_repair_probe.zip",
                    "bottleneck_interior_repair_probe.zip",
                    "repair_aware_interior_repair_probe.zip",
                    "repair_support_aggressive_interior_probe.zip",
                    "four_taylor_comparison.zip",
                    "discretion_no_repair_report.zip",
                    "commitment_no_repair_report.zip",
                    "discretion_explicit_ia_light.zip",
                    "commitment_multi_checkpoint_report.zip",
                ]
            )
            & summary["scenario"].isin(["D_1x", "D_3x", "D_3x_X_lag"])
        ].to_csv(tables_dir / "main_text_natural_benchmark_summary.csv", index=False)


def plot_natural_benchmark_figures(panel: pd.DataFrame, output_dir: Path) -> None:
    if panel.empty:
        return
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(output_dir / "figures" / "natural_benchmark")
    ranked = _rank_preferred_saved_paths(panel)
    taylor_policies = {"fixed", "ba", "bottleneck", "repair_aware", "repair_support_aggressive"}
    natural_vars = [
        "Y",
        "Y_n",
        "Y_over_Y_n",
        "output_gap_pct_log",
        "policy_rate_ann_pct",
        "natural_rate_ann_pct",
        "policy_rate_gap_ann_pct",
        "inflation_ann_pct",
        "chi",
        "I_A",
        "A",
    ]

    def plot_group(df: pd.DataFrame, scenario: str, path: Path, title: str) -> None:
        scen = df[(df["scenario"].astype(str).eq(scenario)) & (df["variable"].isin(natural_vars))].copy()
        if scen.empty:
            return
        n = len(natural_vars)
        cols = 3
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(17, 3.0 * rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, var in zip(axes, natural_vars):
            sub = scen[scen["variable"].eq(var)]
            if sub.empty:
                ax.axis("off")
                continue
            for policy, grp in sub.groupby("policy", sort=False):
                grp = grp.sort_values("t")
                ax.plot(grp["t"], grp["value"], label=pretty_policy(str(policy)), lw=1.9)
            ax.axvline(0, color="0.5", lw=0.8)
            ax.axhline(0, color="0.75", lw=0.8)
            ax.set_title(pretty_variable(var))
            ax.set_xlabel("Quarters from shock")
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.25)
        for ax in axes[n:]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)))
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(path, dpi=180)
        plt.close(fig)

    for scenario in ["D_1x", "D_3x", "D_3x_X_lag"]:
        plot_group(
            ranked[ranked["policy"].isin(taylor_policies)].copy(),
            scenario,
            fig_dir / f"natural_benchmark_taylor_{scenario}.png",
            f"Natural benchmark and policy stance: Taylor rules, {pretty_scenario(scenario)}",
        )
        plot_group(
            ranked.copy(),
            scenario,
            fig_dir / f"natural_benchmark_all_saved_{scenario}.png",
            f"Natural benchmark and policy stance: all saved policies, {pretty_scenario(scenario)}",
        )


def build_derived_mechanism_tables(panel: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = ensure_dir(output_dir / "tables")
    if panel.empty:
        pd.DataFrame().to_csv(tables_dir / "derived_policy_mechanism_summary.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "conditional_policy_tightening_mechanism_summary.csv", index=False)
        return

    needed = [
        "R",
        "policy_rate_ann_pct",
        "Pi",
        "inflation_ann_pct",
        "chi",
        "M",
        "M_zero_rent",
        "cap_pressure_ratio",
        "Q_A",
        "Q_over_threshold",
        "repair_activation_ratio",
        "I_A",
        "A",
    ]
    sub = panel[panel["variable"].isin(needed)].copy()
    if sub.empty:
        return
    index_cols = ["source_archive", "source_entry", "policy", "calibration", "checkpoint", "scenario", "t"]
    value_wide = sub.pivot_table(index=index_cols, columns="variable", values="value", aggfunc="first")
    dev_wide = sub.pivot_table(index=index_cols, columns="variable", values="deviation_from_no_event", aggfunc="first")
    value_wide.columns = [str(c) for c in value_wide.columns]
    dev_wide.columns = [f"d_{c}" for c in dev_wide.columns]
    wide = value_wide.join(dev_wide, how="outer").reset_index()
    if wide.empty:
        return

    def col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(default, index=df.index, dtype=float)

    rows: list[dict] = []
    cond_rows: list[dict] = []
    group_cols = ["source_archive", "source_entry", "policy", "calibration", "checkpoint", "scenario"]
    for keys, grp in wide[~wide["scenario"].astype(str).eq("no_event")].groupby(group_cols, dropna=False, sort=False):
        grp = grp.sort_values("t")
        post = grp[grp["t"] >= 0].copy()
        if post.empty:
            post = grp.copy()
        event = post.iloc[(post["t"].abs()).argmin()]

        dR = col(post, "d_R")
        if dR.isna().all():
            dR = col(post, "d_policy_rate_ann_pct")
        dPi = col(post, "d_Pi")
        if dPi.isna().all():
            dPi = col(post, "d_inflation_ann_pct")
        dchi = col(post, "d_chi")
        dIA = col(post, "d_I_A")
        dM = col(post, "d_M")
        dM0 = col(post, "d_M_zero_rent")
        chi = col(post, "chi")
        cap_pressure = col(post, "cap_pressure_ratio")
        IA = col(post, "I_A")
        q_over = col(post, "Q_over_threshold")
        if q_over.isna().all():
            q_over = col(post, "repair_activation_ratio")

        binding = (chi > 1e-8) | (cap_pressure > 1.0 + 1e-8)
        repair_positive = IA > 1e-8
        repair_interior = (IA > 1e-8) & (IA < 0.03 - 1e-6)
        tightening = dR > 1e-10
        cond = binding & repair_interior & tightening

        key_dict = dict(zip(group_cols, keys))
        row = {
            **key_dict,
            "n_post": int(len(post)),
            "event_dR": float(col(pd.DataFrame([event]), "d_R").iloc[0]) if "d_R" in post.columns else np.nan,
            "event_d_policy_rate_ann_pct": float(col(pd.DataFrame([event]), "d_policy_rate_ann_pct").iloc[0]) if "d_policy_rate_ann_pct" in post.columns else np.nan,
            "event_dPi": float(col(pd.DataFrame([event]), "d_Pi").iloc[0]) if "d_Pi" in post.columns else np.nan,
            "event_d_inflation_ann_pct": float(col(pd.DataFrame([event]), "d_inflation_ann_pct").iloc[0]) if "d_inflation_ann_pct" in post.columns else np.nan,
            "event_dchi": float(col(pd.DataFrame([event]), "d_chi").iloc[0]) if "d_chi" in post.columns else np.nan,
            "event_dI_A": float(col(pd.DataFrame([event]), "d_I_A").iloc[0]) if "d_I_A" in post.columns else np.nan,
            "event_dM": float(col(pd.DataFrame([event]), "d_M").iloc[0]) if "d_M" in post.columns else np.nan,
            "event_dM_zero_rent": float(col(pd.DataFrame([event]), "d_M_zero_rent").iloc[0]) if "d_M_zero_rent" in post.columns else np.nan,
            "event_chi": float(col(pd.DataFrame([event]), "chi").iloc[0]) if "chi" in post.columns else np.nan,
            "event_cap_pressure_ratio": float(col(pd.DataFrame([event]), "cap_pressure_ratio").iloc[0]) if "cap_pressure_ratio" in post.columns else np.nan,
            "event_I_A": float(col(pd.DataFrame([event]), "I_A").iloc[0]) if "I_A" in post.columns else np.nan,
            "event_Q_over_threshold": float(col(pd.DataFrame([event]), "Q_over_threshold").iloc[0]) if "Q_over_threshold" in post.columns else np.nan,
            "mean_dR": finite_mean(dR),
            "mean_dPi": finite_mean(dPi),
            "mean_dchi": finite_mean(dchi),
            "mean_dI_A": finite_mean(dIA),
            "mean_dM": finite_mean(dM),
            "mean_dM_zero_rent": finite_mean(dM0),
            "binding_freq": float(np.nanmean(binding.astype(float))),
            "tightening_freq": float(np.nanmean(tightening.astype(float))),
            "repair_positive_freq": float(np.nanmean(repair_positive.astype(float))),
            "repair_interior_freq": float(np.nanmean(repair_interior.astype(float))),
            "Q_above_threshold_freq": float(np.nanmean((q_over > 1.0).astype(float))),
        }
        rows.append(row)

        cond_post = post[cond]
        if cond_post.empty:
            cond_rows.append(
                {
                    **key_dict,
                    "condition": "binding_and_interior_repair_and_policy_tightening",
                    "n_condition": 0,
                    "Pr_dPi_lt_0": np.nan,
                    "Pr_dchi_lt_0": np.nan,
                    "Pr_dI_A_lt_0": np.nan,
                    "Pr_joint_dPi_dchi_dI_A_lt_0": np.nan,
                    "E_dR": np.nan,
                    "E_dPi": np.nan,
                    "E_dchi": np.nan,
                    "E_dI_A": np.nan,
                    "E_dM": np.nan,
                    "E_dM_zero_rent": np.nan,
                }
            )
            continue
        idx = cond_post.index
        cdR, cdPi, cdchi, cdIA, cdM, cdM0 = dR.loc[idx], dPi.loc[idx], dchi.loc[idx], dIA.loc[idx], dM.loc[idx], dM0.loc[idx]
        cond_rows.append(
            {
                **key_dict,
                "condition": "binding_and_interior_repair_and_policy_tightening",
                "n_condition": int(len(cond_post)),
                "Pr_dPi_lt_0": float(np.nanmean((cdPi < 0).astype(float))),
                "Pr_dchi_lt_0": float(np.nanmean((cdchi < 0).astype(float))),
                "Pr_dI_A_lt_0": float(np.nanmean((cdIA < 0).astype(float))),
                "Pr_joint_dPi_dchi_dI_A_lt_0": float(np.nanmean(((cdPi < 0) & (cdchi < 0) & (cdIA < 0)).astype(float))),
                "E_dR": finite_mean(cdR),
                "E_dPi": finite_mean(cdPi),
                "E_dchi": finite_mean(cdchi),
                "E_dI_A": finite_mean(cdIA),
                "E_dM": finite_mean(cdM),
                "E_dM_zero_rent": finite_mean(cdM0),
            }
        )

    summary = pd.DataFrame(rows)
    conditional = pd.DataFrame(cond_rows)
    summary.to_csv(tables_dir / "derived_policy_mechanism_summary.csv", index=False)
    conditional.to_csv(tables_dir / "conditional_policy_tightening_mechanism_summary.csv", index=False)

    if not summary.empty:
        main = summary[
            summary["source_archive"].isin(
                [
                    "fixed_taylor_interior_repair_probe.zip",
                    "bottleneck_interior_repair_probe.zip",
                    "repair_aware_interior_repair_probe.zip",
                    "repair_support_aggressive_interior_probe.zip",
                    "discretion_no_repair_report.zip",
                    "commitment_no_repair_report.zip",
                    "discretion_explicit_ia_light.zip",
                    "commitment_multi_checkpoint_report.zip",
                ]
            )
            & summary["scenario"].isin(["D_1x", "D_3x", "D_3x_X_lag"])
        ].copy()
        main.to_csv(tables_dir / "main_text_policy_mechanism_summary.csv", index=False)


def build_steady_state_index(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    steady = tables.get("steady_references", pd.DataFrame())
    regimes = tables.get("regime_frequencies", pd.DataFrame())
    active = tables.get("active_reference_scans", pd.DataFrame())
    sss_moments = tables.get("sss_moments", pd.DataFrame())
    fan_stats = tables.get("fan_stats", pd.DataFrame())
    out_dir = ensure_dir(output_dir / "tables")

    if not steady.empty:
        cols = [c for c in ["source_archive", "source_entry", "reference", "checkpoint_label", "C", "Y", "Y_n", "output_gap", "Pi", "R", "chi", "cap_pressure_ratio", "Q_A", "repair_threshold", "Q_over_threshold", "I_A", "A"] if c in steady.columns]
        steady[cols].to_csv(out_dir / "unified_calm_steady_references.csv", index=False)

    if not regimes.empty:
        regimes.to_csv(out_dir / "unified_stochastic_regime_frequencies.csv", index=False)

    if not sss_moments.empty:
        sss_moments.to_csv(out_dir / "unified_stochastic_steady_state_moments.csv", index=False)

    if not fan_stats.empty:
        fan_stats.to_csv(out_dir / "unified_stochastic_fan_stats.csv", index=False)

    if not active.empty:
        # Keep only compact columns if available; otherwise the full table is already saved.
        cols = [
            c
            for c in [
                "source_archive",
                "source_entry",
                "D_multiple",
                "D",
                "A",
                "I_A",
                "Q_A",
                "required_Q_for_steady_A",
                "threshold_zero_I",
                "Q_over_required",
                "Q_over_zero_threshold",
                "cap_pressure_ratio",
                "chi",
                "C",
                "Y",
                "static_err",
            ]
            if c in active.columns
        ]
        active[cols].to_csv(out_dir / "unified_conditional_active_repair_references.csv", index=False)


def extract_archives_for_sss(result_dir: Path, output_dir: Path, archives: list[str]) -> dict[str, Path]:
    unpack_dir = ensure_dir(output_dir / "_unpacked_for_sss")
    out: dict[str, Path] = {}
    for archive_name in archives:
        archive_path = result_dir / archive_name
        if not archive_path.exists():
            continue
        target = unpack_dir / archive_path.stem
        if target.exists():
            shutil.rmtree(target)
        ensure_dir(target)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target)
        out[archive_name] = target
    return out


def run_taylor_sss(
    *,
    result_dir: Path,
    output_dir: Path,
    device: str,
    dtype_name: str,
    n_paths: int,
    total_steps: int,
    burnin: int,
    thin: int,
    qmc_nodes: int,
    seed: int,
) -> None:
    # Delayed imports keep the default table/figure build lightweight.
    import torch
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.critical_input_deqn.config import BaselineParams, QMCConfig, RULE_OUTPUT_NAMES
    from src.critical_input_deqn.episode import random_rule_step
    from src.critical_input_deqn.experiments import params_from_metadata
    from src.critical_input_deqn.natural_oracle import NaturalOracleNet
    from src.critical_input_deqn.postprocess import evaluate_rule_path, load_rule

    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    sss_archives = {
        "fixed": ("fixed_taylor_interior_repair_probe.zip", "fixed"),
        "bottleneck": ("bottleneck_interior_repair_probe.zip", "bottleneck"),
        "repair_aware": ("repair_aware_interior_repair_probe.zip", "repair_aware"),
        "repair_support_aggressive": ("repair_support_aggressive_interior_probe.zip", "repair_aware"),
    }
    extracted = extract_archives_for_sss(result_dir, output_dir, [v[0] for v in sss_archives.values()])
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    def calm_state(n: int, params: BaselineParams) -> torch.Tensor:
        D = torch.zeros(n, device=device, dtype=dtype)
        X = torch.zeros_like(D)
        ell_D = torch.full_like(D, float(params.log_bar_lambda_D))
        ell_X = torch.full_like(D, float(params.log_bar_lambda_X))
        log_Z = torch.full_like(D, float(-0.5 * params.sigma_z**2))
        A = torch.zeros_like(D)
        log_Delta_prev = torch.zeros_like(D)
        return torch.stack([D, X, ell_D, ell_X, log_Z, A, log_Delta_prev], dim=-1)

    moments_rows: list[dict] = []
    regime_rows: list[dict] = []
    sample_rows: list[pd.DataFrame] = []

    for label, (archive_name, load_policy) in sss_archives.items():
        run_dir = extracted.get(archive_name)
        if run_dir is None:
            print(f"Skipping Taylor SSS {label}: missing {archive_name}")
            continue
        run_config_path = run_dir / "run_config.json"
        ckpt_path = run_dir / "checkpoints" / f"{load_policy}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / f"{load_policy}.pt"
        if not run_config_path.exists() or not ckpt_path.exists():
            print(f"Skipping Taylor SSS {label}: missing run_config or checkpoint in {run_dir}")
            continue
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        params = params_from_metadata({"config": run_config}, fallback=BaselineParams())
        loaded = load_rule(ckpt_path, policy=load_policy, device=device, dtype=dtype)
        rule_net = loaded.net.eval()
        qmc_cfg = QMCConfig(n_train=qmc_nodes, n_val=qmc_nodes, seed=seed)
        natural_net = NaturalOracleNet(
            params=params,
            qmc_cfg=qmc_cfg,
            n_nodes=qmc_nodes,
            device=device,
            dtype=dtype,
            chunk_size=8192,
        ).eval()

        z = calm_state(n_paths, params)
        kept_states: list[torch.Tensor] = []
        with torch.no_grad():
            for t in range(total_steps):
                if t >= burnin and (t - burnin) % thin == 0:
                    kept_states.append(z.detach().clone())
                z = random_rule_step(z, rule_net, natural_net, policy=load_policy, params=params)

        if not kept_states:
            continue
        states = torch.stack(kept_states, dim=0)
        _, defs = evaluate_rule_path(states, policy=load_policy, rule_net=rule_net, natural_net=natural_net, params=params)

        # Indicator variables.
        cap_slack = np.asarray(defs.get("cap_slack", np.nan), dtype=float)
        chi = np.asarray(defs.get("chi", np.nan), dtype=float)
        cap_pressure = np.asarray(defs.get("cap_pressure_ratio", np.nan), dtype=float)
        I_A = np.asarray(defs.get("I_A", np.nan), dtype=float)
        q_over = np.asarray(defs.get("repair_activation_ratio", np.nan), dtype=float)
        indicators = {
            "cap_binding": (np.abs(cap_slack) <= 1e-3) & (chi > 1e-5),
            "chi_positive": chi > 1e-5,
            "cap_pressure_gt_1": cap_pressure > 1.0,
            "repair_positive": I_A > 1e-5,
            "repair_interior": (I_A > 1e-5) & (I_A < float(params.repair_capacity) - 1e-5),
            "Q_above_threshold": q_over > 1.0,
        }
        for name, arr in indicators.items():
            x = np.asarray(arr, dtype=float).reshape(-1)
            regime_rows.append({"run": label, "policy": label, "load_policy": load_policy, "indicator": name, "frequency": float(np.nanmean(x)), "n": int(np.isfinite(x).sum())})

        variables = [
            "D",
            "X",
            "C",
            "Y",
            "output_gap",
            "Pi",
            "R",
            "chi",
            "cap_pressure_ratio",
            "M",
            "M_zero_rent",
            "Q_A",
            "repair_activation_ratio",
            "I_A",
            "A",
        ]
        flat_for_sample: dict[str, np.ndarray] = {}
        for var in variables:
            if var not in defs:
                continue
            x = np.asarray(defs[var], dtype=float).reshape(-1)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            moments_rows.append(
                {
                    "run": label,
                    "policy": label,
                    "load_policy": load_policy,
                    "variable": var,
                    "n": int(x.size),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "p01": float(np.quantile(x, 0.01)),
                    "p05": float(np.quantile(x, 0.05)),
                    "p50": float(np.quantile(x, 0.50)),
                    "p95": float(np.quantile(x, 0.95)),
                    "p99": float(np.quantile(x, 0.99)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }
            )
            flat_for_sample[var] = x[: min(x.size, 20_000)]

        if flat_for_sample:
            sample = pd.DataFrame(flat_for_sample)
            sample.insert(0, "run", label)
            sample.insert(1, "policy", label)
            sample.insert(2, "load_policy", load_policy)
            sample_rows.append(sample)

            plot_vars = [v for v in ["D", "X", "Y", "Pi", "R", "chi", "cap_pressure_ratio", "Q_A", "repair_activation_ratio", "I_A", "A"] if v in flat_for_sample]
            n = len(plot_vars)
            cols = 3
            rows = int(math.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(14, 3.0 * rows))
            axes = np.asarray(axes).reshape(-1)
            for ax, var in zip(axes, plot_vars):
                x = flat_for_sample[var]
                if x.size:
                    lo, hi = np.nanquantile(x, [0.005, 0.995])
                    xx = x[(x >= lo) & (x <= hi)]
                    ax.hist(xx, bins=50, alpha=0.85)
                ax.set_title(var)
                ax.grid(alpha=0.25)
            for ax in axes[n:]:
                ax.axis("off")
            fig.suptitle(f"Taylor stochastic steady distribution: {label}")
            fig.tight_layout()
            fig.savefig(figures_dir / f"taylor_sss_histograms_{label}.png", dpi=160)
            plt.close(fig)

    pd.DataFrame(moments_rows).to_csv(tables_dir / "taylor_sss_moments.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(tables_dir / "taylor_sss_regime_frequencies.csv", index=False)
    if sample_rows:
        pd.concat(sample_rows, ignore_index=True, sort=False).to_csv(tables_dir / "taylor_sss_sample.csv", index=False)


def run_local_monetary_wedge_test(
    *,
    result_dir: Path,
    output_dir: Path,
    device: str,
    dtype_name: str,
    horizon: int,
    qmc_nodes: int,
    seed: int,
) -> None:
    """Evaluate a no-retraining local monetary wedge around saved repair-active Taylor paths.

    This is not a full monetary-shock DEQN. The saved rule network is kept fixed and the wedge is
    applied directly to the policy rate entering the repair KKT. The first-period controls are
    therefore held on the saved policy function; future differences come through the altered
    adaptation state. This is the clean no-retraining diagnostic for R -> Omega_A -> I_A.
    """

    import torch
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.critical_input_deqn.config import BaselineParams, QMCConfig, RULE_OUTPUT_NAMES
    from src.critical_input_deqn.economics import derive_rule, unpack_rule_state
    from src.critical_input_deqn.experiments import params_from_metadata
    from src.critical_input_deqn.monetary_shock import annualized_bp_to_log_quarterly, derive_rule_with_monetary_shock
    from src.critical_input_deqn.natural_oracle import NaturalOracleNet, natural_benchmark_outputs
    from src.critical_input_deqn.postprocess import (
        _add_common_ratios,
        _deterministic_physical_step,
        evaluate_rule_path,
        load_rule,
        simulate_rule_ir_scenarios,
    )
    from src.critical_input_deqn.transforms import decode_rule_outputs

    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    test_archives = {
        "fixed": ("fixed_taylor_interior_repair_probe.zip", "fixed"),
        "bottleneck": ("bottleneck_interior_repair_probe.zip", "bottleneck"),
        "repair_aware": ("repair_aware_interior_repair_probe.zip", "repair_aware"),
        "repair_support_aggressive": ("repair_support_aggressive_interior_probe.zip", "repair_aware"),
    }
    extracted = extract_archives_for_sss(result_dir, output_dir, [v[0] for v in test_archives.values()])
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures" / "local_monetary_wedge")

    torch.manual_seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    path_rows: list[dict] = []
    summary_rows: list[dict] = []
    candidate_rows: list[dict] = []
    shock_bp_values = [25.0, 100.0]
    variables = [
        "R",
        "policy_rate_ann_pct",
        "Pi",
        "inflation_ann_pct",
        "output_gap_pct_log",
        "chi",
        "M",
        "M_zero_rent",
        "cap_pressure_ratio",
        "Q_A",
        "repair_activation_ratio",
        "I_A",
        "A",
    ]

    def _need_natural_y(params: BaselineParams) -> bool:
        return abs(float(params.phi_y)) > 1e-14

    def evaluate_local_step(
        *,
        z: torch.Tensor,
        eps_R: torch.Tensor,
        rule_net,
        natural_net,
        params: BaselineParams,
        load_policy: str,
    ) -> dict[str, torch.Tensor]:
        st = unpack_rule_state(z)
        uses_natural_y_ref = _need_natural_y(params)
        if uses_natural_y_ref or load_policy.lower() == "ba":
            out_n = natural_benchmark_outputs(
                z[..., :6],
                natural_net,
                params=params,
                need_rate=load_policy.lower() == "ba",
            )
        else:
            y_ref = torch.full_like(z[..., 0], float(params.steady_state_output))
            out_n = {"Y_n": y_ref, "R_n_real": torch.full_like(y_ref, float(params.bar_R))}
        out = decode_rule_outputs(
            rule_net(z),
            RULE_OUTPUT_NAMES,
            params=params,
            y_ref=out_n["Y_n"] if uses_natural_y_ref else None,
        )
        drv = derive_rule_with_monetary_shock(
            st,
            out,
            params,
            Y_n=out_n["Y_n"],
            R_n=out_n["R_n_real"],
            policy=load_policy,
            eps_R=eps_R,
        )
        data: dict[str, torch.Tensor] = {}
        state_names = ["D", "X", "ell_D", "ell_X", "log_Z", "A", "log_Delta_prev"]
        for i, name in enumerate(state_names):
            data[name] = z[..., i]
        data.update({k: v for k, v in out.items() if k not in data})
        data.update({k: v for k, v in out_n.items() if k not in data})
        data.update({k: v for k, v in drv.items() if k not in data})
        data["Y_n"] = out_n["Y_n"]
        data = _add_common_ratios(data, params)
        if "Pi" in data:
            data["inflation_ann_pct"] = (data["Pi"].pow(4.0) - 1.0) * 100.0
        if "R" in data:
            data["policy_rate_ann_pct"] = (data["R"].pow(4.0) - 1.0) * 100.0
        if "output_gap" in data:
            data["output_gap_pct_log"] = 100.0 * data["output_gap"]
        return data

    def simulate_from_candidate(
        *,
        z0: torch.Tensor,
        eps0: float,
        rule_net,
        natural_net,
        params: BaselineParams,
        load_policy: str,
    ) -> dict[str, np.ndarray]:
        z = z0.clone()
        eps = torch.full((z.shape[0],), float(eps0), device=z.device, dtype=z.dtype)
        data_by_var: dict[str, list[float]] = {var: [] for var in variables}
        for _ in range(int(horizon)):
            data = evaluate_local_step(
                z=z,
                eps_R=eps,
                rule_net=rule_net,
                natural_net=natural_net,
                params=params,
                load_policy=load_policy,
            )
            for var in variables:
                if var in data:
                    data_by_var[var].append(float(data[var].detach().cpu().reshape(-1)[0]))
                else:
                    data_by_var[var].append(np.nan)
            z = _deterministic_physical_step(
                z,
                A_next=data["A_next"],
                Delta_next=data["Delta"],
                add_D=torch.zeros_like(eps),
                add_X=torch.zeros_like(eps),
                params=params,
            )
            eps = 0.5 * eps
        return {k: np.asarray(v, dtype=float) for k, v in data_by_var.items()}

    for run_label, (archive_name, load_policy) in test_archives.items():
        run_dir = extracted.get(archive_name)
        if run_dir is None:
            continue
        run_config_path = run_dir / "run_config.json"
        ckpt_path = run_dir / "checkpoints" / f"{load_policy}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / f"{load_policy}.pt"
        if not run_config_path.exists() or not ckpt_path.exists():
            continue
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        params = params_from_metadata({"config": run_config}, fallback=BaselineParams())
        loaded = load_rule(ckpt_path, policy=load_policy, device=device, dtype=dtype)
        rule_net = loaded.net.eval()
        qmc_cfg = QMCConfig(n_train=qmc_nodes, n_val=qmc_nodes, seed=seed)
        natural_net = NaturalOracleNet(
            params=params,
            qmc_cfg=qmc_cfg,
            n_nodes=qmc_nodes,
            device=device,
            dtype=dtype,
            chunk_size=8192,
        ).eval()

        labels, states = simulate_rule_ir_scenarios(
            policy=load_policy,
            rule_net=rule_net,
            natural_net=natural_net,
            params=params,
            burnin=5,
            horizon=80,
            presteps=5,
            relief_lag=8,
            device=device,
            dtype=dtype,
        )
        _, defs = evaluate_rule_path(
            states,
            policy=load_policy,
            rule_net=rule_net,
            natural_net=natural_net,
            params=params,
        )
        T = states.shape[0]
        t_grid = np.arange(T) - 5
        cap = float(params.repair_capacity)
        candidate_count = 0
        for scenario in ["D_1x", "D_3x", "D_3x_X_lag"]:
            if scenario not in labels:
                continue
            j = labels.index(scenario)
            chi = np.asarray(defs.get("chi", np.full((T, len(labels)), np.nan)))[:, j]
            ia = np.asarray(defs.get("I_A", np.full((T, len(labels)), np.nan)))[:, j]
            cap_pressure = np.asarray(defs.get("cap_pressure_ratio", np.full((T, len(labels)), np.nan)))[:, j]
            interior = (t_grid >= 0) & (chi > 1e-8) & (ia > 1e-8) & (ia < cap - 1e-6)
            interior_idx = np.where(interior)[0]
            fallback = False
            if interior_idx.size == 0:
                # Keep a diagnostic row but do not call this an interior-repair test.
                relaxed = (t_grid >= 0) & ((chi > 1e-8) | (cap_pressure > 1.0)) & (ia > 1e-8)
                interior_idx = np.where(relaxed)[0]
                fallback = bool(interior_idx.size > 0)
            candidate_rows.append(
                {
                    "run": run_label,
                    "policy": run_label,
                    "load_policy": load_policy,
                    "scenario": scenario,
                    "n_strict_interior_candidates": int(np.sum(interior)),
                    "n_relaxed_positive_candidates": int(interior_idx.size) if fallback else int(np.sum(interior)),
                    "used_relaxed_positive_candidate": bool(fallback),
                    "max_chi": finite_max(chi),
                    "max_I_A": finite_max(ia),
                    "max_cap_pressure_ratio": finite_max(cap_pressure),
                }
            )
            if interior_idx.size == 0:
                continue
            idx = int(interior_idx[np.nanargmax(cap_pressure[interior_idx])])
            candidate_count += 1
            z0 = states[idx, j : j + 1, :].to(device=device, dtype=dtype)
            base_path = simulate_from_candidate(
                z0=z0,
                eps0=0.0,
                rule_net=rule_net,
                natural_net=natural_net,
                params=params,
                load_policy=load_policy,
            )
            for shock_bp in shock_bp_values:
                eps0 = annualized_bp_to_log_quarterly(shock_bp)
                shock_path = simulate_from_candidate(
                    z0=z0,
                    eps0=eps0,
                    rule_net=rule_net,
                    natural_net=natural_net,
                    params=params,
                    load_policy=load_policy,
                )
                for var in variables:
                    base = base_path.get(var, np.full(horizon, np.nan))
                    shock = shock_path.get(var, np.full(horizon, np.nan))
                    delta = shock - base
                    for t in range(int(horizon)):
                        path_rows.append(
                            {
                                "run": run_label,
                                "policy": run_label,
                                "load_policy": load_policy,
                                "scenario": scenario,
                                "candidate_t": int(t_grid[idx]),
                                "candidate_type": "relaxed_positive" if fallback else "strict_interior",
                                "shock_bp_annualized": float(shock_bp),
                                "t": int(t),
                                "variable": var,
                                "baseline": float(base[t]),
                                "shock": float(shock[t]),
                                "delta": float(delta[t]),
                            }
                        )
                def get_delta(var: str) -> np.ndarray:
                    return shock_path.get(var, np.full(horizon, np.nan)) - base_path.get(var, np.full(horizon, np.nan))

                dR = get_delta("R")
                dPi = get_delta("Pi")
                dchi = get_delta("chi")
                dIA = get_delta("I_A")
                dM = get_delta("M")
                dM0 = get_delta("M_zero_rent")
                post = np.arange(int(horizon)) >= 0
                summary_rows.append(
                    {
                        "run": run_label,
                        "policy": run_label,
                        "load_policy": load_policy,
                        "scenario": scenario,
                        "candidate_t": int(t_grid[idx]),
                        "candidate_type": "relaxed_positive" if fallback else "strict_interior",
                        "shock_bp_annualized": float(shock_bp),
                        "baseline_chi_event": float(base_path["chi"][0]),
                        "baseline_I_A_event": float(base_path["I_A"][0]),
                        "baseline_cap_pressure_event": float(base_path["cap_pressure_ratio"][0]),
                        "delta_R_event": float(dR[0]),
                        "delta_Pi_event": float(dPi[0]),
                        "delta_chi_event": float(dchi[0]),
                        "delta_I_A_event": float(dIA[0]),
                        "delta_M_event": float(dM[0]),
                        "delta_M_zero_rent_event": float(dM0[0]),
                        "mean_delta_R": finite_mean(dR[post]),
                        "mean_delta_Pi": finite_mean(dPi[post]),
                        "mean_delta_chi": finite_mean(dchi[post]),
                        "mean_delta_I_A": finite_mean(dIA[post]),
                        "mean_delta_M": finite_mean(dM[post]),
                        "mean_delta_M_zero_rent": finite_mean(dM0[post]),
                        "Pr_delta_Pi_lt_0": float(np.nanmean((dPi[post] < 0).astype(float))),
                        "Pr_delta_chi_lt_0": float(np.nanmean((dchi[post] < 0).astype(float))),
                        "Pr_delta_I_A_lt_0": float(np.nanmean((dIA[post] < 0).astype(float))),
                        "Pr_joint_delta_Pi_chi_I_A_lt_0": float(
                            np.nanmean(((dPi[post] < 0) & (dchi[post] < 0) & (dIA[post] < 0)).astype(float))
                        ),
                    }
                )

        if candidate_count == 0:
            print(f"Local monetary wedge: no positive repair candidates found for {run_label}.")

    candidates = pd.DataFrame(candidate_rows)
    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(path_rows)
    candidates.to_csv(tables_dir / "local_monetary_wedge_candidates.csv", index=False)
    summary.to_csv(tables_dir / "local_monetary_wedge_summary.csv", index=False)
    paths.to_csv(tables_dir / "local_monetary_wedge_paths_long.csv", index=False)

    if not paths.empty:
        plot_vars = [
            "policy_rate_ann_pct",
            "inflation_ann_pct",
            "output_gap_pct_log",
            "chi",
            "M",
            "M_zero_rent",
            "cap_pressure_ratio",
            "Q_A",
            "repair_activation_ratio",
            "I_A",
            "A",
        ]
        for (run_label, scenario, shock_bp), grp in paths.groupby(["run", "scenario", "shock_bp_annualized"], sort=False):
            if float(shock_bp) != 100.0:
                continue
            n = len(plot_vars)
            cols = 3
            rows = int(math.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(17, 3.0 * rows), sharex=True)
            axes = np.asarray(axes).reshape(-1)
            for ax, var in zip(axes, plot_vars):
                sub = grp[grp["variable"] == var].sort_values("t")
                if sub.empty:
                    ax.axis("off")
                    continue
                ax.plot(sub["t"], sub["delta"], lw=1.9)
                ax.axhline(0, color="0.75", lw=0.8)
                ax.set_title(pretty_variable(var))
                ax.set_xlabel("Quarters after local monetary wedge")
                ax.tick_params(axis="x", labelbottom=True)
                ax.grid(alpha=0.25)
            for ax in axes[n:]:
                ax.axis("off")
            fig.suptitle(f"Local monetary wedge, {run_label}, {pretty_scenario(str(scenario))}, 100bp")
            fig.tight_layout(rect=(0, 0.02, 1, 0.96))
            fig.savefig(figures_dir / f"local_monetary_wedge_{run_label}_{scenario}_100bp.png", dpi=180)
            plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified result tables and figures from result_learning zip archives.")
    parser.add_argument("--result-dir", type=Path, default=Path("result_learning"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-taylor-sss", action="store_true", help="Also simulate stochastic steady-state distributions for saved Taylor-rule checkpoints.")
    parser.add_argument("--run-local-monetary-test", action="store_true", help="Also run a no-retraining local monetary wedge test around saved repair-active Taylor paths.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--sss-paths", type=int, default=256)
    parser.add_argument("--sss-steps", type=int, default=800)
    parser.add_argument("--sss-burnin", type=int, default=200)
    parser.add_argument("--sss-thin", type=int, default=4)
    parser.add_argument("--sss-qmc-nodes", type=int, default=24)
    parser.add_argument("--local-monetary-horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or (result_dir / "synthesis_outputs")).resolve()
    ensure_dir(output_dir)
    print("Result dir:", result_dir)
    print("Output dir:", output_dir)

    idx = archive_index(result_dir, output_dir)
    print(f"Indexed {len(idx)} files inside result archives.")

    tables = collect_csv_tables(result_dir, output_dir)
    for name, df in tables.items():
        if not df.empty:
            print(f"Collected {name}: {df.shape}")

    catalog, npz_summary, npz_panel = collect_npz_ir_paths(result_dir, output_dir)
    csv_summary, csv_panel = collect_csv_ir_paths(tables, output_dir)
    summary = pd.concat([npz_summary, csv_summary], ignore_index=True, sort=False)
    panel = pd.concat([npz_panel, csv_panel], ignore_index=True, sort=False)
    if not summary.empty:
        summary.to_csv(output_dir / "tables" / "combined_ir_path_summary_long.csv", index=False)
    if not panel.empty:
        panel.to_csv(output_dir / "tables" / "combined_ir_paths_panel_long.csv", index=False)
    print("IR NPZ catalog:", catalog.shape)
    print("IR NPZ summary:", npz_summary.shape)
    print("IR NPZ panel:", npz_panel.shape)
    print("CSV IR summary:", csv_summary.shape)
    print("CSV IR panel:", csv_panel.shape)
    print("Combined IR summary:", summary.shape)
    print("Combined IR panel:", panel.shape)
    plot_ir_comparisons(summary, panel, output_dir)
    plot_optimal_policy_figures(panel, output_dir)
    plot_parameter_sensitivities(tables, output_dir)
    build_derived_mechanism_tables(panel, output_dir)
    build_natural_benchmark_tables(panel, output_dir)
    plot_natural_benchmark_figures(panel, output_dir)
    build_steady_state_index(tables, output_dir)

    if args.run_taylor_sss:
        run_taylor_sss(
            result_dir=result_dir,
            output_dir=output_dir,
            device=args.device,
            dtype_name=args.dtype,
            n_paths=args.sss_paths,
            total_steps=args.sss_steps,
            burnin=args.sss_burnin,
            thin=args.sss_thin,
            qmc_nodes=args.sss_qmc_nodes,
            seed=args.seed,
        )

    if args.run_local_monetary_test:
        run_local_monetary_wedge_test(
            result_dir=result_dir,
            output_dir=output_dir,
            device=args.device,
            dtype_name=args.dtype,
            horizon=args.local_monetary_horizon,
            qmc_nodes=args.sss_qmc_nodes,
            seed=args.seed,
        )

    manifest = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "run_taylor_sss": bool(args.run_taylor_sss),
        "run_local_monetary_test": bool(args.run_local_monetary_test),
        "files": sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done. Manifest:", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
