from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.critical_input_deqn.notebook_diagnostics import rule_ir_mechanism_diagnostics


def resolve_repo_root(preferred: str | Path = "/content/econml") -> Path:
    """Return the repository root in Colab or in a local checkout."""

    candidates = [Path(preferred), Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        if (candidate / "src" / "critical_input_deqn").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find repo root containing src/critical_input_deqn.")


def run_stream(cmd: Sequence[object], *, cwd: Path) -> None:
    """Run a command and stream stdout/stderr into the notebook."""

    printable = " ".join(map(str, cmd))
    print("Running:")
    print(printable)
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, list(map(str, cmd)))


def write_params_json(path: Path, params: Mapping[str, float | int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(dict(params), fh, indent=2, sort_keys=True)
    return path


def train_rule_variant(
    *,
    root: Path,
    output_dir: Path,
    params: Mapping[str, float | int],
    policies: Sequence[str],
    run_train: bool = True,
    retrain: bool = False,
    rule_steps: int = 2500,
    qmc_train: int = 128,
    qmc_val: int = 256,
    n_val_states: int = 512,
    stop_val_states: int = 256,
    hidden_width: int = 192,
    hidden_depth: int = 2,
    batch_size: int = 1024,
    sim_batch_size: int = 256,
    dtype: str = "float64",
    device: str | None = None,
    natural_oracle_nodes: int = 16,
    natural_oracle_chunk_size: int = 8192,
    checkpoint_every: int = 500,
    checkpoint_keep: int = 6,
    log_every: int = 100,
    scenario_q_weight: float = 15.0,
    calm_anchor_weight: float = 2.0,
    calm_residual_weight: float = 2.0,
    scenario_loss_interval: int = 25,
) -> Path:
    """Train one rule-policy variant unless its requested checkpoints already exist."""

    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = write_params_json(output_dir / "params.json", params)
    missing = [
        policy
        for policy in policies
        if not (output_dir / "checkpoints" / f"{policy}_best.pt").exists()
        and not (output_dir / f"{policy}.pt").exists()
    ]
    if not run_train:
        print(f"RUN_TRAIN=False; skipping {output_dir}.")
        return output_dir
    if missing == [] and not retrain:
        print(f"All requested checkpoints already exist in {output_dir}; skipping.")
        return output_dir

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.critical_input_deqn.run_train",
        "--output-dir",
        output_dir,
        "--policies",
        ",".join(policies),
        "--params-json",
        params_path,
        "--natural-benchmark",
        "oracle",
        "--skip-natural-network",
        "--natural-oracle-nodes",
        natural_oracle_nodes,
        "--natural-oracle-chunk-size",
        natural_oracle_chunk_size,
        "--rule-steps",
        rule_steps,
        "--rule-trainer",
        "episode",
        "--qmc-train",
        qmc_train,
        "--qmc-val",
        qmc_val,
        "--n-val-states",
        n_val_states,
        "--stop-val-states",
        stop_val_states,
        "--hidden-width",
        hidden_width,
        "--hidden-depth",
        hidden_depth,
        "--device",
        device,
        "--dtype",
        dtype,
        "--batch-size",
        batch_size,
        "--sim-batch-size",
        sim_batch_size,
        "--episode-length",
        20,
        "--episode-updates-per-episode",
        2,
        "--episode-broad-share",
        0.5,
        "--checkpoint-every",
        checkpoint_every,
        "--checkpoint-keep",
        checkpoint_keep,
        "--log-every",
        log_every,
        "--rule-scenario-q-weight",
        scenario_q_weight,
        "--rule-calm-anchor-weight",
        calm_anchor_weight,
        "--rule-calm-residual-weight",
        calm_residual_weight,
        "--rule-scenario-burnin",
        5,
        "--rule-scenario-horizon",
        10,
        "--rule-scenario-loss-interval",
        scenario_loss_interval,
        "--target-scenario-q-rms",
        0.01,
    ]
    run_stream(cmd, cwd=root)
    return output_dir


def _derive_series(defs_by_label: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for scenario, data in defs_by_label.items():
        series = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        if "R" in series:
            series["policy_rate_ann_pct"] = (series["R"] ** 4 - 1.0) * 100.0
        if "Pi" in series:
            series["inflation_ann_pct"] = (series["Pi"] ** 4 - 1.0) * 100.0
        if "output_gap" in series:
            series["output_gap_pct_log"] = 100.0 * series["output_gap"]
        out[scenario] = series
    return out


def collect_rule_ir_diagnostics(
    *,
    artifact_root: Path,
    runs: Sequence[Mapping[str, object]],
    report_dir: Path,
    device: str | None = None,
    dtype: str = "float64",
    burnin: int = 80,
    horizon: int = 100,
    presteps: int = 5,
    relief_lag: int = 8,
    natural_oracle_nodes: int = 16,
    natural_oracle_chunk_size: int = 8192,
    plot_each: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Load policy checkpoints, compute IR summaries, and save tidy CSV diagnostics."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    series_by_run: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    variables = [
        "policy_rate_ann_pct",
        "inflation_ann_pct",
        "output_gap_pct_log",
        "R",
        "R_over_R_standard",
        "bottleneck_adjustment",
        "repair_adjustment",
        "repair_support",
        "chi",
        "cap_pressure_ratio",
        "cap_pressure_policy",
        "M",
        "M_zero_rent",
        "Q_A",
        "Q_over_repair_threshold",
        "I_A",
        "A",
    ]

    for spec in runs:
        variant = str(spec["variant"])
        policy = str(spec["policy"])
        output_dir = Path(spec["output_dir"])
        checkpoint = spec.get("checkpoint")
        label = f"{variant}__{policy}"
        print("\n" + "=" * 100)
        print(label)
        print("=" * 100)
        labels, defs_by_label, table = rule_ir_mechanism_diagnostics(
            artifact_root=artifact_root,
            output_dir=output_dir,
            policy=policy,
            policy_checkpoint=checkpoint,
            device=device,
            dtype=dtype,
            burnin=burnin,
            horizon=horizon,
            presteps=presteps,
            relief_lag=relief_lag,
            plot=plot_each,
            natural_benchmark="oracle",
            natural_oracle_nodes=natural_oracle_nodes,
            natural_oracle_chunk_size=natural_oracle_chunk_size,
        )
        table = pd.DataFrame(table).copy()
        table.insert(0, "variant", variant)
        table.insert(1, "policy", policy)
        table.to_csv(report_dir / f"{label}_ir_mechanism_table.csv", index=False)
        all_tables.append(table)

        derived = _derive_series(defs_by_label)
        series_by_run[label] = derived

        wide_rows = []
        for scenario, data in derived.items():
            n = len(next(iter(data.values())))
            for t in range(n):
                row = {
                    "variant": variant,
                    "policy": policy,
                    "run": label,
                    "scenario": scenario,
                    "t": t - presteps,
                }
                for key in variables:
                    if key in data:
                        row[key] = float(data[key][t])
                wide_rows.append(row)
        pd.DataFrame(wide_rows).to_csv(report_dir / f"{label}_ir_paths_wide.csv", index=False)

        for scenario in ["D_1x", "D_3x", "D_3x_X_lag"]:
            if scenario not in derived:
                continue
            data = derived[scenario]
            row: dict[str, object] = {
                "variant": variant,
                "policy": policy,
                "run": label,
                "scenario": scenario,
            }
            for key in variables:
                if key not in data:
                    continue
                arr = np.asarray(data[key], dtype=float)
                post = arr[presteps:]
                row[f"{key}.event"] = float(arr[presteps])
                row[f"{key}.post_mean"] = float(np.nanmean(post))
                row[f"{key}.post_min"] = float(np.nanmin(post))
                row[f"{key}.post_max"] = float(np.nanmax(post))
            if "I_A" in data:
                ia = np.asarray(data["I_A"], dtype=float)[presteps:]
                row["repair_positive_freq"] = float(np.mean(ia > 1e-5))
                row["repair_interior_freq"] = float(np.mean((ia > 1e-5) & (ia < 0.03 - 1e-5)))
                row["repair_upper_freq"] = float(np.mean(ia >= 0.03 - 1e-5))
            summary_rows.append(row)

    detail = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(report_dir / "all_ir_mechanism_tables.csv", index=False)
    summary.to_csv(report_dir / "key_scenario_summary.csv", index=False)
    return detail, summary, series_by_run


def plot_run_comparison(
    series_by_run: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    report_dir: Path,
    scenario: str = "D_3x",
    variables: Sequence[str] = (
        "policy_rate_ann_pct",
        "inflation_ann_pct",
        "output_gap_pct_log",
        "chi",
        "cap_pressure_ratio",
        "Q_A",
        "Q_over_repair_threshold",
        "I_A",
        "A",
    ),
    deviation_from_no_event: bool = True,
    filename: str | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    variables = list(variables)
    ncols = 3
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, var in zip(axes, variables):
        for run_label, run_data in series_by_run.items():
            if scenario not in run_data or var not in run_data[scenario]:
                continue
            y = np.asarray(run_data[scenario][var], dtype=float)
            if deviation_from_no_event and "no_event" in run_data and var in run_data["no_event"]:
                y = y - np.asarray(run_data["no_event"][var], dtype=float)
            t = np.arange(len(y))
            ax.plot(t, y, label=run_label)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.grid(alpha=0.25)
        suffix = " dev" if deviation_from_no_event else ""
        ax.set_title(f"{var}{suffix}")
    for ax in axes[len(variables) :]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{scenario}: rule-policy comparison")
    fig.tight_layout()
    if filename is None:
        filename = f"{scenario}_comparison.png"
    path = report_dir / filename
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)
    return path


def zip_folder(folder: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", folder)
    print("Saved zip:", zip_path)
    print("Size MB:", zip_path.stat().st_size / 1e6)
    return zip_path
