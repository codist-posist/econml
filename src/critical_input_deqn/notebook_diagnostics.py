from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .config import BaselineParams, NetworkConfig, QMCConfig
from .experiments import params_from_metadata
from .natural_oracle import NaturalOracleNet
from .postprocess import (
    evaluate_optimal_path,
    evaluate_rule_path,
    simulate_optimal_ir_scenarios,
    simulate_rule_ir_scenarios,
)
from .train import (
    load_checkpoint,
    make_commitment_net,
    make_discretion_net,
    make_natural_net,
    make_rule_net,
)
from .qmc import make_qmc_nodes


def latest_step_checkpoint(folder: Path, name: str) -> Path | None:
    files = sorted((folder / "checkpoints").glob(f"{name}_step_*.pt"))
    return files[-1] if files else None


def first_existing(paths: Iterable[Path | str | None]) -> Path:
    tried: list[str] = []
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError("No existing path found:\n" + "\n".join(tried))


def maybe_first_existing(paths: Iterable[Path | str | None]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        if p.exists():
            return p
    return None


def _as_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).replace("torch.", "")
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _load_run_config(artifact_root: Path, output_dir: Path) -> tuple[dict, Path]:
    run_config_path = first_existing(
        [
            output_dir / "run_config.json",
            artifact_root / "run_config.json",
            artifact_root / "natural" / "run_config.json",
        ]
    )
    with run_config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh), run_config_path


def _params_and_net_cfg(
    run_config: dict,
    *,
    hidden_width: int,
    hidden_depth: int,
) -> tuple[BaselineParams, NetworkConfig]:
    params = params_from_metadata({"config": run_config}, fallback=BaselineParams())
    net_data = run_config.get("network", {})
    net_cfg = NetworkConfig(
        hidden_width=int(net_data.get("hidden_width", hidden_width)),
        hidden_depth=int(net_data.get("hidden_depth", hidden_depth)),
        activation=str(net_data.get("activation", "selu")),
        init_scale=float(net_data.get("init_scale", 0.01)),
    )
    return params, net_cfg


def _display_table(rows: list[dict], *, title: str) -> object:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    try:
        import pandas as pd
        from IPython.display import display

        df = pd.DataFrame(rows)
        display(df)
        return df
    except Exception:
        for row in rows:
            print(row)
        return rows


def _plot_ir_deviations(
    labels: list[str],
    defs_by_label: dict[str, dict[str, np.ndarray]],
    *,
    plot_vars: list[str],
    presteps: int,
    title_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: matplotlib import failed: {exc}")
        return

    if not labels or "no_event" not in defs_by_label:
        return
    base = defs_by_label["no_event"]
    t = np.arange(np.asarray(next(iter(base.values()))).shape[0]) - int(presteps)
    cols = 2
    rows = int(np.ceil(len(plot_vars) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.0 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, var in zip(axes, plot_vars):
        if var not in base:
            ax.axis("off")
            continue
        for label in labels:
            if label == "no_event" or var not in defs_by_label[label]:
                continue
            y = np.asarray(defs_by_label[label][var]) - np.asarray(base[var])
            ax.plot(t, y, label=label)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.grid(alpha=0.25)
        ax.set_title(f"{title_prefix}: {var} deviation from no_event")
    for ax in axes[len(plot_vars) :]:
        ax.axis("off")
    axes[min(len(plot_vars), len(axes)) - 1].legend(loc="best")
    fig.tight_layout()
    plt.show()


def _enrich_common(defs: dict[str, np.ndarray], params: BaselineParams) -> dict[str, np.ndarray]:
    out = dict(defs)
    if {"R", "R_standard"}.issubset(out):
        out["R_over_R_standard"] = out["R"] / np.maximum(out["R_standard"], 1e-12)
    if {"Omega_A", "p_a"}.issubset(out):
        out["repair_threshold"] = out["Omega_A"] * out["p_a"] * float(params.psi_A)
    if {"Q_A", "repair_threshold"}.issubset(out):
        out["Q_over_repair_threshold"] = out["Q_A"] / np.maximum(out["repair_threshold"], 1e-12)
    if "repair_activation_ratio" in out and "Q_over_repair_threshold" not in out:
        out["Q_over_repair_threshold"] = out["repair_activation_ratio"]
    return out


def _scenario_rows(
    labels: list[str],
    defs_by_label: dict[str, dict[str, np.ndarray]],
    *,
    variables: list[str],
    presteps: int,
) -> list[dict]:
    rows: list[dict] = []
    base = defs_by_label.get("no_event", {})
    for label in labels:
        if label == "no_event":
            continue
        defs = defs_by_label[label]
        for var in variables:
            if var not in defs:
                continue
            arr = np.asarray(defs[var], dtype=float)
            base_arr = np.asarray(base.get(var, np.zeros_like(arr)), dtype=float)
            dev = arr - base_arr
            rows.append(
                {
                    "scenario": label,
                    "variable": var,
                    "event": float(arr[int(presteps)]),
                    "peak": float(np.nanmax(arr)),
                    "min_dev": float(np.nanmin(dev)),
                    "max_dev": float(np.nanmax(dev)),
                }
            )
    return rows


def rule_ir_mechanism_diagnostics(
    *,
    artifact_root: str | Path,
    output_dir: str | Path,
    policy: str,
    natural_checkpoint: str | Path | None = None,
    policy_checkpoint: str | Path | None = None,
    device: str = "cpu",
    dtype: torch.dtype | str = torch.float64,
    hidden_width: int = 192,
    hidden_depth: int = 2,
    burnin: int = 400,
    horizon: int = 160,
    presteps: int = 5,
    relief_lag: int = 8,
    plot: bool = True,
    natural_benchmark: str = "oracle",
    natural_oracle_nodes: int = 32,
    natural_oracle_chunk_size: int = 8192,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]], object]:
    """Load a rule-policy checkpoint and report the IRF mechanism diagnostics."""

    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    dtype_t = _as_dtype(dtype)
    run_config, run_config_path = _load_run_config(artifact_root, output_dir)
    params, net_cfg = _params_and_net_cfg(run_config, hidden_width=hidden_width, hidden_depth=hidden_depth)

    natural_candidates = [
        natural_checkpoint,
        artifact_root / "natural" / "checkpoints" / "natural_best.pt",
        artifact_root / "natural" / "natural.pt",
    ]
    natural_path = (
        maybe_first_existing(natural_candidates)
        if natural_benchmark == "oracle"
        else first_existing(natural_candidates)
    )
    policy_path = first_existing(
        [
            policy_checkpoint,
            output_dir / "checkpoints" / f"{policy}_best.pt",
            output_dir / f"{policy}.pt",
            latest_step_checkpoint(output_dir, policy),
        ]
    )

    if natural_benchmark == "oracle":
        natural_net = NaturalOracleNet(
            params=params,
            qmc_cfg=QMCConfig(n_train=natural_oracle_nodes, seed=991),
            n_nodes=natural_oracle_nodes,
            device=device,
            dtype=dtype_t,
            chunk_size=natural_oracle_chunk_size,
        )
        natural_net.eval()
    else:
        natural_net = make_natural_net(net_cfg, device=device, dtype=dtype_t)
        load_checkpoint(natural_path, natural_net, map_location=device)
        natural_net.eval()

    rule_net = make_rule_net(net_cfg, device=device, dtype=dtype_t)
    metadata = load_checkpoint(policy_path, rule_net, map_location=device)
    rule_net.eval()

    print("policy:", policy)
    print("policy_checkpoint:", policy_path)
    print("natural_checkpoint:", natural_path)
    print("natural_benchmark:", "oracle" if getattr(natural_net, "is_natural_oracle", False) else "network")
    print("run_config:", run_config_path)
    if metadata:
        print("checkpoint_step:", metadata.get("step"))
        metrics = metadata.get("metrics", {})
        if isinstance(metrics, dict):
            for key in ["val_rms", "scenario_Q.rms", "calm_anchor.rms", "calm_residual.rms"]:
                if key in metrics:
                    print(f"{key}: {float(metrics[key]):.4e}")

    labels, states = simulate_rule_ir_scenarios(
        policy=policy,
        rule_net=rule_net,
        natural_net=natural_net,
        params=params,
        burnin=burnin,
        horizon=horizon,
        presteps=presteps,
        relief_lag=relief_lag,
        device=device,
        dtype=dtype_t,
    )
    _, defs = evaluate_rule_path(states, policy=policy, rule_net=rule_net, natural_net=natural_net, params=params)
    defs = _enrich_common(defs, params)
    defs_by_label = {label: {k: np.asarray(v)[:, i] for k, v in defs.items()} for i, label in enumerate(labels)}

    variables = [
        "R_standard",
        "R",
        "R_over_R_standard",
        "bottleneck_scarcity",
        "bottleneck_adjustment",
        "repair_margin_standard",
        "repair_support",
        "repair_adjustment",
        "cap_pressure_policy",
        "cap_pressure_ratio",
        "chi",
        "Pi",
        "output_gap",
        "Q_A",
        "Q_over_repair_threshold",
        "B_A",
        "I_A",
        "A",
    ]
    rows = _scenario_rows(labels, defs_by_label, variables=variables, presteps=presteps)
    table = _display_table(rows, title=f"{policy} IRF mechanism diagnostics")

    if plot:
        plot_vars = [
            "R",
            "R_over_R_standard",
            "Pi",
            "output_gap",
            "cap_pressure_ratio",
            "chi",
            "Q_A",
            "Q_over_repair_threshold",
            "I_A",
            "A",
        ]
        if policy in {"bottleneck", "repair_aware"}:
            plot_vars.insert(4, "cap_pressure_policy")
            plot_vars.insert(5, "bottleneck_scarcity")
            plot_vars.insert(6, "bottleneck_adjustment")
        if policy == "repair_aware":
            plot_vars.insert(7, "repair_margin_standard")
            plot_vars.insert(8, "repair_support")
            plot_vars.insert(9, "repair_adjustment")
        _plot_ir_deviations(labels, defs_by_label, plot_vars=plot_vars, presteps=presteps, title_prefix=policy)

    return labels, defs_by_label, table


def optimal_ir_mechanism_diagnostics(
    *,
    artifact_root: str | Path,
    output_dir: str | Path,
    kind: str,
    natural_checkpoint: str | Path | None = None,
    policy_checkpoint: str | Path | None = None,
    device: str = "cpu",
    dtype: torch.dtype | str = torch.float64,
    hidden_width: int = 192,
    hidden_depth: int = 2,
    burnin: int = 400,
    horizon: int = 160,
    presteps: int = 5,
    relief_lag: int = 8,
    plot: bool = True,
    natural_benchmark: str = "oracle",
    natural_oracle_nodes: int = 32,
    natural_oracle_chunk_size: int = 8192,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]], object]:
    """Load an optimal-policy checkpoint and report the IRF mechanism diagnostics."""

    if kind not in {"discretion", "commitment"}:
        raise ValueError("kind must be 'discretion' or 'commitment'.")

    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    dtype_t = _as_dtype(dtype)
    run_config, run_config_path = _load_run_config(artifact_root, output_dir)
    params, net_cfg = _params_and_net_cfg(run_config, hidden_width=hidden_width, hidden_depth=hidden_depth)

    natural_candidates = [
        natural_checkpoint,
        artifact_root / "natural" / "checkpoints" / "natural_best.pt",
        artifact_root / "natural" / "natural.pt",
    ]
    natural_path = (
        maybe_first_existing(natural_candidates)
        if natural_benchmark == "oracle"
        else first_existing(natural_candidates)
    )
    policy_path = first_existing(
        [
            policy_checkpoint,
            output_dir / "checkpoints" / f"{kind}_best.pt",
            output_dir / f"{kind}.pt",
            latest_step_checkpoint(output_dir, kind),
        ]
    )

    if natural_benchmark == "oracle":
        natural_net = NaturalOracleNet(
            params=params,
            qmc_cfg=QMCConfig(n_train=natural_oracle_nodes, seed=991),
            n_nodes=natural_oracle_nodes,
            device=device,
            dtype=dtype_t,
            chunk_size=natural_oracle_chunk_size,
        )
        natural_net.eval()
    else:
        natural_net = make_natural_net(net_cfg, device=device, dtype=dtype_t)
        load_checkpoint(natural_path, natural_net, map_location=device)
        natural_net.eval()

    if kind == "discretion":
        policy_net = make_discretion_net(net_cfg, device=device, dtype=dtype_t)
    else:
        policy_net = make_commitment_net(net_cfg, device=device, dtype=dtype_t)
    metadata = load_checkpoint(policy_path, policy_net, map_location=device)
    policy_net.eval()

    print("kind:", kind)
    print("policy_checkpoint:", policy_path)
    print("natural_checkpoint:", natural_path)
    print("natural_benchmark:", "oracle" if getattr(natural_net, "is_natural_oracle", False) else "network")
    print("run_config:", run_config_path)
    if metadata:
        print("checkpoint_step:", metadata.get("step"))
        metrics = metadata.get("metrics", {})
        if isinstance(metrics, dict):
            for key in ["val_rms", "scenario_Q.rms", "calm_anchor.rms", "calm_residual.rms"]:
                if key in metrics:
                    print(f"{key}: {float(metrics[key]):.4e}")

    opt_qmc_cfg = QMCConfig(n_train=64, seed=777)
    opt_nodes = make_qmc_nodes(64, cfg=opt_qmc_cfg, device=device, dtype=dtype_t)
    labels, states = simulate_optimal_ir_scenarios(
        nodes=opt_nodes,
        qmc_cfg=opt_qmc_cfg,
        kind=kind,
        policy_net=policy_net,
        params=params,
        burnin=burnin,
        horizon=horizon,
        presteps=presteps,
        relief_lag=relief_lag,
        device=device,
        dtype=dtype_t,
    )
    _, defs = evaluate_optimal_path(
        states,
        kind=kind,
        policy_net=policy_net,
        natural_net=natural_net,
        params=params,
        nodes=opt_nodes,
        qmc_cfg=opt_qmc_cfg,
    )
    defs = _enrich_common(defs, params)
    defs_by_label = {label: {k: np.asarray(v)[:, i] for k, v in defs.items()} for i, label in enumerate(labels)}

    variables = [
        "R",
        "Pi",
        "output_gap",
        "cap_pressure_ratio",
        "chi",
        "Q_A",
        "Q_over_repair_threshold",
        "B_A",
        "I_A",
        "A",
    ]
    rows = _scenario_rows(labels, defs_by_label, variables=variables, presteps=presteps)
    table = _display_table(rows, title=f"{kind} IRF mechanism diagnostics")

    if plot:
        _plot_ir_deviations(labels, defs_by_label, plot_vars=variables, presteps=presteps, title_prefix=kind)

    return labels, defs_by_label, table
