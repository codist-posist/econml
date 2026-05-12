from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiments import experiment_root


PLOT_VARS = ("pm", "mbar", "chi", "Pi", "output_gap", "I_A", "A", "R")
VAR_LABELS = {
    "pm": r"$p_t^m$",
    "mbar": r"$\bar m_t$",
    "chi": r"$\chi_t$",
    "Pi": r"$\Pi_t$",
    "output_gap": "output gap",
    "I_A": r"$I_t^A$",
    "A": r"$A_t$",
    "R": r"$R_t$",
}
POLICIES = ("fixed", "ba", "bottleneck", "repair_aware", "discretion", "commitment")


def _postprocess_dir(base_root: Path, experiment: str) -> Path:
    return experiment_root(base_root, experiment) / "postprocess"


def _load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ir_series(data: dict[str, np.ndarray], scenario: str, variable: str) -> np.ndarray | None:
    value = data.get(f"{scenario}__{variable}")
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ir_grid(
    *,
    data: dict[str, np.ndarray],
    scenario: str,
    title: str,
    output: Path,
    variables: tuple[str, ...] = PLOT_VARS,
) -> bool:
    available = [var for var in variables if _ir_series(data, scenario, var) is not None]
    if not available:
        return False
    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.2 * nrows), squeeze=False)
    for ax, var in zip(axes.flat, available):
        y = _ir_series(data, scenario, var)
        assert y is not None
        if var == "Pi":
            y = 400.0 * (y - 1.0)
            ylabel = "annualized pp"
        elif var == "output_gap":
            y = 100.0 * y
            ylabel = "log percent"
        else:
            ylabel = ""
        ax.plot(np.arange(y.shape[0]), y, linewidth=1.8)
        ax.axvline(5, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(available) :]:
        ax.axis("off")
    fig.suptitle(title)
    _save(fig, output)
    return True


def plot_policy_comparison(
    *,
    root: Path,
    scenario: str,
    variable: str,
    output: Path,
    title: str,
) -> bool:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    any_series = False
    for policy in POLICIES:
        data = _load_npz(root / f"IR_{policy}_definitions.npz")
        if data is None:
            continue
        y = _ir_series(data, scenario, variable)
        if y is None:
            continue
        if variable == "Pi":
            y = 400.0 * (y - 1.0)
        elif variable == "output_gap":
            y = 100.0 * y
        ax.plot(np.arange(y.shape[0]), y, label=policy, linewidth=1.8)
        any_series = True
    if not any_series:
        plt.close(fig)
        return False
    ax.axvline(5, color="0.7", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output)
    return True


def _transform_plot_variable(variable: str, y: np.ndarray) -> tuple[np.ndarray, str]:
    if variable == "Pi":
        return 400.0 * (y - 1.0), "annualized pp"
    if variable == "output_gap":
        return 100.0 * y, "log percent"
    return y, ""


def plot_policy_ir_grid_comparison(
    *,
    root: Path,
    policies: tuple[str, ...],
    scenario: str,
    variables: tuple[str, ...],
    output: Path,
    title: str,
    min_policies: int,
) -> bool:
    loaded = {policy: _load_npz(root / f"IR_{policy}_definitions.npz") for policy in policies}
    available_vars = []
    for variable in variables:
        n_series = sum(
            1
            for policy in policies
            if loaded.get(policy) is not None and _ir_series(loaded[policy], scenario, variable) is not None
        )
        if n_series >= int(min_policies):
            available_vars.append(variable)
    if not available_vars:
        return False

    ncols = 2
    nrows = int(np.ceil(len(available_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), squeeze=False)
    for ax, variable in zip(axes.flat, available_vars):
        for policy in policies:
            data = loaded.get(policy)
            if data is None:
                continue
            y = _ir_series(data, scenario, variable)
            if y is None:
                continue
            y, ylabel = _transform_plot_variable(variable, y)
            ax.plot(np.arange(y.shape[0]), y, label=policy, linewidth=1.6)
        ax.axvline(5, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(VAR_LABELS.get(variable, variable))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(available_vars) :]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle(title)
    _save(fig, output)
    return True


def plot_experiment_comparison(
    *,
    base_root: Path,
    experiments: tuple[str, ...],
    policy: str,
    scenario: str,
    variable: str,
    output: Path,
    title: str,
    min_series: int = 2,
) -> bool:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    n_series = 0
    for experiment in experiments:
        data = _load_npz(_postprocess_dir(base_root, experiment) / f"IR_{policy}_definitions.npz")
        if data is None:
            continue
        y = _ir_series(data, scenario, variable)
        if y is None:
            continue
        if variable == "Pi":
            y = 400.0 * (y - 1.0)
        elif variable == "output_gap":
            y = 100.0 * y
        ax.plot(np.arange(y.shape[0]), y, label=experiment, linewidth=1.8)
        n_series += 1
    if n_series < int(min_series):
        plt.close(fig)
        return False
    ax.axvline(5, color="0.7", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output)
    return True


def plot_experiment_ir_grid_comparison(
    *,
    base_root: Path,
    experiments: tuple[str, ...],
    policy: str,
    scenario: str,
    variables: tuple[str, ...],
    output: Path,
    title: str,
    min_series: int,
) -> bool:
    loaded = {
        experiment: _load_npz(_postprocess_dir(base_root, experiment) / f"IR_{policy}_definitions.npz")
        for experiment in experiments
    }
    available_vars = []
    for variable in variables:
        n_series = sum(
            1
            for experiment in experiments
            if loaded.get(experiment) is not None and _ir_series(loaded[experiment], scenario, variable) is not None
        )
        if n_series >= int(min_series):
            available_vars.append(variable)
    if not available_vars:
        return False

    ncols = 2
    nrows = int(np.ceil(len(available_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), squeeze=False)
    for ax, variable in zip(axes.flat, available_vars):
        for experiment in experiments:
            data = loaded.get(experiment)
            if data is None:
                continue
            y = _ir_series(data, scenario, variable)
            if y is None:
                continue
            y, ylabel = _transform_plot_variable(variable, y)
            ax.plot(np.arange(y.shape[0]), y, label=experiment, linewidth=1.6)
        ax.axvline(5, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(VAR_LABELS.get(variable, variable))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(available_vars) :]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle(title)
    _save(fig, output)
    return True


def plot_distribution_panels(*, root: Path, policy: str, output: Path) -> bool:
    data = _load_npz(root / f"{policy}_definitions.npz")
    if data is None:
        return False
    specs = [
        ("Pi", "Inflation, annualized pp", lambda x: 400.0 * (x - 1.0)),
        ("chi", "Scarcity rent", lambda x: x),
        ("I_A", "Repair investment", lambda x: x),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, title, transform) in zip(axes.flat[:3], specs):
        value = data.get(name)
        if value is None:
            ax.axis("off")
            continue
        x = transform(np.asarray(value, dtype=float).reshape(-1))
        ax.hist(x, bins=50, alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    chi = data.get("chi")
    I_A = data.get("I_A")
    if chi is not None and I_A is not None:
        axes.flat[3].scatter(np.asarray(chi).reshape(-1), np.asarray(I_A).reshape(-1), s=4, alpha=0.25)
        axes.flat[3].set_title(r"Joint: $\chi_t$ and $I_t^A$")
        axes.flat[3].set_xlabel(r"$\chi_t$")
        axes.flat[3].set_ylabel(r"$I_t^A$")
        axes.flat[3].grid(alpha=0.2)
    else:
        axes.flat[3].axis("off")
    _save(fig, output)
    return True


def plot_complementarity_diagnostics(*, root: Path, policy: str, output: Path) -> bool:
    data = _load_npz(root / f"{policy}_definitions.npz")
    if data is None:
        return False
    chi = data.get("chi")
    mbar = data.get("mbar")
    M = data.get("M")
    if chi is None or mbar is None or M is None:
        return False
    chi = np.asarray(chi, dtype=float).reshape(-1)
    slack = data.get("cap_slack")
    if slack is None:
        slack = (np.asarray(mbar, dtype=float).reshape(-1) - np.asarray(M, dtype=float).reshape(-1)) / np.maximum(
            np.asarray(mbar, dtype=float).reshape(-1), 1e-12
        )
    else:
        slack = np.asarray(slack, dtype=float).reshape(-1)
    product = data.get("cap_product_scaled", data.get("cap_product"))
    if product is None:
        product = chi * slack
    else:
        product = np.asarray(product, dtype=float).reshape(-1)
    repair_gap = data.get("repair_gap_scaled", data.get("repair_gap"))
    repair_projection = data.get("repair_projection_residual")
    has_repair = repair_gap is not None and repair_projection is not None and data.get("I_A") is not None
    if has_repair:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    flat = np.asarray(axes).reshape(-1)
    flat[0].hist(chi, bins=50, alpha=0.8)
    flat[0].set_title(r"$\chi_t$")
    flat[1].hist(slack, bins=50, alpha=0.8)
    flat[1].set_title("relative cap slack")
    flat[2].hist(product, bins=50, alpha=0.8)
    flat[2].set_title("cap product")
    if has_repair:
        I_A = np.asarray(data.get("I_A"), dtype=float).reshape(-1)
        flat[3].hist(I_A, bins=50, alpha=0.8)
        flat[3].set_title(r"$I_t^A$")
        flat[4].hist(np.asarray(repair_gap, dtype=float).reshape(-1), bins=50, alpha=0.8)
        flat[4].set_title("repair gap")
        flat[5].hist(np.asarray(repair_projection, dtype=float).reshape(-1), bins=50, alpha=0.8)
        flat[5].set_title("repair projection")
    for ax in flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Complementarity diagnostics: {policy}")
    _save(fig, output)
    return True


def plot_training_diagnostics(*, base_root: Path, output: Path) -> bool:
    specs = {
        "natural": base_root / "natural" / "natural_train_log.json",
        "fixed": base_root / "fixed_taylor" / "fixed_train_log.json",
        "ba": base_root / "modified_taylor" / "ba_train_log.json",
        "bottleneck": base_root / "bottleneck_taylor" / "bottleneck_train_log.json",
        "repair_aware": base_root / "repair_aware_taylor" / "repair_aware_train_log.json",
        "discretion": base_root / "discretion" / "discretion_train_log.json",
        "commitment": base_root / "commitment" / "commitment_train_log.json",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    any_series = False
    for label, path in specs.items():
        data = _load_json(path)
        if not data:
            continue
        steps = np.asarray(data.get("steps", []), dtype=float)
        if steps.size == 0:
            continue
        train = np.asarray(data.get("rms", []), dtype=float)
        val = np.asarray(data.get("val_rms", []), dtype=float)
        if train.size:
            axes[0].plot(steps[: train.size], train, label=label, linewidth=1.4)
            any_series = True
        if val.size:
            axes[1].plot(steps[: val.size], val, label=label, linewidth=1.4)
            any_series = True
    if not any_series:
        plt.close(fig)
        return False
    axes[0].set_title("Training RMS residual")
    axes[1].set_title("Validation RMS residual")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("DEQN convergence diagnostics")
    _save(fig, output)
    return True


def plot_eval_residual_bars(*, base_root: Path, output: Path) -> bool:
    specs = {
        "natural": base_root / "natural" / "natural_eval.json",
        "fixed": base_root / "fixed_taylor" / "fixed_eval.json",
        "ba": base_root / "modified_taylor" / "ba_eval.json",
        "bottleneck": base_root / "bottleneck_taylor" / "bottleneck_eval.json",
        "repair_aware": base_root / "repair_aware_taylor" / "repair_aware_eval.json",
        "discretion": base_root / "discretion" / "discretion_eval.json",
        "commitment": base_root / "commitment" / "commitment_eval.json",
    }
    labels = []
    values = []
    for label, path in specs.items():
        data = _load_json(path)
        if not data:
            continue
        overall = data.get("overall.rms")
        if isinstance(overall, (int, float)):
            labels.append(label)
            values.append(float(overall))
    if not values:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_yscale("log")
    ax.set_title("Out-of-sample overall residual RMS")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output)
    return True


def make_figures(base_root: Path, output_dir: Path, *, policy: str = "ba") -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    baseline_root = _postprocess_dir(base_root, "baseline")
    baseline_ir = _load_npz(baseline_root / f"IR_{policy}_definitions.npz")
    if baseline_ir is not None and plot_ir_grid(
        data=baseline_ir,
        scenario="D_1x",
        title="Benchmark mechanism after a disruption event",
        output=output_dir / "figure_1_benchmark_disruption.png",
    ):
        written["figure_1"] = str(output_dir / "figure_1_benchmark_disruption.png")

    if plot_experiment_ir_grid_comparison(
        base_root=base_root,
        experiments=("price_only", "quantity_only", "baseline"),
        policy=policy,
        scenario="D_1x",
        output=output_dir / "figure_2_price_quantity_full_bottleneck.png",
        title="Price-only vs quantity-only vs full bottleneck",
        variables=PLOT_VARS,
        min_series=3,
    ):
        written["figure_2"] = str(output_dir / "figure_2_price_quantity_full_bottleneck.png")

    policy_grid_vars = ("Pi", "output_gap", "chi", "I_A", "A", "R")
    if plot_policy_ir_grid_comparison(
        root=baseline_root,
        policies=POLICIES,
        scenario="D_1x",
        variables=policy_grid_vars,
        output=output_dir / "figure_3_policy_comparison_disruption.png",
        title="Policy comparison after disruption",
        min_policies=4,
    ):
        written["figure_3"] = str(output_dir / "figure_3_policy_comparison_disruption.png")
    if plot_policy_ir_grid_comparison(
        root=baseline_root,
        policies=("fixed", "ba", "bottleneck", "repair_aware"),
        scenario="D_1x",
        variables=policy_grid_vars,
        output=output_dir / "figure_4_rule_taylor_comparison.png",
        title="Fixed vs natural-rate-adjusted vs bottleneck vs repair-aware Taylor",
        min_policies=2,
    ):
        written["figure_4"] = str(output_dir / "figure_4_rule_taylor_comparison.png")
    if plot_policy_ir_grid_comparison(
        root=baseline_root,
        policies=("discretion", "commitment"),
        scenario="D_1x",
        variables=policy_grid_vars,
        output=output_dir / "figure_5_discretion_vs_commitment.png",
        title="Discretion vs commitment",
        min_policies=2,
    ):
        written["figure_5"] = str(output_dir / "figure_5_discretion_vs_commitment.png")

    for variable in ("Pi", "chi", "I_A"):
        path = output_dir / f"diagnostic_policy_comparison_{variable}.png"
        if plot_policy_comparison(
            root=baseline_root,
            scenario="D_1x",
            variable=variable,
            output=path,
            title=f"Policy comparison after disruption: {VAR_LABELS.get(variable, variable)}",
        ):
            written[f"diagnostic_{variable}"] = str(path)

    sensitivity_specs = [
        ("persistent_crisis", "figure_6_persistence_chi.png", "chi", ("baseline", "persistent_crisis")),
        ("fast_relief", "figure_7_relief_I_A.png", "I_A", ("baseline", "fast_relief", "fragile_relief")),
        ("low_substitutability", "figure_8_substitutability_chi.png", "chi", ("baseline", "low_substitutability", "high_substitutability")),
        ("high_financing_sensitivity", "figure_9_financing_I_A.png", "I_A", ("baseline", "no_financing", "high_financing_sensitivity")),
    ]
    for _, filename, variable, experiments in sensitivity_specs:
        path = output_dir / filename
        if plot_experiment_comparison(
            base_root=base_root,
            experiments=experiments,
            policy=policy,
            scenario="D_1x",
            variable=variable,
            output=path,
            title=f"Sensitivity: {VAR_LABELS.get(variable, variable)}",
            min_series=len(experiments),
        ):
            written[filename] = str(path)

    policy_sensitivity_path = output_dir / "figure_11_policy_aggressiveness.png"
    if plot_experiment_ir_grid_comparison(
        base_root=base_root,
        experiments=("dovish_policy", "baseline", "hawkish_policy"),
        policy=policy,
        scenario="D_1x",
        variables=policy_grid_vars,
        output=policy_sensitivity_path,
        title="Policy aggressiveness sensitivity",
        min_series=3,
    ):
        written["figure_11"] = str(policy_sensitivity_path)

    output_gap_rule_path = output_dir / "figure_12_output_gap_policy.png"
    if plot_experiment_ir_grid_comparison(
        base_root=base_root,
        experiments=("baseline", "output_gap_policy"),
        policy=policy,
        scenario="D_1x",
        variables=policy_grid_vars,
        output=output_gap_rule_path,
        title="Output-gap response in the policy rule",
        min_series=2,
    ):
        written["figure_12"] = str(output_gap_rule_path)

    dist_path = output_dir / f"figure_10_distribution_{policy}.png"
    if plot_distribution_panels(root=baseline_root, policy=policy, output=dist_path):
        written["figure_10"] = str(dist_path)

    comp_path = output_dir / f"diagnostic_complementarity_{policy}.png"
    if plot_complementarity_diagnostics(root=baseline_root, policy=policy, output=comp_path):
        written["diagnostic_complementarity"] = str(comp_path)

    training_path = output_dir / "diagnostic_training_residuals.png"
    if plot_training_diagnostics(base_root=base_root, output=training_path):
        written["diagnostic_training"] = str(training_path)

    eval_path = output_dir / "diagnostic_eval_residuals.png"
    if plot_eval_residual_bars(base_root=base_root, output=eval_path):
        written["diagnostic_eval"] = str(eval_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures from critical-input DEQN postprocess artifacts.")
    parser.add_argument("--base-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--policy", default="repair_aware", choices=POLICIES)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.base_root / "figures")
    written = make_figures(args.base_root, output_dir, policy=args.policy)
    print(json.dumps(written, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
