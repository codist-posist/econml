from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from .config import BaselineParams


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    group: str
    description: str
    overrides: dict[str, float]
    requires_retrain: bool = True


def _half_life_delta(quarters: float) -> float:
    return 1.0 - 2.0 ** (-1.0 / float(quarters))


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline": ExperimentSpec(
        name="baseline",
        group="baseline",
        description="Full baseline bottleneck economy.",
        overrides={},
    ),
    "quantity_only": ExperimentSpec(
        name="quantity_only",
        group="core_irf",
        description="Quantity bottleneck without an external import-price channel.",
        overrides={"nu_pD": 0.0, "nu_pX": 0.0},
    ),
    "price_only": ExperimentSpec(
        name="price_only",
        group="core_irf",
        description="Import-price shock without an external quantity channel.",
        overrides={"nu_qD": 0.0, "nu_qX": 0.0, "bar_m": 1.0e6},
    ),
    "no_cap": ExperimentSpec(
        name="no_cap",
        group="counterfactual",
        description="Imported-input cap is effectively slack, so scarcity rent should vanish.",
        overrides={"nu_qD": 0.0, "nu_qX": 0.0, "bar_m": 1.0e6},
    ),
    "no_adaptation": ExperimentSpec(
        name="no_adaptation",
        group="counterfactual",
        description="Adaptation has no input-saving effect.",
        overrides={"kappa_a": 0.0},
    ),
    "no_financing": ExperimentSpec(
        name="no_financing",
        group="counterfactual",
        description="Repair user cost is independent of the policy rate.",
        overrides={"vartheta_A": 0.0},
    ),
    "no_relief": ExperimentSpec(
        name="no_relief",
        group="counterfactual",
        description="External relief arrivals and relief effects are shut down.",
        overrides={
            "log_bar_lambda_X": -30.0,
            "beta_X": 0.0,
            "mark_X": 0.0,
            "nu_pX": 0.0,
            "nu_qX": 0.0,
        },
    ),
    "deep_crisis": ExperimentSpec(
        name="deep_crisis",
        group="sensitivity",
        description="Larger disruption events and stronger price/quantity effects.",
        overrides={"mark_D": 0.50, "nu_qD": 0.45, "nu_pD": 0.225},
    ),
    "persistent_crisis": ExperimentSpec(
        name="persistent_crisis",
        group="sensitivity",
        description="Slower disruption decay and more clustered disruption arrivals.",
        overrides={"delta_D": _half_life_delta(48.0), "rho_lambda_D": 0.80, "kappa_D_lambda": 0.10},
    ),
    "fast_relief": ExperimentSpec(
        name="fast_relief",
        group="sensitivity",
        description="Relief arrives more often and decays more slowly.",
        overrides={"log_bar_lambda_X": math.log(1.0 / 12.0), "beta_X": 0.10, "delta_X": _half_life_delta(24.0)},
    ),
    "fragile_relief": ExperimentSpec(
        name="fragile_relief",
        group="sensitivity",
        description="Relief is rarer, smaller, and less durable.",
        overrides={"log_bar_lambda_X": math.log(1.0 / 48.0), "beta_X": 0.02, "delta_X": _half_life_delta(6.0), "mark_X": 0.075},
    ),
    "low_slack": ExperimentSpec(
        name="low_slack",
        group="sensitivity",
        description="Tighter normal imported-input availability.",
        overrides={"bar_m": 0.20},
    ),
    "high_slack": ExperimentSpec(
        name="high_slack",
        group="sensitivity",
        description="Looser normal imported-input availability.",
        overrides={"bar_m": 0.45},
    ),
    "low_substitutability": ExperimentSpec(
        name="low_substitutability",
        group="sensitivity",
        description="Critical imported input is harder to substitute.",
        overrides={"rho": 0.05},
    ),
    "high_substitutability": ExperimentSpec(
        name="high_substitutability",
        group="sensitivity",
        description="Critical imported input is easier to substitute.",
        overrides={"rho": 0.50},
    ),
    "high_import_exposure": ExperimentSpec(
        name="high_import_exposure",
        group="sensitivity",
        description="Firms initially rely more heavily on the critical imported input.",
        overrides={"omega0": 0.45},
    ),
    "low_import_exposure": ExperimentSpec(
        name="low_import_exposure",
        group="sensitivity",
        description="Firms initially rely less heavily on the critical imported input.",
        overrides={"omega0": 0.15},
    ),
    "high_adaptation_effectiveness": ExperimentSpec(
        name="high_adaptation_effectiveness",
        group="sensitivity",
        description="Installed adaptation reduces import dependence more effectively.",
        overrides={"kappa_a": 0.30},
    ),
    "low_adaptation_effectiveness": ExperimentSpec(
        name="low_adaptation_effectiveness",
        group="sensitivity",
        description="Installed adaptation reduces import dependence less effectively.",
        overrides={"kappa_a": 0.075},
    ),
    "high_repair_cost": ExperimentSpec(
        name="high_repair_cost",
        group="sensitivity",
        description="Adaptation investment is more convex and costly.",
        overrides={"phi_A": 6.0},
    ),
    "low_repair_cost": ExperimentSpec(
        name="low_repair_cost",
        group="sensitivity",
        description="Adaptation investment is less costly.",
        overrides={"phi_A": 1.5},
    ),
    "high_repair_depreciation": ExperimentSpec(
        name="high_repair_depreciation",
        group="sensitivity",
        description="Installed adaptation depreciates faster.",
        overrides={"delta_A": 0.07},
    ),
    "low_repair_depreciation": ExperimentSpec(
        name="low_repair_depreciation",
        group="sensitivity",
        description="Installed adaptation is more durable.",
        overrides={"delta_A": 0.0175},
    ),
    "high_financing_sensitivity": ExperimentSpec(
        name="high_financing_sensitivity",
        group="sensitivity",
        description="Repair cost is more sensitive to the policy rate.",
        overrides={"vartheta_A": 0.80},
    ),
    "hawkish_policy": ExperimentSpec(
        name="hawkish_policy",
        group="sensitivity",
        description="Rule-based policies respond more aggressively to inflation.",
        overrides={"phi_pi": 3.0},
    ),
    "dovish_policy": ExperimentSpec(
        name="dovish_policy",
        group="sensitivity",
        description="Rule-based policies respond less aggressively to inflation.",
        overrides={"phi_pi": 1.25},
    ),
    "output_gap_policy": ExperimentSpec(
        name="output_gap_policy",
        group="sensitivity",
        description="Rule-based policies also respond to the output gap.",
        overrides={"phi_y": 0.25},
    ),
}


TABLE_EXPERIMENTS = {
    "counterfactual": ["baseline", "no_cap", "no_adaptation", "no_financing", "price_only", "quantity_only", "no_relief"],
    "sensitivity": [
        "baseline",
        "deep_crisis",
        "persistent_crisis",
        "fast_relief",
        "fragile_relief",
        "low_substitutability",
        "high_repair_cost",
        "high_financing_sensitivity",
    ],
}


def _validate_overrides(overrides: Mapping[str, Any]) -> dict[str, float]:
    allowed = {field.name for field in fields(BaselineParams)}
    bad = sorted(set(overrides) - allowed)
    if bad:
        raise ValueError(f"Unknown BaselineParams override(s): {bad}")
    return {key: float(value) for key, value in overrides.items()}


def params_from_overrides(overrides: Mapping[str, Any] | None = None) -> BaselineParams:
    return replace(BaselineParams(), **_validate_overrides(overrides or {}))


def params_from_dict(data: Mapping[str, Any]) -> BaselineParams:
    allowed = {field.name for field in fields(BaselineParams)}
    return replace(BaselineParams(), **{key: float(value) for key, value in data.items() if key in allowed})


def load_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError("Parameter override JSON must contain an object.")
    if "overrides" in payload and isinstance(payload["overrides"], Mapping):
        payload = payload["overrides"]
    return _validate_overrides(payload)


def experiment_spec(name: str) -> ExperimentSpec:
    key = name.lower().strip()
    if key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Available: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[key]


def resolve_params(experiment: str = "baseline", params_json: Path | None = None) -> tuple[BaselineParams, dict[str, Any]]:
    spec = experiment_spec(experiment)
    overrides = dict(spec.overrides)
    overrides.update(load_overrides(params_json))
    params = params_from_overrides(overrides)
    metadata = {
        "experiment": spec.name,
        "group": spec.group,
        "description": spec.description,
        "base_overrides": spec.overrides,
        "extra_overrides": load_overrides(params_json),
        "effective_overrides": overrides,
        "params": asdict(params),
    }
    return params, metadata


def params_from_metadata(metadata: Mapping[str, Any], fallback: BaselineParams | None = None) -> BaselineParams:
    candidates: list[Any] = []
    candidates.append(metadata.get("params"))
    config = metadata.get("config")
    if isinstance(config, Mapping):
        candidates.append(config.get("params"))
        experiment = config.get("experiment")
        if isinstance(experiment, Mapping):
            candidates.append(experiment.get("params"))
    experiment = metadata.get("experiment")
    if isinstance(experiment, Mapping):
        candidates.append(experiment.get("params"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return params_from_dict(candidate)
    return fallback or BaselineParams()


def experiment_root(base: Path, experiment: str) -> Path:
    key = experiment_spec(experiment).name
    return Path(base) if key == "baseline" else Path(base) / "experiments" / key


def experiment_registry_payload() -> dict[str, Any]:
    return {
        "experiments": {name: asdict(spec) for name, spec in EXPERIMENTS.items()},
        "tables": TABLE_EXPERIMENTS,
    }
