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


def _omega_from_import_share(share: float, rho: float) -> float:
    share = min(max(float(share), 1e-8), 1.0 - 1e-8)
    ratio = (share / (1.0 - share)) ** (1.0 / float(rho))
    return ratio / (1.0 + ratio)


def _steady_import_demand(params: BaselineParams) -> float:
    """No-shock desired import use used to discipline normal capacity.

    The target share is the imported-input cost share within the composite
    intermediate bundle at equal input prices.  Steady intermediate spending is
    alpha times desired real marginal cost times the target output scale.
    """

    desired_mc = (float(params.epsilon) - 1.0) / float(params.epsilon)
    return (
        float(params.target_import_cost_share)
        * float(params.alpha)
        * desired_mc
        * float(params.steady_state_output)
        / float(params.bar_p_m)
    )


def _calibrated_params(params: BaselineParams, overrides: Mapping[str, Any]) -> BaselineParams:
    """Apply internally consistent baseline calibrations unless overridden."""

    override_keys = set(overrides)
    updates: dict[str, float] = {}
    if "omega0" not in override_keys:
        updates["omega0"] = _omega_from_import_share(params.target_import_cost_share, params.rho)

    a10 = -math.log(0.90)
    psi_A = float(params.psi_A)
    if "psi_A" not in override_keys:
        psi_A = float(params.repair_cost_share_10pct) * float(params.steady_state_output) / a10
        updates["psi_A"] = psi_A
    if "phi_A" not in override_keys:
        updates["phi_A"] = (
            2.0
            * float(params.repair_convex_share_10pct)
            * float(params.repair_horizon_quarters)
            * psi_A
            / a10
        )
    if "bar_m" not in override_keys:
        updates["bar_m"] = (1.0 + float(params.normal_capacity_slack)) * _steady_import_demand(
            replace(params, **updates) if updates else params
        )
    return replace(params, **updates) if updates else params


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
        description="Adaptation investment is shut down and installed adaptation has no input-saving effect.",
        overrides={"adaptation_enabled": 0.0, "kappa_a": 0.0},
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
    "larger_disruption_marks": ExperimentSpec(
        name="larger_disruption_marks",
        group="sensitivity",
        description="Disruption events are larger, holding price and quantity loadings fixed.",
        overrides={"mark_D": 0.50},
    ),
    "stronger_quantity_effect": ExperimentSpec(
        name="stronger_quantity_effect",
        group="sensitivity",
        description="Disruption has a stronger effect on imported-input availability.",
        overrides={"nu_qD": 0.45},
    ),
    "stronger_price_effect": ExperimentSpec(
        name="stronger_price_effect",
        group="sensitivity",
        description="Disruption has a stronger effect on the import procurement price.",
        overrides={"nu_pD": 0.225},
    ),
    "persistent_crisis": ExperimentSpec(
        name="persistent_crisis",
        group="sensitivity",
        description="Slower disruption decay and more clustered disruption arrivals.",
        overrides={"delta_D": _half_life_delta(48.0), "rho_lambda_D": 0.80, "kappa_D_lambda": 0.10},
    ),
    "low_disruption_decay": ExperimentSpec(
        name="low_disruption_decay",
        group="sensitivity",
        description="Disruption stock decays more slowly.",
        overrides={"delta_D": _half_life_delta(48.0)},
    ),
    "high_disruption_clustering": ExperimentSpec(
        name="high_disruption_clustering",
        group="sensitivity",
        description="Disruption-arrival intensity is more persistent.",
        overrides={"rho_lambda_D": 0.80},
    ),
    "high_disruption_self_excitation": ExperimentSpec(
        name="high_disruption_self_excitation",
        group="sensitivity",
        description="Current disruption stock raises future disruption-arrival intensity more strongly.",
        overrides={"kappa_D_lambda": 0.10},
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
    "high_relief_arrival": ExperimentSpec(
        name="high_relief_arrival",
        group="sensitivity",
        description="External relief events arrive more frequently.",
        overrides={"log_bar_lambda_X": math.log(1.0 / 12.0)},
    ),
    "low_relief_arrival": ExperimentSpec(
        name="low_relief_arrival",
        group="sensitivity",
        description="External relief events arrive less frequently.",
        overrides={"log_bar_lambda_X": math.log(1.0 / 48.0)},
    ),
    "high_relief_response": ExperimentSpec(
        name="high_relief_response",
        group="sensitivity",
        description="Accumulated disruption triggers relief-arrival intensity more strongly.",
        overrides={"beta_X": 0.10},
    ),
    "low_relief_response": ExperimentSpec(
        name="low_relief_response",
        group="sensitivity",
        description="Accumulated disruption weakly triggers relief-arrival intensity.",
        overrides={"beta_X": 0.02},
    ),
    "durable_relief": ExperimentSpec(
        name="durable_relief",
        group="sensitivity",
        description="Relief stock is more durable.",
        overrides={"delta_X": _half_life_delta(24.0)},
    ),
    "transient_relief": ExperimentSpec(
        name="transient_relief",
        group="sensitivity",
        description="Relief stock decays faster.",
        overrides={"delta_X": _half_life_delta(6.0)},
    ),
    "large_relief_marks": ExperimentSpec(
        name="large_relief_marks",
        group="sensitivity",
        description="Relief events are larger.",
        overrides={"mark_X": 0.25},
    ),
    "small_relief_marks": ExperimentSpec(
        name="small_relief_marks",
        group="sensitivity",
        description="Relief events are smaller.",
        overrides={"mark_X": 0.075},
    ),
    "low_slack": ExperimentSpec(
        name="low_slack",
        group="sensitivity",
        description="Tighter normal imported-input availability.",
        overrides={"normal_capacity_slack": 0.03},
    ),
    "high_slack": ExperimentSpec(
        name="high_slack",
        group="sensitivity",
        description="Looser normal imported-input availability.",
        overrides={"normal_capacity_slack": 0.25},
    ),
    "very_low_capacity": ExperimentSpec(
        name="very_low_capacity",
        group="sensitivity",
        description="Lower absolute imported-input capacity.",
        overrides={"bar_m": 0.040},
    ),
    "very_high_capacity": ExperimentSpec(
        name="very_high_capacity",
        group="sensitivity",
        description="Higher absolute imported-input capacity.",
        overrides={"bar_m": 0.080},
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
        overrides={"target_import_cost_share": 0.45},
    ),
    "low_import_exposure": ExperimentSpec(
        name="low_import_exposure",
        group="sensitivity",
        description="Firms initially rely less heavily on the critical imported input.",
        overrides={"target_import_cost_share": 0.15},
    ),
    "low_residual_import_dependence": ExperimentSpec(
        name="low_residual_import_dependence",
        group="sensitivity",
        description="Adaptation can reduce critical-input dependence to a lower residual floor.",
        overrides={"target_min_import_cost_share": 0.05},
    ),
    "high_residual_import_dependence": ExperimentSpec(
        name="high_residual_import_dependence",
        group="sensitivity",
        description="Adaptation leaves a higher irreducible critical-input dependence.",
        overrides={"target_min_import_cost_share": 0.15},
    ),
    "high_adaptation_effectiveness": ExperimentSpec(
        name="high_adaptation_effectiveness",
        group="sensitivity",
        description="Installed adaptation reduces import dependence more effectively.",
        overrides={"kappa_a": 1.50},
    ),
    "low_adaptation_effectiveness": ExperimentSpec(
        name="low_adaptation_effectiveness",
        group="sensitivity",
        description="Installed adaptation reduces import dependence less effectively.",
        overrides={"kappa_a": 0.50},
    ),
    "high_repair_cost": ExperimentSpec(
        name="high_repair_cost",
        group="sensitivity",
        description="Adaptation investment is more convex and costly.",
        overrides={"repair_cost_share_10pct": 0.10},
    ),
    "low_repair_cost": ExperimentSpec(
        name="low_repair_cost",
        group="sensitivity",
        description="Adaptation investment is less costly.",
        overrides={"repair_cost_share_10pct": 0.02},
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
    "low_financing_sensitivity": ExperimentSpec(
        name="low_financing_sensitivity",
        group="sensitivity",
        description="Repair cost is less sensitive to the policy rate.",
        overrides={"vartheta_A": 0.20},
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
        "larger_disruption_marks",
        "stronger_quantity_effect",
        "stronger_price_effect",
        "persistent_crisis",
        "low_disruption_decay",
        "high_disruption_clustering",
        "high_disruption_self_excitation",
        "fast_relief",
        "fragile_relief",
        "high_relief_arrival",
        "low_relief_arrival",
        "high_relief_response",
        "low_relief_response",
        "durable_relief",
        "transient_relief",
        "large_relief_marks",
        "small_relief_marks",
        "low_slack",
        "high_slack",
        "low_substitutability",
        "high_substitutability",
        "high_import_exposure",
        "low_import_exposure",
        "low_residual_import_dependence",
        "high_residual_import_dependence",
        "high_adaptation_effectiveness",
        "low_adaptation_effectiveness",
        "high_repair_cost",
        "low_repair_cost",
        "high_repair_depreciation",
        "low_repair_depreciation",
        "no_financing",
        "low_financing_sensitivity",
        "high_financing_sensitivity",
        "hawkish_policy",
        "dovish_policy",
        "output_gap_policy",
    ],
}


def _validate_overrides(overrides: Mapping[str, Any]) -> dict[str, float]:
    allowed = {field.name for field in fields(BaselineParams)}
    bad = sorted(set(overrides) - allowed)
    if bad:
        raise ValueError(f"Unknown BaselineParams override(s): {bad}")
    return {key: float(value) for key, value in overrides.items()}


def params_from_overrides(overrides: Mapping[str, Any] | None = None) -> BaselineParams:
    checked = _validate_overrides(overrides or {})
    return _calibrated_params(replace(BaselineParams(), **checked), checked)


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
