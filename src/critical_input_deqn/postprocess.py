from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .config import (
    COMMITMENT_OUTPUT_NAMES,
    COMMITMENT_PROMISE_INIT_MEAN,
    COMMITMENT_STATE_NAMES,
    DISCRETION_OUTPUT_NAMES,
    NATURAL_OUTPUT_NAMES,
    RULE_OUTPUT_NAMES,
    RULE_STATE_NAMES,
    BaselineParams,
    NetworkConfig,
)
from .experiments import params_from_metadata, resolve_params
from .economics import derive_free, derive_rule, unpack_rule_state
from .episode import simulate_rule_episode
from .optimal import decode_commitment, decode_discretion, simulate_optimal_episode
from .sampling import sample_rule_states
from .train import (
    _initial_optimal_states,
    make_commitment_net,
    make_discretion_net,
    make_natural_net,
    make_rule_net,
)
from .transforms import decode_natural_outputs, decode_rule_outputs


TensorDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class LoadedPolicy:
    kind: str
    net: torch.nn.Module
    metadata: dict[str, object]


def _dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("--dtype must be float64 or float32.")


def _network_config_from_metadata(metadata: Mapping[str, object]) -> NetworkConfig:
    default = NetworkConfig()
    if "network" in metadata and isinstance(metadata["network"], Mapping):
        data = metadata["network"]
    else:
        config = metadata.get("config", {})
        data = config.get("network", {}) if isinstance(config, Mapping) else {}
    return NetworkConfig(
        hidden_width=int(data.get("hidden_width", default.hidden_width)),
        hidden_depth=int(data.get("hidden_depth", default.hidden_depth)),
        activation=str(data.get("activation", default.activation)),
        init_scale=float(data.get("init_scale", default.init_scale)),
    )


def _load_payload(path: Path, *, device: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Checkpoint has unexpected format: {path}")
    return payload


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these checkpoints exists: " + ", ".join(str(p) for p in paths))


def load_natural(path: Path, *, device: str, dtype: torch.dtype) -> LoadedPolicy:
    payload = _load_payload(path, device=device)
    metadata = dict(payload.get("metadata", {}))
    net = make_natural_net(_network_config_from_metadata(metadata), device=device, dtype=dtype)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return LoadedPolicy("natural", net, metadata)


def load_rule(path: Path, *, policy: str, device: str, dtype: torch.dtype) -> LoadedPolicy:
    payload = _load_payload(path, device=device)
    metadata = dict(payload.get("metadata", {}))
    net = make_rule_net(_network_config_from_metadata(metadata), device=device, dtype=dtype)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return LoadedPolicy(policy, net, metadata)


def load_optimal(path: Path, *, kind: str, device: str, dtype: torch.dtype) -> LoadedPolicy:
    payload = _load_payload(path, device=device)
    metadata = dict(payload.get("metadata", {}))
    net_cfg = _network_config_from_metadata(metadata)
    if kind == "discretion":
        net = make_discretion_net(net_cfg, device=device, dtype=dtype)
    elif kind == "commitment":
        net = make_commitment_net(net_cfg, device=device, dtype=dtype)
    else:
        raise ValueError("kind must be discretion or commitment.")
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return LoadedPolicy(kind, net, metadata)


def _numpy_dict(data: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in data.items()}


def _state_dict(z: torch.Tensor, names: tuple[str, ...]) -> TensorDict:
    return {name: z[..., i] for i, name in enumerate(names)}


def _summary_stats(data: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, arr in data.items():
        if arr.dtype.kind not in {"f", "i", "u", "b"}:
            continue
        x = np.asarray(arr, dtype=float).reshape(-1)
        if x.size == 0:
            continue
        out[name] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }
    return out


def _add_common_ratios(data: TensorDict) -> TensorDict:
    if "Y" in data and "Y_n" in data:
        data["output_gap"] = data["Y"] / torch.clamp(data["Y_n"], min=1e-12) - 1.0
    if "M" in data and "mbar" in data:
        data["cap_slack"] = (data["mbar"] - data["M"]) / torch.clamp(data["mbar"], min=1e-12)
    if "I_A" in data and "A_next" in data and "A" in data:
        data["A_growth"] = data["A_next"] - data["A"]
    if "R" in data and "Pi" in data:
        data["real_rate_ex_post_proxy"] = data["R"] / torch.clamp(data["Pi"], min=1e-12)
    return data


def _normal_initial_rule_state(
    batch_size: int,
    *,
    params: BaselineParams,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    D = torch.zeros(batch_size, device=device, dtype=dtype)
    X = torch.zeros_like(D)
    ell_D = torch.full_like(D, float(params.log_bar_lambda_D))
    ell_X = torch.full_like(D, float(params.log_bar_lambda_X))
    log_Z = torch.full_like(D, float(-0.5 * params.sigma_z**2))
    A = torch.zeros_like(D)
    log_Delta = torch.zeros_like(D)
    return torch.stack([D, X, ell_D, ell_X, log_Z, A, log_Delta], dim=-1)


def _deterministic_physical_step(
    z_phys: torch.Tensor,
    *,
    A_next: torch.Tensor,
    Delta_next: torch.Tensor,
    add_D: torch.Tensor,
    add_X: torch.Tensor,
    params: BaselineParams,
) -> torch.Tensor:
    st = unpack_rule_state(z_phys)
    D_next = (1.0 - float(params.delta_D)) * st.D + add_D
    X_next = (1.0 - float(params.delta_X)) * st.X + add_X
    ell_D_next = (
        (1.0 - float(params.rho_lambda_D)) * float(params.log_bar_lambda_D)
        + float(params.rho_lambda_D) * st.ell_D
        + float(params.kappa_D_lambda) * st.D
    )
    ell_X_next = (
        (1.0 - float(params.rho_lambda_X)) * float(params.log_bar_lambda_X)
        + float(params.rho_lambda_X) * st.ell_X
        + float(params.beta_X) * st.D
    )
    log_Z_next = float(params.rho_z) * st.log_Z
    log_Delta_next = torch.log(torch.clamp(Delta_next, min=1e-12))
    return torch.stack([D_next, X_next, ell_D_next, ell_X_next, log_Z_next, A_next, log_Delta_next], dim=-1)


def _default_ir_scenarios(params: BaselineParams, *, pulse: int, relief_lag: int) -> dict[str, dict[int, tuple[float, float]]]:
    return {
        "no_event": {},
        "D_1x": {pulse: (float(params.mark_D), 0.0)},
        "D_3x": {pulse: (3.0 * float(params.mark_D), 0.0)},
        "X_1x": {pulse: (0.0, float(params.mark_X))},
        "D_1x_X_lag": {
            pulse: (float(params.mark_D), 0.0),
            pulse + int(relief_lag): (0.0, float(params.mark_X)),
        },
        "D_3x_X_lag": {
            pulse: (3.0 * float(params.mark_D), 0.0),
            pulse + int(relief_lag): (0.0, float(params.mark_X)),
        },
    }


def _scenario_additions(
    scenarios: Mapping[str, Mapping[int, tuple[float, float]]],
    *,
    t: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    add_D = []
    add_X = []
    for spec in scenarios.values():
        d, x = spec.get(int(t), (0.0, 0.0))
        add_D.append(float(d))
        add_X.append(float(x))
    return (
        torch.tensor(add_D, device=device, dtype=dtype),
        torch.tensor(add_X, device=device, dtype=dtype),
    )


def simulate_rule_ir_scenarios(
    *,
    policy: str,
    rule_net,
    natural_net,
    params: BaselineParams,
    burnin: int,
    horizon: int,
    presteps: int,
    relief_lag: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor]:
    pulse = int(burnin)
    scenarios = _default_ir_scenarios(params, pulse=pulse, relief_lag=relief_lag)
    labels = list(scenarios.keys())
    z = _normal_initial_rule_state(len(labels), params=params, device=device, dtype=dtype)
    states = [z]
    total = int(burnin) + int(horizon)
    with torch.no_grad():
        for t in range(1, total):
            st = unpack_rule_state(z)
            out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES)
            out_n = decode_natural_outputs(natural_net(z[..., :6]), NATURAL_OUTPUT_NAMES)
            drv = derive_rule(st, out, params, Y_n=out_n["Y_n"], R_n=out_n["R_n_real"], policy=policy)
            add_D, add_X = _scenario_additions(scenarios, t=t, device=z.device, dtype=z.dtype)
            z = _deterministic_physical_step(
                z,
                A_next=drv["A_next"],
                Delta_next=drv["Delta"],
                add_D=add_D,
                add_X=add_X,
                params=params,
            )
            states.append(z)
    start = max(0, int(burnin) - int(presteps))
    return labels, torch.stack(states, dim=0)[start:]


def simulate_optimal_ir_scenarios(
    *,
    kind: str,
    policy_net,
    params: BaselineParams,
    burnin: int,
    horizon: int,
    presteps: int,
    relief_lag: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor]:
    pulse = int(burnin)
    scenarios = _default_ir_scenarios(params, pulse=pulse, relief_lag=relief_lag)
    labels = list(scenarios.keys())
    z_phys = _normal_initial_rule_state(len(labels), params=params, device=device, dtype=dtype)
    if kind == "commitment":
        promises = torch.tensor(COMMITMENT_PROMISE_INIT_MEAN, device=device, dtype=dtype)[None, :].expand(len(labels), 4)
        z = torch.cat([z_phys, promises], dim=-1)
    elif kind == "discretion":
        z = z_phys
    else:
        raise ValueError("kind must be discretion or commitment.")
    states = [z]
    total = int(burnin) + int(horizon)
    with torch.no_grad():
        for t in range(1, total):
            z_phys = z[..., :7]
            st = unpack_rule_state(z_phys)
            if kind == "commitment":
                out = decode_commitment(policy_net(z))
            else:
                out = decode_discretion(policy_net(z))
            drv = derive_free(st, out, params)
            add_D, add_X = _scenario_additions(scenarios, t=t, device=z.device, dtype=z.dtype)
            z_phys_next = _deterministic_physical_step(
                z_phys,
                A_next=drv["A_next"],
                Delta_next=drv["Delta"],
                add_D=add_D,
                add_X=add_X,
                params=params,
            )
            if kind == "commitment":
                promises_next = torch.stack(
                    [out["promise_E"], out["promise_S"], out["promise_F"], out["promise_Q"]],
                    dim=-1,
                )
                z = torch.cat([z_phys_next, promises_next], dim=-1)
            else:
                z = z_phys_next
            states.append(z)
    start = max(0, int(burnin) - int(presteps))
    return labels, torch.stack(states, dim=0)[start:]


def evaluate_rule_path(
    states: torch.Tensor,
    *,
    policy: str,
    rule_net,
    natural_net,
    params: BaselineParams,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    T, B, K = states.shape
    z = states.reshape(T * B, K)
    st = unpack_rule_state(z)
    out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES)
    out_n = decode_natural_outputs(natural_net(z[..., :6]), NATURAL_OUTPUT_NAMES)
    drv = derive_rule(st, out, params, Y_n=out_n["Y_n"], R_n=out_n["R_n_real"], policy=policy)
    data: TensorDict = {}
    data.update(_state_dict(z, RULE_STATE_NAMES))
    for k, v in out.items():
        data[k if k not in data else f"out_{k}"] = v
    data.update({k: v for k, v in out_n.items() if k not in data})
    data.update({k: v for k, v in drv.items() if k not in data})
    data["Y_n"] = out_n["Y_n"]
    data["R_n_real"] = out_n["R_n_real"]
    data = _add_common_ratios(data)
    shaped = {k: v.reshape(T, B) for k, v in data.items()}
    return _numpy_dict(_state_dict(states, RULE_STATE_NAMES)), _numpy_dict(shaped)


def evaluate_optimal_path(
    states: torch.Tensor,
    *,
    kind: str,
    policy_net,
    natural_net,
    params: BaselineParams,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    T, B, K = states.shape
    z = states.reshape(T * B, K)
    z_phys = z[..., :7]
    st = unpack_rule_state(z_phys)
    if kind == "discretion":
        out = decode_discretion(policy_net(z))
        state_names = RULE_STATE_NAMES
    elif kind == "commitment":
        out = decode_commitment(policy_net(z))
        state_names = COMMITMENT_STATE_NAMES
    else:
        raise ValueError("kind must be discretion or commitment.")
    out_n = decode_natural_outputs(natural_net(z_phys[..., :6]), NATURAL_OUTPUT_NAMES)
    drv = derive_free(st, out, params)
    data: TensorDict = {}
    data.update(_state_dict(z, state_names))
    data.update(out)
    data.update({k: v for k, v in out_n.items() if k not in data})
    data.update({k: v for k, v in drv.items() if k not in data})
    data["Y_n"] = out_n["Y_n"]
    data["R_n_real"] = out_n["R_n_real"]
    data = _add_common_ratios(data)
    shaped = {k: v.reshape(T, B) for k, v in data.items()}
    return _numpy_dict(_state_dict(states, state_names)), _numpy_dict(shaped)


def save_policy_artifacts(
    *,
    policy: str,
    states_np: dict[str, np.ndarray],
    outputs_np: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{policy}_states.npz", **states_np)
    np.savez_compressed(out_dir / f"{policy}_definitions.npz", **outputs_np)
    with (out_dir / f"{policy}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_summary_stats(outputs_np), fh, indent=2, sort_keys=True)


def save_ir_artifacts(
    *,
    policy: str,
    labels: list[str],
    states_np: dict[str, np.ndarray],
    outputs_np: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_states: dict[str, np.ndarray] = {"labels": np.asarray(labels)}
    flat_defs: dict[str, np.ndarray] = {"labels": np.asarray(labels)}
    for i, label in enumerate(labels):
        for name, arr in states_np.items():
            flat_states[f"{label}__{name}"] = arr[:, i]
        for name, arr in outputs_np.items():
            flat_defs[f"{label}__{name}"] = arr[:, i]
    np.savez_compressed(out_dir / f"IR_{policy}_states.npz", **flat_states)
    np.savez_compressed(out_dir / f"IR_{policy}_definitions.npz", **flat_defs)


def run_postprocess(
    *,
    artifact_root: Path,
    output_dir: Path,
    experiment: str,
    params_json: Path | None,
    length: int,
    batch_size: int,
    seed: int,
    ir_burnin: int,
    ir_horizon: int,
    ir_presteps: int,
    ir_relief_lag: int,
    save_ir: bool,
    device: str,
    dtype: torch.dtype,
) -> None:
    params, experiment_meta = resolve_params(experiment, params_json)
    natural_path = _first_existing([artifact_root / "natural" / "natural.pt", artifact_root / "natural.pt"])
    fixed_path = _first_existing([artifact_root / "fixed_taylor" / "fixed.pt", artifact_root / "fixed.pt"])
    ba_path = _first_existing([artifact_root / "modified_taylor" / "ba.pt", artifact_root / "ba.pt"])
    discretion_path = _first_existing([artifact_root / "discretion" / "discretion.pt"])
    commitment_path = _first_existing([artifact_root / "commitment" / "commitment.pt"])

    natural = load_natural(natural_path, device=device, dtype=dtype)
    if params_json is None:
        params = params_from_metadata(natural.metadata, fallback=params)
        experiment_meta["params"] = asdict(params)
    fixed = load_rule(fixed_path, policy="fixed", device=device, dtype=dtype)
    ba = load_rule(ba_path, policy="ba", device=device, dtype=dtype)
    discretion = load_optimal(discretion_path, kind="discretion", device=device, dtype=dtype)
    commitment = load_optimal(commitment_path, kind="commitment", device=device, dtype=dtype)

    torch.manual_seed(int(seed))
    z0 = sample_rule_states(batch_size, params=params, device=device, dtype=dtype, seed=seed)
    with torch.no_grad():
        for policy, loaded in (("fixed", fixed), ("ba", ba)):
            states = simulate_rule_episode(z0, loaded.net, natural.net, policy=policy, params=params, length=length)
            states_np, outputs_np = evaluate_rule_path(
                states,
                policy=policy,
                rule_net=loaded.net,
                natural_net=natural.net,
                params=params,
            )
            save_policy_artifacts(policy=policy, states_np=states_np, outputs_np=outputs_np, out_dir=output_dir)

        z0_disc = z0
        states_disc = simulate_optimal_episode(z0_disc, discretion.net, kind="discretion", params=params, length=length)
        states_np, outputs_np = evaluate_optimal_path(
            states_disc,
            kind="discretion",
            policy_net=discretion.net,
            natural_net=natural.net,
            params=params,
        )
        save_policy_artifacts(policy="discretion", states_np=states_np, outputs_np=outputs_np, out_dir=output_dir)

        z0_com = _initial_optimal_states(
            batch_size,
            kind="commitment",
            params=params,
            device=device,
            dtype=dtype,
            promise_init_scale=1.0,
        )
        states_com = simulate_optimal_episode(z0_com, commitment.net, kind="commitment", params=params, length=length)
        states_np, outputs_np = evaluate_optimal_path(
            states_com,
            kind="commitment",
            policy_net=commitment.net,
            natural_net=natural.net,
            params=params,
        )
        save_policy_artifacts(policy="commitment", states_np=states_np, outputs_np=outputs_np, out_dir=output_dir)

        if save_ir:
            for policy, loaded in (("fixed", fixed), ("ba", ba)):
                labels, ir_states = simulate_rule_ir_scenarios(
                    policy=policy,
                    rule_net=loaded.net,
                    natural_net=natural.net,
                    params=params,
                    burnin=ir_burnin,
                    horizon=ir_horizon,
                    presteps=ir_presteps,
                    relief_lag=ir_relief_lag,
                    device=device,
                    dtype=dtype,
                )
                states_np, outputs_np = evaluate_rule_path(
                    ir_states,
                    policy=policy,
                    rule_net=loaded.net,
                    natural_net=natural.net,
                    params=params,
                )
                save_ir_artifacts(policy=policy, labels=labels, states_np=states_np, outputs_np=outputs_np, out_dir=output_dir)

            for kind, loaded in (("discretion", discretion), ("commitment", commitment)):
                labels, ir_states = simulate_optimal_ir_scenarios(
                    kind=kind,
                    policy_net=loaded.net,
                    params=params,
                    burnin=ir_burnin,
                    horizon=ir_horizon,
                    presteps=ir_presteps,
                    relief_lag=ir_relief_lag,
                    device=device,
                    dtype=dtype,
                )
                states_np, outputs_np = evaluate_optimal_path(
                    ir_states,
                    kind=kind,
                    policy_net=loaded.net,
                    natural_net=natural.net,
                    params=params,
                )
                save_ir_artifacts(policy=kind, labels=labels, states_np=states_np, outputs_np=outputs_np, out_dir=output_dir)

    manifest = {
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "experiment": experiment_meta,
        "params": asdict(params),
        "length": int(length),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "policies": ["fixed", "ba", "discretion", "commitment"],
        "ir": {
            "saved": bool(save_ir),
            "burnin": int(ir_burnin),
            "horizon": int(ir_horizon),
            "presteps": int(ir_presteps),
            "relief_lag": int(ir_relief_lag),
            "scenarios": list(_default_ir_scenarios(params, pulse=int(ir_burnin), relief_lag=int(ir_relief_lag)).keys()),
        },
        "files": sorted(str(p.name) for p in output_dir.glob("*")),
    }
    with (output_dir / "postprocess_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulation artifacts from trained critical-input DEQN checkpoints.")
    parser.add_argument("--artifact-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--ir-burnin", type=int, default=400)
    parser.add_argument("--ir-horizon", type=int, default=200)
    parser.add_argument("--ir-presteps", type=int, default=5)
    parser.add_argument("--ir-relief-lag", type=int, default=8)
    parser.add_argument("--no-ir", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    args = parser.parse_args()

    output_dir = args.output_dir or (args.artifact_root / "postprocess")
    run_postprocess(
        artifact_root=args.artifact_root,
        output_dir=output_dir,
        experiment=args.experiment,
        params_json=args.params_json,
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        ir_burnin=args.ir_burnin,
        ir_horizon=args.ir_horizon,
        ir_presteps=args.ir_presteps,
        ir_relief_lag=args.ir_relief_lag,
        save_ir=not args.no_ir,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    print(f"critical_input_deqn postprocess artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
