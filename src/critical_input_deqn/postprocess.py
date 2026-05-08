from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .config import (
    COMMITMENT_OUTPUT_NAMES,
    COMMITMENT_STATE_NAMES,
    DISCRETION_OUTPUT_NAMES,
    NATURAL_OUTPUT_NAMES,
    RULE_OUTPUT_NAMES,
    RULE_STATE_NAMES,
    BaselineParams,
    NetworkConfig,
)
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


def run_postprocess(
    *,
    artifact_root: Path,
    output_dir: Path,
    length: int,
    batch_size: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    params = BaselineParams()
    natural_path = _first_existing([artifact_root / "natural" / "natural.pt", artifact_root / "natural.pt"])
    fixed_path = _first_existing([artifact_root / "fixed_taylor" / "fixed.pt", artifact_root / "fixed.pt"])
    ba_path = _first_existing([artifact_root / "modified_taylor" / "ba.pt", artifact_root / "ba.pt"])
    discretion_path = _first_existing([artifact_root / "discretion" / "discretion.pt"])
    commitment_path = _first_existing([artifact_root / "commitment" / "commitment.pt"])

    natural = load_natural(natural_path, device=device, dtype=dtype)
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

    manifest = {
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "length": int(length),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "policies": ["fixed", "ba", "discretion", "commitment"],
        "files": sorted(str(p.name) for p in output_dir.glob("*")),
    }
    with (output_dir / "postprocess_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulation artifacts from trained critical-input DEQN checkpoints.")
    parser.add_argument("--artifact-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    args = parser.parse_args()

    output_dir = args.output_dir or (args.artifact_root / "postprocess")
    run_postprocess(
        artifact_root=args.artifact_root,
        output_dir=output_dir,
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    print(f"critical_input_deqn postprocess artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
