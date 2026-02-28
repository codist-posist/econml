from __future__ import annotations
from typing import Dict
import torch


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    # Clamp logits to avoid inf in exp while preserving monotonic mapping.
    return torch.exp(torch.clamp(x, min=-40.0, max=40.0))


def positive(x: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    # Strict log-parameterization: positive variable = exp(raw) + floor.
    return _safe_exp(x) + float(floor)


def ge_one(x: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    """Map to >= 1 (used for price dispersion Delta)."""
    return 1.0 + _safe_exp(x) + float(floor)


def decode_outputs(policy: str, raw: torch.Tensor, floors: Dict[str, float]) -> Dict[str, torch.Tensor]:
    if policy in ["taylor", "mod_taylor"]:
        names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta"]
    elif policy == "discretion":
        names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "rho", "zeta"]
    elif policy == "commitment":
        names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "nu", "zeta", "vartheta", "varrho"]
    else:
        raise ValueError(policy)

    if raw.shape[-1] != len(names):
        raise ValueError(f"decode_outputs: raw last dim {raw.shape[-1]} != expected {len(names)} for policy={policy}")

    out = {n: raw[..., i] for i, n in enumerate(names)}

    # Make inflation safe: parameterize 1+pi > 0
    _raw_pi = out["pi"]
    pi_floor = floors.get("one_plus_pi", floors.get("pi", 1e-6))
    out["one_plus_pi"] = positive(_raw_pi, pi_floor)
    out["pi"] = out["one_plus_pi"] - 1.0

    # Positive domain vars
    out["c"]     = positive(out["c"], floors.get("c", 1e-10))

    # Price dispersion: Delta >= 1
    out["Delta"] = ge_one(out["Delta"], floors.get("Delta", 1e-10))

    out["pstar"] = positive(out["pstar"], floors.get("pstar", 1e-10))
    out["lam"]   = positive(out["lam"], floors.get("lam", 1e-12))
    out["w"]     = positive(out["w"], floors.get("w", 1e-12))
    out["XiN"]   = positive(out["XiN"], floors.get("XiN", 1e-14))
    out["XiD"]   = positive(out["XiD"], floors.get("XiD", 1e-14))

    return out
