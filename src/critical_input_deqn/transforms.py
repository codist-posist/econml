from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def positive(raw: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    """Smooth nonnegative transform with a small strictly positive floor."""

    return F.softplus(raw) + floor


def gross_from_log(raw: torch.Tensor) -> torch.Tensor:
    """Positive gross rate/inflation transform."""

    return torch.exp(raw)


def decode_rule_outputs(raw: torch.Tensor, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        x = raw[..., i]
        if name == "Pi":
            out[name] = gross_from_log(x)
        else:
            out[name] = positive(x)
    return out


def decode_natural_outputs(raw: torch.Tensor, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        out[name] = positive(raw[..., i])
    return out


def decode_optimal_outputs(raw: torch.Tensor, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    positive_names = {"C", "Y", "R", "Pi", "chi", "I_A", "Q_A", "S_p", "F_p"}
    for i, name in enumerate(names):
        x = raw[..., i]
        if name == "Pi":
            out[name] = gross_from_log(x)
        elif name in positive_names:
            out[name] = positive(x)
        else:
            out[name] = x
    return out
