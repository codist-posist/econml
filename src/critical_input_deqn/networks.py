from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .config import NetworkConfig


def _activation(name: str) -> Callable[[], nn.Module]:
    key = name.lower().strip()
    if key == "silu":
        return nn.SiLU
    if key == "selu":
        return nn.SELU
    if key == "tanh":
        return nn.Tanh
    if key == "relu":
        return nn.ReLU
    raise ValueError(f"Unknown activation {name!r}.")


class MLP(nn.Module):
    """Small DEQN policy network.

    The architecture is deliberately plain: the economic structure lives in the
    residuals, not in a custom neural-network trick.
    """

    def __init__(self, d_in: int, d_out: int, cfg: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        act = _activation(cfg.activation)
        width = int(cfg.hidden_width)
        prev = int(d_in)
        for _ in range(int(cfg.hidden_depth)):
            lin = nn.Linear(prev, width)
            nn.init.xavier_uniform_(lin.weight, gain=float(cfg.init_scale))
            nn.init.zeros_(lin.bias)
            layers.extend([lin, act()])
            prev = width
        out = nn.Linear(prev, int(d_out))
        nn.init.xavier_uniform_(out.weight, gain=float(cfg.init_scale))
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

