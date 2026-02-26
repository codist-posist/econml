from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GHQuadrature:
    """
    One-dimensional Gauss–Hermite quadrature for expectation under N(0,1).

    nodes  : transformed nodes (sqrt(2) * x_i)
    weights: normalized weights (w_i / sqrt(pi))

    So that:
        E[f(eps)] ≈ sum_i weights[i] * f(nodes[i]),
        where eps ~ N(0,1).
    """
    nodes: torch.Tensor   # (n,)
    weights: torch.Tensor # (n,)


def gauss_hermite(n: int, device: str, dtype: torch.dtype) -> GHQuadrature:
    import numpy as np
    from numpy.polynomial.hermite import hermgauss

    x, w = hermgauss(n)

    # Transform for N(0,1):
    # ∫ f(eps) φ(eps) d eps = (1/√π) ∫ e^{-x^2} f(√2 x) dx
    nodes = torch.tensor((2.0 ** 0.5) * x, device=device, dtype=dtype)
    weights = torch.tensor(w / (np.pi ** 0.5), device=device, dtype=dtype)

    # Sanity check: weights must sum to 1
    if not torch.allclose(weights.sum(), torch.tensor(1.0, device=device, dtype=dtype), atol=1e-6):
        raise ValueError("GH weights do not sum to 1 — normalization error")

    return GHQuadrature(nodes=nodes, weights=weights)
