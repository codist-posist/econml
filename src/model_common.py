from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .config import ModelParams


@dataclass(frozen=True)
class State:
    """
    STATE ORDER CONTRACT (must be consistent project-wide):

    For policies: "taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb",
                  "discretion", "discretion_zlb"
        x = [Delta_prev, logA, loggtilde, xi, s]            (dim=5)

    For policy: "taylor_para"
        x = [Delta_prev, logA, loggtilde, xi, s, i_old, p21] (dim=7)

    For policy: "commitment"
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev] (dim=7)
        or
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev, c_prev] (dim=8)

    For policy: "commitment_zlb"
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev, c_prev, i_nom_prev, varphi_prev] (dim=10)

    Convention: s is an integer regime id in [0, ..., params.n_regimes-1].
    """

    Delta_prev: torch.Tensor
    logA: torch.Tensor
    loggtilde: torch.Tensor
    xi: torch.Tensor
    s: torch.Tensor
    i_old: torch.Tensor | None = None
    p21: torch.Tensor | None = None
    vartheta_prev: torch.Tensor | None = None
    varrho_prev: torch.Tensor | None = None
    c_prev: torch.Tensor | None = None
    i_nom_prev: torch.Tensor | None = None
    varphi_prev: torch.Tensor | None = None


def unpack_state(x: torch.Tensor, policy: str) -> State:
    if policy in ["taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "discretion", "discretion_zlb"]:
        if x.shape[-1] != 5:
            raise AssertionError(f"Expected state dim 5 for policy={policy}, got {x.shape[-1]}")
        return State(
            Delta_prev=x[..., 0],
            logA=x[..., 1],
            loggtilde=x[..., 2],
            xi=x[..., 3],
            s=x[..., 4].long(),
        )
    if policy == "taylor_para":
        if x.shape[-1] not in (5, 7):
            raise AssertionError(f"Expected state dim 5 or 7 for policy=taylor_para, got {x.shape[-1]}")
        return State(
            Delta_prev=x[..., 0],
            logA=x[..., 1],
            loggtilde=x[..., 2],
            xi=x[..., 3],
            s=x[..., 4].long(),
            i_old=x[..., 5] if x.shape[-1] >= 6 else None,
            p21=x[..., 6] if x.shape[-1] >= 7 else None,
        )
    if policy == "commitment":
        if x.shape[-1] not in (7, 8):
            raise AssertionError(f"Expected state dim 7 or 8 for commitment, got {x.shape[-1]}")
        return State(
            Delta_prev=x[..., 0],
            logA=x[..., 1],
            loggtilde=x[..., 2],
            xi=x[..., 3],
            s=x[..., 4].long(),
            vartheta_prev=x[..., 5],
            varrho_prev=x[..., 6],
            c_prev=x[..., 7] if x.shape[-1] == 8 else None,
        )
    if policy == "commitment_zlb":
        if x.shape[-1] != 10:
            raise AssertionError(f"Expected state dim 10 for commitment_zlb, got {x.shape[-1]}")
        return State(
            Delta_prev=x[..., 0],
            logA=x[..., 1],
            loggtilde=x[..., 2],
            xi=x[..., 3],
            s=x[..., 4].long(),
            vartheta_prev=x[..., 5],
            varrho_prev=x[..., 6],
            c_prev=x[..., 7],
            i_nom_prev=x[..., 8],
            varphi_prev=x[..., 9],
        )
    raise ValueError(f"Unknown policy: {policy}")


def _regime_lookup(levels: Tuple[float, ...], s: torch.Tensor, *, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = torch.tensor(levels, device=device, dtype=dtype)
    idx = torch.clamp(s.to(device=device, dtype=torch.long), min=0, max=int(vals.numel()) - 1)
    return vals[idx]


def regime_eta(params: ModelParams, s: torch.Tensor) -> torch.Tensor:
    return _regime_lookup(
        tuple(float(v) for v in params.eta_by_regime),
        s,
        device=s.device,
        dtype=params.dtype,
    )


def transition_probs_to_next_regimes(
    params: ModelParams,
    s: torch.Tensor,
    *,
    xi: torch.Tensor | None = None,
    p21_state: torch.Tensor | None = None,
    p12_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Regime transition probabilities with optional per-path p12/p21 overrides.

    Returns probs with shape s.shape + (R,), where probs[..., j] = Pr(s_{t+1}=j | state_t)
    and R = params.n_regimes.

    For R>=3 we preserve the escalation structure:
      - row 0 uses p12 (0->1)
      - row 1 uses p21 (1->0) and keeps row-1 mass to higher regimes from params.P
      - rows >=2 follow params.P (e.g., severe row uses p32 from config)
    """
    s_t = s.to(device=params.device, dtype=torch.long)
    dt = params.dtype
    for cand in (xi, p21_state, p12_state):
        if torch.is_tensor(cand) and cand.dtype.is_floating_point:
            dt = cand.dtype
            break
    zeros = torch.zeros_like(s_t, dtype=dt)
    P = params.P.to(device=s_t.device, dtype=dt)
    R = int(P.shape[0])
    if R <= 0:
        raise ValueError("params.P must have positive size")
    if P.shape[1] != R:
        raise ValueError(f"params.P must be square, got shape={tuple(P.shape)}")

    s_idx = torch.clamp(s_t, min=0, max=R - 1)
    probs = P[s_idx, :].clone()

    p21 = torch.full_like(zeros, float(params.p21))
    if p21_state is not None:
        p21 = torch.as_tensor(p21_state, device=s_t.device, dtype=dt) + zeros * 0.0

    p12 = torch.full_like(zeros, float(params.p12))
    if p12_state is not None:
        p12 = torch.as_tensor(p12_state, device=s_t.device, dtype=dt) + zeros * 0.0

    eps = 1e-9
    p21 = torch.clamp(p21, min=eps, max=1.0 - eps)
    p12 = torch.clamp(p12, min=eps, max=1.0 - eps)

    if R == 1:
        return probs

    mask0 = (s_idx == 0)
    if mask0.any():
        probs[mask0, 0] = 1.0 - p12[mask0]
        probs[mask0, 1] = p12[mask0]
        if R > 2:
            probs[mask0, 2:] = 0.0

    mask1 = (s_idx == 1)
    if mask1.any():
        tail = probs[mask1, 2:].sum(dim=-1) if R > 2 else torch.zeros_like(p21[mask1])
        p21_max = torch.clamp(1.0 - tail - eps, min=eps, max=1.0 - eps)
        p21_row = torch.minimum(torch.maximum(p21[mask1], torch.full_like(p21[mask1], eps)), p21_max)
        probs[mask1, 0] = p21_row
        probs[mask1, 1] = 1.0 - p21_row - tail

    probs = torch.clamp(probs, min=0.0)
    probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
    return probs


def identities(params: ModelParams, st: State, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Pure identities (NOT residuals), strictly per paper.

    Requires out contains at least:
      - out["c"]
      - out["Delta"]
    """
    A = torch.exp(st.logA)
    gtilde = torch.exp(st.loggtilde)
    g = params.g_bar * gtilde

    eta = regime_eta(params, st.s).to(device=st.xi.device, dtype=st.xi.dtype)

    # (1 + tau_t) = 1 - tau_bar + xi_t + eta_t
    one_plus_tau = 1.0 - params.tau_bar + st.xi + eta

    # Goods market: y = c + g
    y = out["c"] + g

    # Aggregation with price dispersion: y = A*h/Delta  =>  h = y*Delta/A
    h = y * out["Delta"] / A

    return {
        "A": A,
        "gtilde": gtilde,
        "g": g,
        "eta": eta,
        "one_plus_tau": one_plus_tau,
        "y": y,
        "h": h,
    }


def _ar1_lognorm_drift(rho: float, sigma: float) -> float:
    """Paper/author-code drift term for AR(1) in logs."""
    return float((1.0 - rho) * (-(sigma**2) / 2.0))


def shock_laws_of_motion(
    params: ModelParams,
    st: State,
    epsA: torch.Tensor,
    epsg: torch.Tensor,
    epstau: torch.Tensor,
    s_next: torch.Tensor,
):
    """
    Laws of motion for exogenous shocks, strictly per paper.

    Supports:
      - (B,) simulation draws
      - (B,Q) flattened quadrature grids
      - (B,n,n,n) full tensor-product grids
      - ... any shape (B, ...)

    Contract:
      epsA, epsg, epstau must have identical shapes.
    """
    assert epsA.shape == epsg.shape == epstau.shape, "eps tensors must have identical shapes"
    assert st.logA.ndim == 1 and st.loggtilde.ndim == 1 and st.xi.ndim == 1, \
        "State shock components must be 1D (B,) at this stage"

    driftA = _ar1_lognorm_drift(float(params.rho_A), float(params.sigma_A))
    driftg = _ar1_lognorm_drift(float(params.rho_g), float(params.sigma_g))

    B = st.logA.shape[0]
    view_shape = (B,) + (1,) * (epsA.ndim - 1)  # (B,1,1,...)

    logA = st.logA.view(view_shape)
    logg = st.loggtilde.view(view_shape)
    xi = st.xi.view(view_shape)

    sigma_xi = float(params.sigma_tau)
    drift_xi_next = (1.0 - float(params.rho_tau)) * (-(sigma_xi ** 2) / 2.0)

    logA_next = driftA + params.rho_A * logA + params.sigma_A * epsA
    logg_next = driftg + params.rho_g * logg + params.sigma_g * epsg
    xi_next = drift_xi_next + params.rho_tau * xi + sigma_xi * epstau

    return logA_next, logg_next, xi_next, s_next.long()
