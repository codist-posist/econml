from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from .config import ModelParams


@dataclass(frozen=True)
class State:
    """
    STATE ORDER CONTRACT (must be consistent project-wide):

    For policies: "taylor", "mod_taylor", "discretion"
        x = [Delta_prev, logA, loggtilde, xi, s]            (dim=5)

    For policy: "commitment"
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev] (dim=7)

    Convention: s in {0,1}, with s=0 normal, s=1 bad (params.bad_state=1 by default).
    """
    Delta_prev: torch.Tensor
    logA: torch.Tensor
    loggtilde: torch.Tensor
    xi: torch.Tensor
    s: torch.Tensor
    vartheta_prev: torch.Tensor | None = None
    varrho_prev: torch.Tensor | None = None


def unpack_state(x: torch.Tensor, policy: str) -> State:
    if policy in ["taylor", "mod_taylor", "discretion"]:
        assert x.shape[-1] == 5, f"Expected state dim 5 for {policy}, got {x.shape[-1]}"
        return State(
            Delta_prev=x[..., 0],
            logA=x[..., 1],
            loggtilde=x[..., 2],
            xi=x[..., 3],
            s=x[..., 4].long(),
        )
    if policy == "commitment":
        assert x.shape[-1] == 7, f"Expected state dim 7 for commitment, got {x.shape[-1]}"
        return State(
            Delta_prev=x[..., 0],
            logA=x[..., 1],
            loggtilde=x[..., 2],
            xi=x[..., 3],
            s=x[..., 4].long(),
            vartheta_prev=x[..., 5],
            varrho_prev=x[..., 6],
        )
    raise ValueError(f"Unknown policy: {policy}")


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

    # Level cost-push shock (Markov-switching): active only in the "bad" regime.
    # Use params.bad_state to avoid silent regime-label inversions.
    bad = (st.s == int(params.bad_state))
    eta = torch.where(bad, torch.full_like(st.xi, params.eta_bar), torch.zeros_like(st.xi))

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

    # Regime-dependent volatilities: sigma can differ between normal (0) and bad (1).
    # We index by s_{t+1} to keep each next-state branch internally consistent.
    s_next_long = s_next.to(torch.long)
    sigA0, sigA1 = params.sigma_by_regime("sigma_A")
    sigg0, sigg1 = params.sigma_by_regime("sigma_g")
    sigt0, sigt1 = params.sigma_by_regime("sigma_tau")
    sigma_A = torch.where(
        s_next_long == 0,
        torch.full_like(epsA, float(sigA0)),
        torch.full_like(epsA, float(sigA1)),
    )
    sigma_g = torch.where(
        s_next_long == 0,
        torch.full_like(epsg, float(sigg0)),
        torch.full_like(epsg, float(sigg1)),
    )
    sigma_tau = torch.where(
        s_next_long == 0,
        torch.full_like(epstau, float(sigt0)),
        torch.full_like(epstau, float(sigt1)),
    )

    # Drift corrections for log AR(1) so levels keep the intended mean under lognormality.
    driftA = (1.0 - params.rho_A) * (-(sigma_A ** 2) / (2.0 * (1.0 - params.rho_A ** 2)))
    driftg = (1.0 - params.rho_g) * (-(sigma_g ** 2) / (2.0 * (1.0 - params.rho_g ** 2)))

    B = st.logA.shape[0]
    view_shape = (B,) + (1,) * (epsA.ndim - 1)  # (B,1,1,...)

    logA = st.logA.view(view_shape)
    logg = st.loggtilde.view(view_shape)
    xi = st.xi.view(view_shape)

    logA_next = driftA + params.rho_A * logA + sigma_A * epsA
    logg_next = driftg + params.rho_g * logg + sigma_g * epsg
    xi_next = params.rho_tau * xi + sigma_tau * epstau

    return logA_next, logg_next, xi_next, s_next.long()
