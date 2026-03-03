from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from .config import ModelParams


@dataclass(frozen=True)
class State:
    """
    STATE ORDER CONTRACT (must be consistent project-wide):

    For policies: "taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb",
                  "discretion", "discretion_zlb"
        x = [Delta_prev, logA, loggtilde, xi, s]            (dim=5)

    For policy: "commitment"
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev] (dim=7)
        or
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev, c_prev] (dim=8)

    For policy: "commitment_zlb"
        x = [Delta_prev, logA, loggtilde, xi, s, vartheta_prev, varrho_prev, c_prev, i_nom_prev, varphi_prev] (dim=10)

    Convention: s in {0,1}, with s=0 normal, s=1 bad (params.bad_state=1 by default).
    """
    Delta_prev: torch.Tensor
    logA: torch.Tensor
    loggtilde: torch.Tensor
    xi: torch.Tensor
    s: torch.Tensor
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

    # Drift corrections per paper (Appendix A.1) and author code.
    driftA = _ar1_lognorm_drift(float(params.rho_A), float(params.sigma_A))
    driftg = _ar1_lognorm_drift(float(params.rho_g), float(params.sigma_g))
    drift_xi = _ar1_lognorm_drift(float(params.rho_tau), float(params.sigma_tau))

    B = st.logA.shape[0]
    view_shape = (B,) + (1,) * (epsA.ndim - 1)  # (B,1,1,...)

    logA = st.logA.view(view_shape)
    logg = st.loggtilde.view(view_shape)
    xi = st.xi.view(view_shape)

    logA_next = driftA + params.rho_A * logA + params.sigma_A * epsA
    logg_next = driftg + params.rho_g * logg + params.sigma_g * epsg
    xi_next = drift_xi + params.rho_tau * xi + params.sigma_tau * epstau

    return logA_next, logg_next, xi_next, s_next.long()
