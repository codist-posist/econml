from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def moments(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def residual_quality(resid: np.ndarray, *, tol: float = 1e-3) -> Dict[str, float]:
    r = np.asarray(resid, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"rms": float("nan"), "max_abs": float("nan"), "share_all_lt_tol": float("nan"), "tol": float(tol)}
    rms = float(np.sqrt(np.mean(r**2)))
    max_abs = float(np.max(np.abs(r)))
    # if resid has shape (B, K), require all equations < tol
    share = float(np.mean(np.all(np.abs(resid) < tol, axis=-1))) if resid.ndim == 2 else float(np.mean(np.abs(resid) < tol))
    return {"rms": rms, "max_abs": max_abs, "share_all_lt_tol": share, "tol": float(tol)}


def summarize_series(series: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    return {k: moments(v) for k, v in series.items()}


def split_by_regime(arr: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s).astype(int)
    a = np.asarray(arr)
    return a[s == 0], a[s == 1]


def ergodic_moments(sim_paths: Dict[str, np.ndarray], *, by_regime: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    s = sim_paths.get("s")
    for k, v in sim_paths.items():
        if k == "s":
            continue
        out[k] = moments(v)
        if by_regime and s is not None and v.shape == s.shape:
            v0, v1 = split_by_regime(v, s)
            out[k + "_s0"] = moments(v0)
            out[k + "_s1"] = moments(v1)
    return out


def moment_stability(
    x: np.ndarray,
    *,
    n_blocks: int = 6,
    fn=np.mean,
) -> Dict[str, float]:
    """Quick diagnostic for whether moments are stable in long simulations.

    Splits the series into `n_blocks` consecutive blocks and computes `fn` per block.
    Returns the mean across blocks and the max block-to-block deviation.

    This is useful because Table-2 skewness (in particular) can be noisy if the saved
    sim_paths are short.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < n_blocks:
        return {"blocks": float(n_blocks), "block_mean": float("nan"), "max_dev": float("nan")}
    blocks = np.array_split(x, n_blocks)
    vals = np.array([fn(b) for b in blocks], dtype=np.float64)
    mu = float(np.mean(vals))
    dev = float(np.max(np.abs(vals - mu)))
    return {"blocks": float(n_blocks), "block_mean": mu, "max_dev": dev}




def _efficient_c_hat_series_from_A_g(
    params: Any,
    *,
    A: np.ndarray,
    g: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """Compute hat{c}_t in the **efficient allocation** given paths of A_t and g_t.

    Paper (Section 3.1, Efficient allocation) defines hat{c}_t implicitly by:
        ((hat{c}_t + g_t) / A_t)^omega - A_t * hat{c}_t^{-gamma} = 0,
    equivalently:
        (hat{c}_t + g_t)^omega - A_t^{omega+1} * hat{c}_t^{-gamma} = 0.

    IMPORTANT: By construction, this efficient allocation depends on (A_t, g_t) only
    and **does not depend** on cost-push shocks (xi_t) or regimes (eta_t).
    """
    # Be robust to (T,1) shapes that can arise from saved torch tensors.
    # The efficient allocation is defined pointwise in time, so we treat A and g as 1D time series.
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if A.size != g.size:
        raise ValueError("A and g must have the same number of observations")

    gamma = float(getattr(params, "gamma"))
    omega = float(getattr(params, "omega"))

    T = int(A.size)
    c_eff = np.empty(T, dtype=np.float64)

    # generic warm start
    c0 = 1.0
    for t in range(T):
        c = c0 if t == 0 else c_eff[t - 1]
        # A[t] / g[t] might still be 0-d arrays; force scalar.
        At = float(np.asarray(A[t]).reshape(()))
        gt = float(np.asarray(g[t]).reshape(()))

        for _ in range(max_iter):
            term = (At ** (omega + 1.0))
            f = (c + gt) ** omega - term * (c ** (-gamma))
            if abs(f) < tol:
                break
            df = omega * (c + gt) ** (omega - 1.0) + term * gamma * (c ** (-gamma - 1.0))
            step = f / df
            c_new = c - step
            if c_new <= 1e-12 or not np.isfinite(c_new):
                c_new = max(1e-12, 0.5 * c)
            c = c_new

        c_eff[t] = c
    return c_eff


def output_gap_from_consumption(
    sim_paths: Dict[str, np.ndarray],
    efficient_sss: Dict[str, float] | float,
    *,
    params: Any | None = None,
    time_varying: bool = True,
) -> np.ndarray:
    """Efficient (consumption) gap as defined in the paper: x_t = log(c_t) - log(hat{c}_t).

    hat{c}_t is consumption in the **efficient allocation** (Section 3.1). It depends on (A_t, g_t)
    and does **not** depend on cost-push shocks or regimes.
    """
    # sim_paths may store series as (T,1); make them 1D.
    c = np.asarray(sim_paths["c"], dtype=np.float64).reshape(-1)

    # constant fallback (only sensible for SSS / fixed A,g)
    if isinstance(efficient_sss, (float, int, np.floating)):
        c_hat_const = float(efficient_sss)
    else:
        if "c_hat" not in efficient_sss:
            raise KeyError("efficient_sss must contain key 'c_hat' (efficient consumption).")  # type: ignore[arg-type]
        c_hat_const = float(efficient_sss["c_hat"])  # type: ignore[index]

    if not time_varying:
        return np.log(c) - np.log(c_hat_const)

    if params is None:
        raise ValueError("params must be provided when time_varying=True (to compute efficient hat{c}_t from A_t and g_t).")

    if "A" in sim_paths:
        A = np.asarray(sim_paths.get("A"), dtype=np.float64)
    elif "logA" in sim_paths:
        A = np.exp(np.asarray(sim_paths.get("logA"), dtype=np.float64))
    else:
        A = np.asarray(None)
    if "g" in sim_paths:
        g = np.asarray(sim_paths.get("g"), dtype=np.float64)
    elif "loggtilde" in sim_paths:
        # g = g_bar * exp(loggtilde)
        if params is None:
            raise ValueError("params required to build g from loggtilde")
        g = float(getattr(params, "g_bar")) * np.exp(np.asarray(sim_paths.get("loggtilde"), dtype=np.float64))
    else:
        g = np.asarray(None)
    if A is None or g is None or (not hasattr(A, "shape")) or (not hasattr(g, "shape")):
        raise ValueError("sim_paths must contain 'A' (or 'logA') and 'g' (or 'loggtilde') when time_varying=True.")
    # Allow (T,1) vs (T,) mismatches by comparing sizes after flattening.
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if A.size != c.size or g.size != c.size:
        raise ValueError("A and g must have the same length as c when time_varying=True.")

    c_hat_t = _efficient_c_hat_series_from_A_g(params, A=A, g=g)
    return np.log(c) - np.log(c_hat_t)


def table2_dataframe(sss_by_policy: Dict[str, Dict[str, Any]]):
    """Build a robust Table-2-like DataFrame from arbitrary SSS dicts.

    Expects each policy dict to have either:
      - {"by_regime": {0:{...}, 1:{...}}} (preferred), or
      - {0:{...}, 1:{...}} directly.
    Produces a tidy DataFrame with rows (policy, regime) and columns = union of keys.
    """
    if pd is None:
        raise ImportError("pandas is required for table2_dataframe()")
    rows = []
    for pol, obj in sss_by_policy.items():
        by_reg = obj.get("by_regime", obj)
        for s in [0, 1]:
            d = dict(by_reg.get(str(s), by_reg.get(s, {})))
            d["policy"] = pol
            d["regime"] = s
            rows.append(d)
    df = pd.DataFrame(rows).set_index(["policy", "regime"]).sort_index()
    return df
