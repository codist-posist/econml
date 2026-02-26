
from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import ModelParams, PolicyName
from .io_utils import load_selected_run, find_latest_run_dir, load_torch, load_json
from .steady_states import solve_flexprice_sss, solve_efficient_sss, solve_commitment_sss_from_policy, solve_discretion_sss_from_policy
from .sss_from_policy import switching_policy_sss_by_regime_from_policy, frozen_policy_sss_by_regime_from_policy
from .deqn import PolicyNetwork, implied_nominal_rate_from_euler
from .deqn import Trainer
from .config import TrainConfig
from .metrics import output_gap_from_consumption



# Network dimensions used across notebooks/training
DIMS: Dict[str, Tuple[int, int]] = {
    "taylor": (5, 8),
    "mod_taylor": (5, 8),
    "discretion": (5, 11),
    "commitment": (7, 13),
}

# ---- Flex-prices simulation (needed for Table 2 moments) ----

def _tensor_grid(nodes: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 3D tensor-product nodes/weights for independent N(0,1) shocks."""
    n = nodes.numel()
    a = nodes.view(n, 1, 1).expand(n, n, n)
    b = nodes.view(1, n, 1).expand(n, n, n)
    c = nodes.view(1, 1, n).expand(n, n, n)
    w = (weights.view(n, 1, 1) * weights.view(1, n, 1) * weights.view(1, 1, n)).reshape(-1)
    return a.reshape(-1), b.reshape(-1), c.reshape(-1), w

def _solve_flex_c_batch(
    params: ModelParams,
    *,
    logA: torch.Tensor,
    logg: torch.Tensor,
    xi: torch.Tensor,
    s: torch.Tensor,
    h_init: torch.Tensor,
    newton_iters: int = 30,
) -> torch.Tensor:
    """
    Flexible-price allocation each period by static conditions implied in the paper:
      - optimal pricing under flex prices implies w*(1+tau)/A = 1/M  ->  w = A / (M(1+tau))
      - labor supply: h^omega = w * c^{-gamma}
      - goods: c = A h - g, with g = gbar * exp(logg)
    Solve for h by Newton, then return c.
    """
    dev, dt = params.device, params.dtype
    A = torch.exp(logA)
    g = params.g_bar * torch.exp(logg)
    # (1+tau_t) = 1 - tau_bar + xi_t + eta_t
    bad = (s.to(torch.long) == int(params.bad_state))
    eta = torch.where(bad, torch.tensor(params.eta_bar, device=dev, dtype=dt), torch.zeros_like(xi))
    one_plus_tau = (1.0 - params.tau_bar) + xi + eta
    M = params.M
    w = A / (M * one_plus_tau)

    h = torch.clamp(h_init, min=1e-4)
    for _ in range(int(newton_iters)):
        c = A * h - g
        c = torch.clamp(c, min=1e-8)
        f = h.pow(params.omega) - w * c.pow(-params.gamma)
        # derivative df/dh
        df = params.omega * h.pow(params.omega - 1.0) - w * (-params.gamma) * A * c.pow(-params.gamma - 1.0)
        step = f / torch.clamp(df, min=1e-8)
        h_new = torch.clamp(h - step, min=1e-6)
        # keep feasibility A*h > g by pushing h up a bit if needed
        h = torch.maximum(h_new, (g / torch.clamp(A, min=1e-8)) + 1e-6)
    c = torch.clamp(A * h - g, min=1e-8)
    return c

@torch.inference_mode()
def _simulate_flex_prices_for_table2(
    params: ModelParams,
    *,
    T: int = 6000,
    burn_in: int = 1000,
    B: int = 2048,
    gh_n: int = 3,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    """
    Simulate the flexible-price economy to produce ergodic moments for Table 2.
    This is needed because Table 2 in the paper reports Mean/Std/Skew for the flex-price benchmark.

    We keep the Markov switching (eta regime) and temporary shocks (A, g, xi) exactly as in the model,
    set inflation to zero (flex prices), and compute the natural nominal/real rate from the Euler equation:
        1 + i_t = (1/beta) * lambda_t / E_t[lambda_{t+1}]
    with expectations evaluated via Gauss–Hermite quadrature and the Markov transition matrix.

    Returns a sim dict with keys compatible with table2_builder moment code: c, pi, i, s.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev, dt = params.device, params.dtype
    from .quadrature import gauss_hermite

    gh = gauss_hermite(int(gh_n), device=str(dev), dtype=dt)
    epsA_nodes, epsg_nodes, epst_nodes, w_nodes = _tensor_grid(gh.nodes, gh.weights)  # (N,)
    N = w_nodes.numel()

    # start near SSS in normal regime
    flex = solve_flexprice_sss(params)
    h0 = torch.tensor(flex.by_regime[0]["h"], device=dev, dtype=dt)

    logA = torch.full((B,), float(-(params.sigma_A**2)/(2.0*(1.0-params.rho_A**2))), device=dev, dtype=dt)
    logg = torch.full((B,), float(-(params.sigma_g**2)/(2.0*(1.0-params.rho_g**2))), device=dev, dtype=dt)
    xi = torch.zeros((B,), device=dev, dtype=dt)
    s = torch.zeros((B,), device=dev, dtype=torch.long)  # start normal
    h_guess = h0.expand(B)

    # drift corrections (same as in model_common)
    driftA = (1.0 - params.rho_A) * (-(params.sigma_A**2) / (2.0 * (1.0 - params.rho_A**2)))
    driftg = (1.0 - params.rho_g) * (-(params.sigma_g**2) / (2.0 * (1.0 - params.rho_g**2)))

    c_list = []
    s_list = []
    i_list = []
    logA_list = []
    logg_list = []
    xi_list = []

    for t in range(int(T + burn_in)):
        # current c and lambda
        c = _solve_flex_c_batch(params, logA=logA, logg=logg, xi=xi, s=s, h_init=h_guess)
        lam = c.pow(-params.gamma)

        # Expectation of lambda_{t+1} via GH + Markov sum
        # Build next exogenous states for each node (broadcast B x N)
        logA_n = driftA + params.rho_A * logA[:, None] + params.sigma_A * epsA_nodes[None, :]
        logg_n = driftg + params.rho_g * logg[:, None] + params.sigma_g * epsg_nodes[None, :]
        xi_n   = params.rho_tau * xi[:, None] + params.sigma_tau * epst_nodes[None, :]

        # Two possible next regimes
        P = params.P  # shape (2,2) with orientation P[s_current, s_next] (row-stochastic)
        p0 = P[s.to(torch.long), 0]  # (B,)
        p1 = P[s.to(torch.long), 1]  # (B,)

        # compute lambda_next for s_next=0 and s_next=1
        s0 = torch.zeros((B, N), device=dev, dtype=dt)
        s1 = torch.ones((B, N), device=dev, dtype=dt)

        # initial h guess for next period: keep last h_guess
        h_init_n = h_guess[:, None].expand(B, N).reshape(-1)
        c0 = _solve_flex_c_batch(
            params,
            logA=logA_n.reshape(-1),
            logg=logg_n.reshape(-1),
            xi=xi_n.reshape(-1),
            s=s0.reshape(-1),
            h_init=h_init_n,
        ).reshape(B, N)
        c1 = _solve_flex_c_batch(
            params,
            logA=logA_n.reshape(-1),
            logg=logg_n.reshape(-1),
            xi=xi_n.reshape(-1),
            s=s1.reshape(-1),
            h_init=h_init_n,
        ).reshape(B, N)
        lam0 = c0.pow(-params.gamma)
        lam1 = c1.pow(-params.gamma)

        # E_t[lambda_{t+1}] = sum_{s'} P(s'|s_t) * E[lambda_{t+1}(s') | s']
        # NOTE: (lam0*w_nodes).sum(dim=1) is (B,), so the transition probabilities must also be (B,)
        # to avoid unintended broadcasting to (B,B).
        Et_lam_next = (
            p0 * (lam0 * w_nodes[None, :]).sum(dim=1)
            + p1 * (lam1 * w_nodes[None, :]).sum(dim=1)
        )

        i = (1.0 / params.beta) * (lam / torch.clamp(Et_lam_next, min=1e-12)) - 1.0

        # store after burn-in
        if t >= burn_in:
            c_list.append(c.detach().cpu().numpy())
            s_list.append(s.detach().cpu().numpy())
            i_list.append(i.detach().cpu().numpy())
            logA_list.append(logA.detach().cpu().numpy())
            logg_list.append(logg.detach().cpu().numpy())
            xi_list.append(xi.detach().cpu().numpy())

        # draw realized shocks for next period
        epsA = torch.randn((B,), device=dev, dtype=dt)
        epsg = torch.randn((B,), device=dev, dtype=dt)
        epst = torch.randn((B,), device=dev, dtype=dt)
        logA = driftA + params.rho_A * logA + params.sigma_A * epsA
        logg = driftg + params.rho_g * logg + params.sigma_g * epsg
        xi = params.rho_tau * xi + params.sigma_tau * epst

        # Markov draw for next regime
        u = torch.rand((B,), device=dev, dtype=dt)
        pstay0 = P[s.to(torch.long), 0]
        s = torch.where(u < pstay0, torch.zeros_like(s), torch.ones_like(s))

        # update h_guess roughly with current h implied
        # (approx: h = (c+g)/A)
        A = torch.exp(logA)
        g = params.g_bar * torch.exp(logg)
        h_guess = torch.clamp((c + g) / torch.clamp(A, min=1e-8), min=1e-6)

    c_arr = np.concatenate(c_list, axis=0)
    s_arr = np.concatenate(s_list, axis=0).astype(np.int64)
    i_arr = np.concatenate(i_list, axis=0)
    logA_arr = np.concatenate(logA_list, axis=0)
    logg_arr = np.concatenate(logg_list, axis=0)
    xi_arr = np.concatenate(xi_list, axis=0)

    # pi is identically zero in flex prices
    pi_arr = np.zeros_like(i_arr)

    return {"c": c_arr, "pi": pi_arr, "i": i_arr, "s": s_arr, "logA": logA_arr, "loggtilde": logg_arr, "xi": xi_arr}


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        return 0.0
    z = (x - mu) / sd
    return float(np.mean(z**3))


def _moments_with_skew(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "skew": float("nan")}
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "skew": _skew(x)}


def _split_by_regime(x: np.ndarray, s: np.ndarray, regime: int) -> np.ndarray:
    """Return the subset of x where regime indicator s equals `regime`.

    Robust to inputs coming in as shape (T,) or (T,1); we always flatten to 1D.
    """
    s = np.asarray(s).astype(int).reshape(-1)
    x = np.asarray(x).reshape(-1)
    # Align lengths defensively (smoke runs can create tiny arrays)
    T = min(len(s), len(x))
    if T == 0:
        return x[:0]
    s = s[:T]
    x = x[:T]
    return x[s == int(regime)]


def _annualize_pct(x: np.ndarray | float) -> np.ndarray | float:
    # model units: quarterly net (e.g., pi=0.007); table units: annualized percent
    return 400.0 * x


def _load_run_dir(artifacts_root: str, policy: str, *, use_selected: bool = True) -> str:
    rd: Optional[str] = None
    if use_selected:
        rd = load_selected_run(artifacts_root, policy)
    if rd is None:
        rd = find_latest_run_dir(artifacts_root, policy)
    if rd is None:
        raise FileNotFoundError(f"No run directory found for policy='{policy}' under {artifacts_root}")
    return rd


def _load_net_from_run(run_dir: str, params: ModelParams, policy: PolicyName) -> PolicyNetwork:
    """Load a trained policy network from a run directory.

    The network architecture (hidden layers, activation) is read from config.json
    saved alongside the run. Input/output dimensions are policy-specific and match
    the training notebooks.
    """
    w_path = os.path.join(run_dir, "weights.pt")
    if not os.path.exists(w_path):
        # back-compat
        w_path = os.path.join(run_dir, "weights_best.pt")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Missing weights in {run_dir} (expected weights.pt or weights_best.pt)")

    # Load training config to reconstruct the exact architecture
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = TrainConfig.dev()
    if os.path.exists(cfg_path):
        try:
            packed = load_json(cfg_path)
            train_cfg = packed.get("train_cfg", {})
            hidden = tuple(train_cfg.get("hidden_layers", cfg.hidden_layers))
            activation = str(train_cfg.get("activation", cfg.activation))
            cfg = TrainConfig.dev(hidden_layers=hidden, activation=activation)
        except Exception:
            # If parsing fails, fall back to dev defaults
            cfg = TrainConfig.dev()

    d_in, d_out = DIMS[str(policy)]
    net = PolicyNetwork(d_in, d_out, hidden=cfg.hidden_layers, activation=cfg.activation).to(
        device=params.device, dtype=params.dtype
    )
    state = load_torch(w_path, map_location=params.device)
    net.load_state_dict(state)
    net.eval()
    return net

def _load_sim_paths(run_dir: str) -> Dict[str, np.ndarray]:
    p = os.path.join(run_dir, "sim_paths.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing sim_paths.npz in {run_dir}. Re-run training notebook to save it.")
    data = np.load(p)
    return {k: data[k] for k in data.files}


def _compute_real_rate_series(sim: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Real rate per period computed from realized next inflation:
        r_t = (1+i_t)/(1+pi_{t+1}) - 1
    Returned arrays aligned with s_t (length T-1):
        r, s_aligned
    """
    i = np.asarray(sim["i"])
    pi = np.asarray(sim["pi"])
    s = np.asarray(sim["s"])
    # align to t where pi_{t+1} exists
    r = (1.0 + i[:-1]) / (1.0 + pi[1:]) - 1.0
    s_aligned = s[:-1]
    return r, s_aligned


def _policy_sss_for_policy(
    params: ModelParams,
    policy: PolicyName,
    net: PolicyNetwork,
    regime: int,
) -> Dict[str, float]:
    """Paper-faithful 'SSS by regime' for any policy.

    For Table 2, the paper's 'SSS by regime' concept is the switching-consistent fixed point
    conditional on the current regime.

    - For commitment, we must also pin down the lagged multipliers (timeless perspective), so we
      use solve_commitment_sss_from_policy(), which returns (c,pi,...) plus vartheta_prev/varrho_prev.
    - For other policies, we compute the switching-consistent SSS as a fixed point of the trained
      policy function.
    """
    if policy == "commitment":
        sss_all = solve_commitment_sss_from_policy(params, net)
        return sss_all.by_regime[int(regime)]
    sss_all = switching_policy_sss_by_regime_from_policy(params, net, policy=policy)
    return sss_all.by_regime[int(regime)]

def _implied_i_at_sss(
    params: ModelParams,
    policy: PolicyName,
    net: PolicyNetwork,
    sss: Dict[str, float],
    *,
    regime: int,
    rbar_by_regime: torch.Tensor | None,
) -> float:
    # build a single-state tensor consistent with model_common.State order
    dev, dt = params.device, params.dtype
    if policy == "commitment":
        x = torch.tensor(
            [sss["Delta_prev"], sss["logA"], sss["loggtilde"], sss["xi"], float(regime), sss["vartheta_prev"], sss["varrho_prev"]],
            device=dev, dtype=dt
        ).view(1, -1)
    else:
        x = torch.tensor([sss["Delta_prev"], sss["logA"], sss["loggtilde"], sss["xi"], float(regime)], device=dev, dtype=dt).view(1, -1)

    out = {
        "c": torch.tensor([sss["c"]], device=dev, dtype=dt),
        "pi": torch.tensor([sss["pi"]], device=dev, dtype=dt),
        "pstar": torch.tensor([sss["pstar"]], device=dev, dtype=dt),
        "lam": torch.tensor([sss["lam"]], device=dev, dtype=dt),
        "w": torch.tensor([sss["w"]], device=dev, dtype=dt),
        "XiN": torch.tensor([sss["XiN"]], device=dev, dtype=dt),
        "XiD": torch.tensor([sss["XiD"]], device=dev, dtype=dt),
        "Delta": torch.tensor([sss["Delta"]], device=dev, dtype=dt),
    }
    # Commitment also needs contemporaneous multipliers for the implied-i Euler inversion
    if policy == "commitment":
        out["vartheta"] = torch.tensor([sss["vartheta"]], device=dev, dtype=dt)
        out["varrho"] = torch.tensor([sss["varrho"]], device=dev, dtype=dt)

    # minimal trainer (no training)
    if params.device == "cpu":
        cfg_sim = TrainConfig.dev(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    else:
        cfg_sim = TrainConfig.full(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, gh_n=7, rbar_by_regime=rbar_by_regime)
    i_t = implied_nominal_rate_from_euler(params, policy, x, out, 7, trainer)
    return float(i_t.item())


def build_table2(
    artifacts_root: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    use_selected: bool = True,
    include_rules: bool = True,
) -> pd.DataFrame:
    """
    Build a Table-2-like summary after trainings.

    Policies included:
      - flex prices
      - commitment
      - discretion
      - (optional) taylor, mod_taylor

    Uses:
      - SSS computed AFTER training as fixed points of the trained policies.
      - Ergodic moments from saved sim_paths.npz in each run directory.
    """
    params = ModelParams(device=device, dtype=dtype)
    # flex SSS
    flex = solve_flexprice_sss(params)

    # efficient c_hat: per paper definition, use normal-times efficient allocation
    eff_ss = solve_efficient_sss(params)
    c_hat = float(eff_ss["c_hat"])

        # rbar_by_regime (needed for mod_taylor): use flex-price natural rates if nothing saved
    rbar_by_regime = torch.tensor(
        [flex.by_regime[0]["r_star"], flex.by_regime[1]["r_star"]],
        device=params.device,
        dtype=params.dtype,
    )
    # If a previously-saved file exists, allow it to override (back-compat)
    rbar_path = os.path.join(artifacts_root, "flex", "sss.json")
    if os.path.exists(rbar_path):
        try:
            js = load_json(rbar_path)
            if "rbar_by_regime" in js:
                rbar_by_regime = torch.tensor(js["rbar_by_regime"], device=params.device, dtype=params.dtype)
        except Exception:
            pass

    policies: List[Tuple[str, Optional[str]]] = [
        ("flex", None),
        ("commitment", "commitment"),
        ("discretion", "discretion"),
    ]
    if include_rules:
        policies += [("taylor", "taylor"), ("mod_taylor", "mod_taylor")]

    rows: List[Dict[str, Any]] = []

    def add_block(label: str, policy_key: str, regime: int, sss: Dict[str, float], sim: Optional[Dict[str, np.ndarray]]):
        # SSS statistics
        pi_ss = float(sss["pi"])
        # output gap SSS: log(c) - log(c_hat)
        x_ss = float(np.log(float(sss["c"])) - np.log(c_hat))
        # nominal/real in SSS
        if policy_key == "flex":
            i_ss = float(flex.by_regime[regime]["r_star"])  # pi=0 so nominal=real
        else:
            net = nets[policy_key]
            i_ss = _implied_i_at_sss(params, policy_key, net, sss, regime=regime, rbar_by_regime=rbar_by_regime if policy_key=="mod_taylor" else None)
        r_ss = (1.0 + i_ss) / (1.0 + pi_ss) - 1.0

        # moments from sim
        if sim is None:
            pi_m = {"mean": np.nan, "std": np.nan, "skew": np.nan}
            x_m = {"mean": np.nan, "std": np.nan, "skew": np.nan}
            r_m = {"mean": np.nan, "std": np.nan, "skew": np.nan}
            i_m = {"mean": np.nan, "std": np.nan, "skew": np.nan}
        else:
            s = sim["s"]
            # inflation
            pi_series = _split_by_regime(sim["pi"], s, regime)
            pi_m = _moments_with_skew(pi_series)
            # output gap
            x_all = output_gap_from_consumption(sim, c_hat, params=params)
            x_series = _split_by_regime(x_all, s, regime)
            x_m = _moments_with_skew(x_series)
            # nominal
            i_series = _split_by_regime(sim["i"], s, regime)
            i_m = _moments_with_skew(i_series)
            # real (aligned to t with pi_{t+1})
            r_all, s_al = _compute_real_rate_series(sim)
            r_series = _split_by_regime(r_all, s_al, regime)
            r_m = _moments_with_skew(r_series)

        rows.append({
            "policy": label,
            "regime": "normal" if regime == 0 else "bad",
            # Inflation
            "pi_sss_pct": _annualize_pct(pi_ss),
            "pi_mean_pct": _annualize_pct(pi_m["mean"]),
            "pi_std_pct": _annualize_pct(pi_m["std"]),
            "pi_skew": pi_m["skew"],
            # Output gap
            "x_sss_pct": 100.0 * x_ss,
            "x_mean_pct": 100.0 * x_m["mean"],
            "x_std_pct": 100.0 * x_m["std"],
            "x_skew": x_m["skew"],
            # Real rate
            "r_sss_pct": _annualize_pct(r_ss),
            "r_mean_pct": _annualize_pct(r_m["mean"]),
            "r_std_pct": _annualize_pct(r_m["std"]),
            "r_skew": r_m["skew"],
            # Nominal rate
            "i_sss_pct": _annualize_pct(i_ss),
            "i_mean_pct": _annualize_pct(i_m["mean"]),
            "i_std_pct": _annualize_pct(i_m["std"]),
            "i_skew": i_m["skew"],
        })

    nets: Dict[str, PolicyNetwork] = {}
    sims: Dict[str, Dict[str, np.ndarray]] = {}

    # load nets + sims for trained policies
    for label, pkey in policies:
        if pkey is None:
            continue
        run_dir = _load_run_dir(artifacts_root, pkey, use_selected=use_selected)
        nets[pkey] = _load_net_from_run(run_dir, params, pkey)
        sims[pkey] = _load_sim_paths(run_dir)
        # Quick diagnostic: Table-2 skewness can be unstable with short saved sims.
        try:
            n = int(np.asarray(sims[pkey]["s"]).size)
            if n < 5000:
                print(
                    f"[build_table2] WARNING: sim_paths for policy='{pkey}' has only {n} observations. "
                    "Skewness and even means can deviate from the paper. Consider re-saving sim_paths with larger T/thin." 
                )
        except Exception:
            pass

    # flex block (paper reports ergodic moments for flex prices as well)
    flex_sim: Optional[Dict[str, np.ndarray]] = None
    flex_sim_path = os.path.join(artifacts_root, "flex", "sim_paths.npz")
    if os.path.exists(flex_sim_path):
        try:
            flex_sim = _load_sim_paths(os.path.join(artifacts_root, "flex"))
        except Exception:
            flex_sim = None
    if flex_sim is None:
        # Fallback: simulate flex prices directly (no network) using the model's switching + temporary shocks
        flex_sim = _simulate_flex_prices_for_table2(params)
    for s in [0, 1]:
        add_block("flex", "flex", s, flex.by_regime[s], flex_sim)

    # ---- Diagnostic: frozen-regime SSS (P = I), printed only ----
    # This is NOT used in Table 2 calculations; it is a comparison object.
    for label, pkey in policies:
        if pkey is None or pkey == "flex":
            continue
        try:
            frz = frozen_policy_sss_by_regime_from_policy(params, nets[pkey], policy=pkey)
            for reg in (0, 1):
                sss_r = frz.by_regime[reg]
                print(
                    f"[frozen regime SSS] policy={pkey} regime={reg} "
                    f"pi={sss_r['pi']:.6g} c={sss_r['c']:.6g} "
                    f"Delta_prev={sss_r.get('Delta_prev', float('nan')):.6g}"
                )
        except Exception as e:
            print(f"[frozen regime SSS] policy={pkey} failed: {e}")

    # others
    for label, pkey in policies:
        if pkey is None:
            continue
        for s in [0, 1]:
            sss = _policy_sss_for_policy(params, pkey, nets[pkey], s)
            # add required fields for commitment (vartheta_prev,varrho_prev) if missing
            if pkey == "commitment":
                # solve_commitment_sss_from_policy returns them already
                pass
            sim = sims.get(pkey)
            add_block(label, pkey, s, sss, sim)

    df = pd.DataFrame(rows)
    # Order like paper: by policy then regime
    pol_order = ["flex", "commitment", "discretion", "taylor", "mod_taylor"]
    df["policy"] = pd.Categorical(df["policy"], categories=pol_order, ordered=True)
    df["regime"] = pd.Categorical(df["regime"], categories=["normal", "bad"], ordered=True)
    df = df.sort_values(["policy", "regime"]).reset_index(drop=True)
    return df


def save_table2_csv(df: pd.DataFrame, artifacts_root: str, *, filename: str = "table2_reproduced.csv") -> str:
    out_path = os.path.join(artifacts_root, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
