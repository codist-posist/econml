#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams
from src.table2_builder import _load_net_from_run, _load_run_dir
from src.deqn import Trainer, implied_nominal_rate_from_euler
from src.config import TrainConfig
from src.model_common import unpack_state, identities
from src.metrics import flex_c_series_from_A_g_tau
from src.io_utils import load_torch
from src.steady_states import solve_flexprice_sss, export_rbar_tensor


def _resolve_cons_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in {"author", "paper"}:
        raise ValueError(f"Unsupported cons_mode={mode!r}; expected 'author' or 'paper'")
    return m


def _cons_flex_author(
    params: ModelParams,
    A: np.ndarray,
) -> np.ndarray:
    # Author Definitions.cons_flex_y:
    # cons_flex = (A^(1+omega))^(1/(omega+gamma))
    expo = (1.0 + float(params.omega)) / (float(params.omega) + float(params.gamma))
    return np.clip(np.asarray(A, dtype=np.float64), 1e-12, None) ** expo


def _cons_flex_paper(
    params: ModelParams,
    A: np.ndarray,
    g: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    return flex_c_series_from_A_g_tau(params, A=A, g=g, tau=tau)


def _transition_probs_to_next_regime(
    params: ModelParams,
    s: np.ndarray,
    p21_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    s_i = np.asarray(s, dtype=np.int64).reshape(-1)
    p0 = np.where(s_i == 0, 1.0 - float(params.p12), float(params.p21)).astype(np.float64)
    if p21_state is not None:
        p21v = np.clip(np.asarray(p21_state, dtype=np.float64).reshape(-1), 0.0, 1.0)
        bad = (s_i == 1)
        if np.any(bad):
            p0[bad] = p21v[bad]
    p1 = 1.0 - p0
    return p0, p1


def _i_flex_author_like(
    params: ModelParams,
    *,
    cons_flex: np.ndarray,
    A: np.ndarray,
    g: np.ndarray | None = None,
    s: np.ndarray | None = None,
    xi: np.ndarray | None = None,
    p21_state: np.ndarray | None = None,
    n_quad_pts: int = 3,
) -> np.ndarray:
    """
    Author Definitions.i_flex_y with author cons_flex definition:
      cons_flex_t = f(A_t)
      i_flex_t = lambda_t / (beta * E_t[lambda_{t+1}]) - 1
    Since lambda_flex only depends on A, Markov probabilities and non-A shocks drop out.
    """
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    cons_flex = np.asarray(cons_flex, dtype=np.float64).reshape(-1)
    if A.size != cons_flex.size:
        raise ValueError("A and cons_flex must have the same length.")

    A_clip = np.clip(A, 1e-12, None)
    logA = np.log(A_clip)
    beta = float(params.beta)
    rho_A = float(params.rho_A)
    sigma_A = float(params.sigma_A)
    gamma = float(params.gamma)

    driftA = (1.0 - rho_A) * (-(sigma_A**2) / 2.0)

    x, w = hermgauss(int(n_quad_pts))
    nodes = np.sqrt(2.0) * np.asarray(x, dtype=np.float64)
    weights = np.asarray(w, dtype=np.float64) / np.sqrt(np.pi)

    lam_t = np.clip(cons_flex, 1e-12, None) ** (-gamma)
    logA_next = driftA + rho_A * logA[:, None] + sigma_A * nodes[None, :]

    E_lam_next = np.zeros_like(lam_t)
    for ia in range(int(n_quad_pts)):
        A_n = np.exp(logA_next[:, ia])
        wA = weights[ia]
        for ig in range(int(n_quad_pts)):
            wAG = wA * weights[ig]
            for it in range(int(n_quad_pts)):
                w_all = wAG * weights[it]
                c_next = _cons_flex_author(params, A_n)
                lam_next = np.clip(c_next, 1e-12, None) ** (-gamma)
                E_lam_next += w_all * lam_next

    return lam_t / np.clip(beta * E_lam_next, 1e-12, None) - 1.0


def _i_flex_paper_like(
    params: ModelParams,
    *,
    cons_flex: np.ndarray,
    A: np.ndarray,
    g: np.ndarray,
    s: np.ndarray,
    xi: np.ndarray,
    p21_state: np.ndarray | None = None,
    n_quad_pts: int = 3,
) -> np.ndarray:
    """
    Paper-style i_flex:
      i_flex_t = lambda_flex_t / (beta * E_t[lambda_flex_{t+1}]) - 1
    with lambda_flex from paper-style flex consumption c_flex(A,g,tau),
    expectation over shocks and Markov regime transitions.
    """
    A = np.asarray(A, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    s = np.asarray(s, dtype=np.int64).reshape(-1)
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    cons_flex = np.asarray(cons_flex, dtype=np.float64).reshape(-1)

    if A.size != g.size or A.size != s.size or A.size != xi.size or A.size != cons_flex.size:
        raise ValueError("A, g, s, xi, and cons_flex must have the same length.")

    A_clip = np.clip(A, 1e-12, None)
    g_bar = float(params.g_bar)
    if g_bar <= 0.0:
        raise ValueError("g_bar must be strictly positive to recover log(gtilde) from g.")
    logA = np.log(A_clip)
    logg = np.log(np.clip(g / g_bar, 1e-12, None))
    beta = float(params.beta)
    rho_A = float(params.rho_A)
    sigma_A = float(params.sigma_A)
    rho_g = float(params.rho_g)
    sigma_g = float(params.sigma_g)
    rho_tau = float(params.rho_tau)
    sigma_tau = float(params.sigma_tau)
    tau_bar = float(params.tau_bar)
    eta_bar = float(params.eta_bar)
    bad_state = int(params.bad_state)
    gamma = float(params.gamma)

    driftA = (1.0 - rho_A) * (-(sigma_A**2) / 2.0)
    driftg = (1.0 - rho_g) * (-(sigma_g**2) / 2.0)
    drift_xi = (1.0 - rho_tau) * (-(sigma_tau**2) / 2.0)

    # N(0,1) GH nodes/weights
    x, w = hermgauss(int(n_quad_pts))
    nodes = np.sqrt(2.0) * np.asarray(x, dtype=np.float64)
    weights = np.asarray(w, dtype=np.float64) / np.sqrt(np.pi)

    lam_t = np.clip(cons_flex, 1e-12, None) ** (-gamma)
    logA_next = driftA + rho_A * logA[:, None] + sigma_A * nodes[None, :]
    logg_next = driftg + rho_g * logg[:, None] + sigma_g * nodes[None, :]
    xi_next = drift_xi + rho_tau * xi[:, None] + sigma_tau * nodes[None, :]

    p0, p1 = _transition_probs_to_next_regime(params, s, p21_state=p21_state)
    eta0 = eta_bar if bad_state == 0 else 0.0
    eta1 = eta_bar if bad_state == 1 else 0.0

    E_lam_next = np.zeros_like(lam_t)
    for ia in range(int(n_quad_pts)):
        A_n = np.exp(logA_next[:, ia])
        wA = weights[ia]
        for ig in range(int(n_quad_pts)):
            g_n = g_bar * np.exp(logg_next[:, ig])
            wAG = wA * weights[ig]
            for it in range(int(n_quad_pts)):
                w_all = wAG * weights[it]
                xi_n = xi_next[:, it]

                one_plus_tau_0 = (1.0 - tau_bar) + xi_n + eta0
                one_plus_tau_1 = (1.0 - tau_bar) + xi_n + eta1
                c_next_0 = flex_c_series_from_A_g_tau(
                    params,
                    A=A_n,
                    g=g_n,
                    one_plus_tau=one_plus_tau_0,
                    max_iter=80,
                    tol=1e-12,
                )
                c_next_1 = flex_c_series_from_A_g_tau(
                    params,
                    A=A_n,
                    g=g_n,
                    one_plus_tau=one_plus_tau_1,
                    max_iter=80,
                    tol=1e-12,
                )
                lam_next_0 = np.clip(c_next_0, 1e-12, None) ** (-gamma)
                lam_next_1 = np.clip(c_next_1, 1e-12, None) ** (-gamma)
                E_lam_next += w_all * (p0 * lam_next_0 + p1 * lam_next_1)

    return lam_t / np.clip(beta * E_lam_next, 1e-12, None) - 1.0


def _to_author_defs(
    sim: Dict[str, np.ndarray],
    params: ModelParams,
    *,
    cons_mode: str = "author",
) -> Dict[str, np.ndarray]:
    mode = _resolve_cons_mode(cons_mode)
    c = np.asarray(sim["c"], dtype=np.float64).reshape(-1)
    Delta = np.asarray(sim["Delta"], dtype=np.float64).reshape(-1)
    i_nom = np.asarray(sim["i"], dtype=np.float64).reshape(-1)
    h = np.asarray(sim["h"], dtype=np.float64).reshape(-1)
    pi = np.asarray(sim["pi"], dtype=np.float64).reshape(-1)
    pstar = np.asarray(sim["pstar"], dtype=np.float64).reshape(-1)
    tau = np.asarray(sim["tau"], dtype=np.float64).reshape(-1)
    A = np.asarray(sim["A"], dtype=np.float64).reshape(-1)
    g = np.asarray(sim["g"], dtype=np.float64).reshape(-1)
    y = np.asarray(sim["y"], dtype=np.float64).reshape(-1)
    s = np.asarray(sim["s"], dtype=np.float64).reshape(-1)
    xi = np.asarray(sim.get("xi", np.zeros_like(c)), dtype=np.float64).reshape(-1)
    p21_state = np.asarray(sim.get("p21", np.full_like(c, float(params.p21))), dtype=np.float64).reshape(-1)

    one_plus_pi = np.clip(1.0 + pi, 1e-12, None)
    eps = float(params.eps)
    pi_aux = one_plus_pi ** eps
    p_star_aux = np.clip(pstar, 1e-12, None) ** (-eps)
    r_real = i_nom - pi

    if mode == "author":
        cons_flex = _cons_flex_author(params, A)
    else:
        cons_flex = _cons_flex_paper(params, A, g, tau)
    out_gap = np.log(np.clip(c, 1e-12, None) / np.clip(cons_flex, 1e-12, None))

    if mode == "author":
        i_flex = _i_flex_author_like(
            params,
            cons_flex=cons_flex,
            A=A,
            n_quad_pts=3,
        )
    else:
        i_flex = _i_flex_paper_like(
            params,
            cons_flex=cons_flex,
            A=A,
            g=g,
            s=s,
            xi=xi,
            p21_state=p21_state,
            n_quad_pts=3,
        )

    return {
        "cons_y": c,
        "disp_y": Delta,
        "i_nom_y": i_nom,
        "h_work_y": h,
        "pi_tot_y": pi,
        "pi_aux_y": pi_aux,
        "p_star_y": pstar,
        "p_star_aux_y": p_star_aux,
        "r_real_y": r_real,
        "tau_x": tau,
        "a_x": A,
        "xi_x": xi,
        "g_x": g,
        "y_tot_y": y,
        "out_gap_y": out_gap,
        "i_flex_y": i_flex,
        "cons_flex_y": cons_flex,
        "regime_x": s,
    }


def _export_for_policy(
    artifacts_root: str,
    policy: str,
    *,
    run_dir: str | None = None,
    device: str,
    dtype: torch.dtype,
    use_selected: bool,
    T: int,
    cons_mode: str = "author",
) -> str:
    mode = _resolve_cons_mode(cons_mode)
    if run_dir is None:
        run_dir = _load_run_dir(artifacts_root, policy, use_selected=use_selected, required_files=())
    params = ModelParams(device=device, dtype=dtype).to_torch()
    net = _load_net_from_run(run_dir, params, policy)  # type: ignore[arg-type]

    cfg_sim = TrainConfig.author_like(policy=policy)  # type: ignore[arg-type]
    rbar_by_regime = None
    if policy in ("mod_taylor", "mod_taylor_zlb"):
        flex = solve_flexprice_sss(params)
        rbar_by_regime = export_rbar_tensor(params, flex)
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, rbar_by_regime=rbar_by_regime)

    # Author post_process starts from Parameters.starting_state[0:1].
    x0 = None
    start_state_path = os.path.join(run_dir, "starting_state.pt")
    if os.path.exists(start_state_path):
        try:
            x_saved = load_torch(start_state_path)
            if isinstance(x_saved, torch.Tensor) and x_saved.ndim == 2 and int(x_saved.shape[1]) >= 1:
                x0 = x_saved[:1].to(device=params.device, dtype=params.dtype)
        except Exception:
            x0 = None
    if x0 is None:
        x0 = trainer.simulate_initial_state(1, commitment_sss=None)
    try:
        _ = trainer._policy_outputs(x0)
    except Exception as e:
        raise RuntimeError(
            f"Run '{run_dir}' for policy='{policy}' is incompatible with current author decoder: {e}. "
            "Use an author-compatible run (or retrain this policy), then re-run this script."
        ) from e

    names = ["full", "NT", "SS", "ss_only", "xi_only"]
    explicit_i = {"taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "commitment_zlb"}
    keys = ["c", "pi", "pstar", "Delta", "y", "h", "g", "A", "tau", "s", "i", "xi", "p21"]
    sims: Dict[str, Dict[str, np.ndarray]] = {
        nm: {k: np.zeros((int(T), 1), dtype=np.float64) for k in keys} for nm in names
    }

    x = x0.repeat(5, 1)
    for t in range(int(T)):
        out = trainer._policy_outputs(x)
        st = unpack_state(x, policy)
        ids = identities(params, st, out)

        if policy in explicit_i:
            i_now = out["i_nom"]
        else:
            i_now = implied_nominal_rate_from_euler(params, policy, x, out, 3, trainer)

        for b, nm in enumerate(names):
            sims[nm]["c"][t, 0] = float(out["c"][b].detach().cpu())
            sims[nm]["pi"][t, 0] = float(out["pi"][b].detach().cpu())
            sims[nm]["pstar"][t, 0] = float(out["pstar"][b].detach().cpu())
            sims[nm]["Delta"][t, 0] = float(out["Delta"][b].detach().cpu())
            sims[nm]["y"][t, 0] = float(ids["y"][b].detach().cpu())
            sims[nm]["h"][t, 0] = float(ids["h"][b].detach().cpu())
            sims[nm]["g"][t, 0] = float(ids["g"][b].detach().cpu())
            sims[nm]["A"][t, 0] = float(ids["A"][b].detach().cpu())
            sims[nm]["tau"][t, 0] = float((ids["one_plus_tau"][b] - 1.0).detach().cpu())
            sims[nm]["s"][t, 0] = int(st.s[b].detach().cpu())
            sims[nm]["i"][t, 0] = float(i_now[b].detach().cpu())
            sims[nm]["xi"][t, 0] = float(st.xi[b].detach().cpu())
            if st.p21 is not None:
                sims[nm]["p21"][t, 0] = float(st.p21[b].detach().cpu())
            else:
                sims[nm]["p21"][t, 0] = float(params.p21)

        # One shared episode step, then branch transformations as in author post_process.py
        x = trainer._step_state(x)
        # regime_x: [full, 0, 1, keep, 1]
        x[1, 4] = 0.0
        x[2, 4] = 1.0
        x[4, 4] = 1.0
        # log_a_x for ss_only/xi_only -> 0
        x[3, 1] = 0.0
        x[4, 1] = 0.0
        # log_xi_x: ss_only -> 0, xi_only <- previous ss_only xi
        xi_ss_only_prev = x[3, 3].clone()
        x[3, 3] = 0.0
        x[4, 3] = xi_ss_only_prev
        # log_g_x for ss_only/xi_only -> 0
        x[3, 2] = 0.0
        x[4, 2] = 0.0

    out_dir = os.path.join(run_dir, "author_postprocess")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "simulated_definitions.npz"),
        **_to_author_defs(sims["full"], params, cons_mode=mode),
    )
    np.savez_compressed(
        os.path.join(out_dir, "simulated_definitions_NT.npz"),
        **_to_author_defs(sims["NT"], params, cons_mode=mode),
    )
    np.savez_compressed(
        os.path.join(out_dir, "simulated_definitions_SS.npz"),
        **_to_author_defs(sims["SS"], params, cons_mode=mode),
    )
    np.savez_compressed(
        os.path.join(out_dir, "simulated_definitions_ss_only.npz"),
        **_to_author_defs(sims["ss_only"], params, cons_mode=mode),
    )
    np.savez_compressed(
        os.path.join(out_dir, "simulated_definitions_xi_only.npz"),
        **_to_author_defs(sims["xi_only"], params, cons_mode=mode),
    )
    with open(os.path.join(out_dir, "author_postprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"cons_mode": mode}, f, ensure_ascii=True, indent=2, sort_keys=True)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Build author-like post-process definition files (NT/SS/full) for selected runs.")
    ap.add_argument("--artifacts_root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument(
        "--policies",
        nargs="+",
        default=[
            "taylor",
            "mod_taylor",
            "discretion",
            "commitment",
            "taylor_zlb",
            "mod_taylor_zlb",
            "discretion_zlb",
            "commitment_zlb",
            "taylor_para",
        ],
        help="Policies to export.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--T", type=int, default=10000, help="Simulation length (author post_process uses 10000).")
    ap.add_argument(
        "--cons-mode",
        default="author",
        choices=["author", "paper"],
        help="Flex-consumption/output-gap mode for saved definitions.",
    )
    ap.add_argument("--use_selected", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    for pol in args.policies:
        out_dir = _export_for_policy(
            args.artifacts_root,
            pol,
            device=args.device,
            dtype=dtype,
            use_selected=bool(args.use_selected),
            T=int(args.T),
            cons_mode=str(args.cons_mode),
        )
        print(f"[ok] {pol} ({args.cons_mode}): {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
