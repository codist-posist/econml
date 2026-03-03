#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams
from src.table2_builder import _load_net_from_run, _load_run_dir
from src.deqn import Trainer, simulate_paths
from src.config import TrainConfig


def _cons_flex_from_A(params: ModelParams, A: np.ndarray) -> np.ndarray:
    # Author Definitions.cons_flex_y uses A only in this setup.
    expo = (1.0 + float(params.omega)) / (float(params.omega) + float(params.gamma))
    return np.asarray(A, dtype=np.float64) ** expo


def _to_author_defs(sim: Dict[str, np.ndarray], params: ModelParams) -> Dict[str, np.ndarray]:
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

    one_plus_pi = np.clip(1.0 + pi, 1e-12, None)
    eps = float(params.eps)
    pi_aux = one_plus_pi ** eps
    p_star_aux = np.clip(pstar, 1e-12, None) ** (-eps)
    r_real = i_nom - pi

    cons_flex = _cons_flex_from_A(params, A)
    out_gap = np.log(np.clip(c, 1e-12, None) / np.clip(cons_flex, 1e-12, None))

    # Author Definitions.i_flex_y uses conditional expectation. We use a
    # trajectory-consistent approximation from adjacent simulated points.
    lam_flex = np.clip(cons_flex, 1e-12, None) ** (-float(params.gamma))
    i_flex = np.empty_like(c)
    if lam_flex.size > 1:
        i_flex[:-1] = lam_flex[:-1] / (float(params.beta) * np.clip(lam_flex[1:], 1e-12, None)) - 1.0
        i_flex[-1] = i_flex[-2]
    else:
        i_flex[:] = (1.0 / float(params.beta)) - 1.0

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
    device: str,
    dtype: torch.dtype,
    use_selected: bool,
    T: int,
) -> str:
    run_dir = _load_run_dir(artifacts_root, policy, use_selected=use_selected)
    params = ModelParams(device=device, dtype=dtype).to_torch()
    net = _load_net_from_run(run_dir, params, policy)  # type: ignore[arg-type]

    cfg_sim = TrainConfig.author_like(policy=policy)  # type: ignore[arg-type]
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net)

    # Single shared start state (author post_process also starts from a single state replicated across branches).
    x0 = trainer.simulate_initial_state(1, commitment_sss=None)
    # Fast compatibility check: old runs with legacy output dimensions are not decodable
    # under strict author-style transforms used by current code.
    try:
        _ = trainer._policy_outputs(x0)
    except Exception as e:
        raise RuntimeError(
            f"Run '{run_dir}' for policy='{policy}' is incompatible with current author decoder: {e}. "
            "Use an author-compatible run (or retrain this policy), then re-run this script."
        ) from e

    full = simulate_paths(
        params,
        policy,  # type: ignore[arg-type]
        net,
        T=T,
        burn_in=0,
        x0=x0,
        compute_implied_i=True,
        gh_n=3,
        thin=1,
        show_progress=False,
        store_states=True,
    )
    nt = simulate_paths(
        params,
        policy,  # type: ignore[arg-type]
        net,
        T=T,
        burn_in=0,
        x0=x0.clone(),
        compute_implied_i=True,
        gh_n=3,
        thin=1,
        show_progress=False,
        store_states=True,
        force_regime=0,
    )
    ss = simulate_paths(
        params,
        policy,  # type: ignore[arg-type]
        net,
        T=T,
        burn_in=0,
        x0=x0.clone(),
        compute_implied_i=True,
        gh_n=3,
        thin=1,
        show_progress=False,
        store_states=True,
        force_regime=1,
    )
    ss_only = simulate_paths(
        params,
        policy,  # type: ignore[arg-type]
        net,
        T=T,
        burn_in=0,
        x0=x0.clone(),
        compute_implied_i=True,
        gh_n=3,
        thin=1,
        show_progress=False,
        store_states=True,
        force_logA=0.0,
        force_loggtilde=0.0,
    )
    xi_only = simulate_paths(
        params,
        policy,  # type: ignore[arg-type]
        net,
        T=T,
        burn_in=0,
        x0=x0.clone(),
        compute_implied_i=True,
        gh_n=3,
        thin=1,
        show_progress=False,
        store_states=True,
        force_regime=1,
        force_logA=0.0,
        force_loggtilde=0.0,
    )

    out_dir = os.path.join(run_dir, "author_postprocess")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "simulated_definitions.npz"), **_to_author_defs(full, params))
    np.savez_compressed(os.path.join(out_dir, "simulated_definitions_NT.npz"), **_to_author_defs(nt, params))
    np.savez_compressed(os.path.join(out_dir, "simulated_definitions_SS.npz"), **_to_author_defs(ss, params))
    np.savez_compressed(os.path.join(out_dir, "simulated_definitions_ss_only.npz"), **_to_author_defs(ss_only, params))
    np.savez_compressed(os.path.join(out_dir, "simulated_definitions_xi_only.npz"), **_to_author_defs(xi_only, params))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Build author-like post-process definition files (NT/SS/full) for selected runs.")
    ap.add_argument("--artifacts_root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument(
        "--policies",
        nargs="+",
        default=["taylor", "mod_taylor", "discretion", "commitment", "taylor_para"],
        help="Policies to export.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--T", type=int, default=10000, help="Simulation length (author post_process uses 10000).")
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
        )
        print(f"[ok] {pol}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
