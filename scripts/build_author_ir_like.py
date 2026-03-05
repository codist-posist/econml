#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams, PolicyName, TrainConfig
from src.deqn import Trainer, implied_nominal_rate_from_euler
from src.io_utils import load_json, load_torch
from src.metrics import flex_c_series_from_A_g_tau
from src.model_common import identities, shock_laws_of_motion, unpack_state, transition_probs_to_next_regimes, regime_sigma_tau
from src.steady_states import export_rbar_tensor, solve_flexprice_sss
from src.table2_builder import _load_net_from_run, _load_run_dir
from scripts.build_author_postprocess_like import _i_flex_author_like, _i_flex_paper_like


# Author impulse_response.py uses quantile=0.84134 and norm.ppf(quantile).
_AUTHOR_IRS_SIGMA = 0.9998150936147446

_SHOCK_GROUPS = [
    ["3sigA_NT", "1sigA_NT", "-1sigA_NT", "-3sigA_NT"],
    ["3sigT_NT", "1sigT_NT", "-1sigT_NT", "-3sigT_NT"],
    ["3sigG_NT", "1sigG_NT", "-1sigG_NT", "-3sigG_NT"],
    ["3sigA_SS", "1sigA_SS", "-1sigA_SS", "-3sigA_SS"],
    ["3sigT_SS", "1sigT_SS", "-1sigT_SS", "-3sigT_SS"],
    ["3sigG_SS", "1sigG_SS", "-1sigG_SS", "-3sigG_SS"],
    ["NT", "SS"],
]

_PARA_GRID_AUTHOR = np.array(
    [
        0.01666667,
        1.0 / 50.0,
        1.0 / 40.0,
        1.0 / 30.0,
        1.0 / 20.0,
        1.0 / 10.0,
        0.115,
        0.21333334,
        0.3116667,
        0.41000003,
        0.5083333,
        0.6066667,
        0.705,
        0.8033333,
        0.90166664,
        1.0,
    ],
    dtype=np.float64,
)


def _parse_dtype(s: str) -> torch.dtype:
    ss = str(s).strip().lower()
    if ss in ("float64", "torch.float64"):
        return torch.float64
    if ss in ("float32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s!r}")


def _cons_flex_author(params: ModelParams, A: np.ndarray) -> np.ndarray:
    expo = (1.0 + float(params.omega)) / (float(params.omega) + float(params.gamma))
    return np.clip(np.asarray(A, dtype=np.float64), 1e-12, None) ** expo


def _cons_flex_paper(params: ModelParams, A: np.ndarray, g: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return flex_c_series_from_A_g_tau(params, A=A, g=g, tau=tau)


def _resolve_cons_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in {"author", "paper"}:
        raise ValueError(f"Unsupported cons_mode={mode!r}; expected 'author' or 'paper'")
    return m


def _flatten_shock_labels() -> list[str]:
    out: list[str] = []
    for row in _SHOCK_GROUPS:
        out.extend(row)
    return out


def _parse_bool(s: str) -> bool:
    x = str(s).strip().lower()
    if x in {"1", "true", "yes", "on"}:
        return True
    if x in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean string, got: {s!r}")


def _resolve_compute_i_flex(mode: str, *, para_mode: bool) -> bool:
    m = str(mode).strip().lower()
    if m == "on":
        return True
    if m == "off":
        return False
    if m != "auto":
        raise ValueError(f"Unsupported --compute-i-flex mode: {mode!r}")
    return not para_mode


def _load_params_from_run(
    run_dir: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> ModelParams:
    p = ModelParams(device=device, dtype=dtype)
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return p.to_torch()
    try:
        blob = load_json(cfg_path)
        p_blob = blob.get("params", {})
        fields = set(ModelParams.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in p_blob.items() if k in fields}
        kwargs["device"] = device
        kwargs["dtype"] = dtype
        return ModelParams(**kwargs).to_torch()
    except Exception:
        return p.to_torch()


def _compose_next_state(
    trainer: Trainer,
    x: torch.Tensor,
    out: Dict[str, torch.Tensor],
    logA_n: torch.Tensor,
    logg_n: torch.Tensor,
    xi_n: torch.Tensor,
    s_n: torch.Tensor,
) -> torch.Tensor:
    policy = trainer.policy
    dt = x.dtype
    st = unpack_state(x, policy)
    if policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "discretion", "discretion_zlb"):
        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
    if policy == "taylor_para":
        if bool(getattr(trainer, "_taylor_para_has_extended_state", False)):
            p21_prev = st.p21 if st.p21 is not None else torch.full_like(out["Delta"], float(trainer.params.p21))
            return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["i_nom"], p21_prev], dim=-1)
        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
    if policy == "commitment":
        if st.c_prev is not None or bool(getattr(trainer, "_commitment_has_c_prev", False)):
            return torch.stack(
                [out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"], out["c"]],
                dim=-1,
            )
        return torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"]], dim=-1)
    if policy == "commitment_zlb":
        return torch.stack(
            [
                out["Delta"],
                logA_n,
                logg_n,
                xi_n,
                s_n.to(dt),
                out["vartheta"],
                out["varrho"],
                out["c"],
                out["i_nom"],
                out["varphi"],
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported policy: {policy!r}")


def _author_ir_shock_vectors(params: ModelParams, *, n_batches: int = 26) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(n_batches) != 26:
        raise ValueError(f"Author IR setup requires n_batches=26, got {n_batches}.")
    z = float(_AUTHOR_IRS_SIGMA)
    # Keep eps in standard-normal units. sigma scaling is applied once in
    # shock_laws_of_motion(...): log_next = ... + sigma * eps.
    # This matches author Dynamics.ir_shock semantics (3*sigma*IRS_shock, ...).
    sA = z
    sT = z
    sG = z
    epsA = np.array(
        [
            3.0 * sA,
            sA,
            -sA,
            -3.0 * sA,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 * sA,
            sA,
            -sA,
            -3.0 * sA,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    epsT = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 * sT,
            sT,
            -sT,
            -3.0 * sT,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 * sT,
            sT,
            -sT,
            -3.0 * sT,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    epsG = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 * sG,
            sG,
            -sG,
            -3.0 * sG,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 * sG,
            sG,
            -sG,
            -3.0 * sG,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return epsA, epsG, epsT


def _author_regimes(
    *,
    n_batches: int = 26,
) -> Tuple[np.ndarray, np.ndarray]:
    if int(n_batches) != 26:
        raise ValueError(f"Author IR setup requires n_batches=26, got {n_batches}.")
    half = (int(n_batches) - 2) // 2
    init = np.concatenate(
        [
            np.zeros(half, dtype=np.int64),
            np.ones(half, dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        ]
    )
    shock = np.concatenate(
        [
            np.zeros(half, dtype=np.int64),
            np.ones(half, dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
        ]
    )
    return init, shock


def _load_starting_state(trainer: Trainer, run_dir: str) -> torch.Tensor:
    p = os.path.join(run_dir, "starting_state.pt")
    if os.path.exists(p):
        try:
            x = load_torch(p)
            if isinstance(x, torch.Tensor) and x.ndim == 2 and int(x.shape[0]) >= 1:
                x = x.to(device=trainer.params.device, dtype=trainer.params.dtype)
                return torch.mean(x, dim=0, keepdim=True)
        except Exception:
            pass
    x = trainer.simulate_initial_state(64, commitment_sss=None)
    return torch.mean(x, dim=0, keepdim=True)


def _policy_supports_para_grid(policy: str) -> bool:
    return str(policy) == "taylor_para"


def _run_author_ir_episode(
    trainer: Trainer,
    *,
    x0_single: torch.Tensor,
    no_steps: int,
    no_steps_decay: int,
    presteps: int,
    use_para_grid: bool,
    para_grid: np.ndarray | None,
) -> torch.Tensor:
    params = trainer.params
    policy = trainer.policy
    dev, dt = params.device, params.dtype

    n_batches = 26
    epsA_26, epsg_26, epst_26 = _author_ir_shock_vectors(params, n_batches=n_batches)
    reg_init_26, reg_shock_26 = _author_regimes(n_batches=n_batches)

    x = x0_single.repeat(n_batches, 1)
    x[:, 4] = torch.as_tensor(reg_init_26, device=dev, dtype=dt)
    epsA = torch.as_tensor(epsA_26, device=dev, dtype=dt)
    epsg = torch.as_tensor(epsg_26, device=dev, dtype=dt)
    epst = torch.as_tensor(epst_26, device=dev, dtype=dt)
    s_shock = torch.as_tensor(reg_shock_26, device=dev, dtype=torch.long)

    if use_para_grid:
        if para_grid is None:
            raise ValueError("para_grid is required when use_para_grid=True.")
        if not _policy_supports_para_grid(policy):
            raise ValueError(f"Para grid is only supported for policy='taylor_para', got {policy!r}.")
        g = torch.as_tensor(np.asarray(para_grid, dtype=np.float64), device=dev, dtype=dt)
        n_grid = int(g.shape[0])
        x = x.repeat_interleave(n_grid, dim=0)
        epsA = epsA.repeat_interleave(n_grid, dim=0)
        epsg = epsg.repeat_interleave(n_grid, dim=0)
        epst = epst.repeat_interleave(n_grid, dim=0)
        s_shock = s_shock.repeat_interleave(n_grid, dim=0)
        if int(x.shape[1]) >= 7:
            x[:, 6] = g.repeat(n_batches)
    else:
        n_grid = 1

    T_total = int(no_steps) + int(no_steps_decay)
    if int(presteps) >= int(no_steps):
        raise ValueError(f"presteps ({presteps}) must be < no_steps ({no_steps}).")
    B = int(x.shape[0])
    D = int(x.shape[1])
    states = torch.empty((T_total, B, D), dtype=dt, device="cpu")
    states[0] = x.detach().to(device="cpu")

    zeros = torch.zeros(B, device=dev, dtype=dt)
    for t in range(1, T_total):
        out = trainer._policy_outputs(x)
        st = unpack_state(x, policy)
        if t == int(no_steps):
            logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, epsA, epsg, epst, s_shock)
        else:
            logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, zeros, zeros, zeros, st.s)
        x = _compose_next_state(trainer, x, out, logA_n, logg_n, xi_n, s_n)
        states[t] = x.detach().to(device="cpu")

    start = int(no_steps) - int(presteps)
    return states[start:, :, :]


def _compute_sim_arrays_from_states(
    trainer: Trainer,
    states: torch.Tensor,
    *,
    gh_n: int,
    chunk: int = 32768,
) -> Dict[str, np.ndarray]:
    params = trainer.params
    policy = trainer.policy
    dev, dt = params.device, params.dtype

    T, B, D = states.shape
    flat = states.reshape(T * B, D)

    vals: Dict[str, list[np.ndarray]] = {
        "c": [],
        "pi": [],
        "pstar": [],
        "Delta": [],
        "y": [],
        "h": [],
        "g": [],
        "A": [],
        "tau": [],
        "s": [],
        "i": [],
        "xi": [],
        "p21": [],
        "p12_eff": [],
        "p21_eff": [],
        "sigma_tau": [],
    }
    explicit_i = ("taylor", "taylor_para", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "commitment_zlb")

    for i0 in range(0, int(flat.shape[0]), int(chunk)):
        i1 = min(int(flat.shape[0]), i0 + int(chunk))
        x = flat[i0:i1].to(device=dev, dtype=dt)
        out = trainer._policy_outputs(x)
        st = unpack_state(x, policy)
        ids = identities(params, st, out)

        if policy in explicit_i:
            i_now = out["i_nom"]
        else:
            i_now = implied_nominal_rate_from_euler(params, policy, x, out, int(gh_n), trainer)

        vals["c"].append(out["c"].detach().cpu().numpy())
        vals["pi"].append(out["pi"].detach().cpu().numpy())
        vals["pstar"].append(out["pstar"].detach().cpu().numpy())
        vals["Delta"].append(out["Delta"].detach().cpu().numpy())
        vals["y"].append(ids["y"].detach().cpu().numpy())
        vals["h"].append(ids["h"].detach().cpu().numpy())
        vals["g"].append(ids["g"].detach().cpu().numpy())
        vals["A"].append(ids["A"].detach().cpu().numpy())
        vals["tau"].append((ids["one_plus_tau"] - 1.0).detach().cpu().numpy())
        vals["s"].append(st.s.detach().cpu().numpy())
        vals["i"].append(i_now.detach().cpu().numpy())
        vals["xi"].append(st.xi.detach().cpu().numpy())
        p21_state = st.p21 if st.p21 is not None else None
        probs_s0 = transition_probs_to_next_regimes(
            params,
            torch.zeros_like(st.s),
            xi=st.xi,
            p21_state=p21_state,
        )
        probs_s1 = transition_probs_to_next_regimes(
            params,
            torch.ones_like(st.s),
            xi=st.xi,
            p21_state=p21_state,
        )
        vals["p12_eff"].append(probs_s0[:, 1].detach().cpu().numpy())
        vals["p21_eff"].append(probs_s1[:, 0].detach().cpu().numpy())
        vals["sigma_tau"].append(regime_sigma_tau(params, st.s).detach().cpu().numpy())
        if st.p21 is None:
            vals["p21"].append(np.full((i1 - i0,), float(params.p21), dtype=np.float64))
        else:
            vals["p21"].append(st.p21.detach().cpu().numpy())

    out_np: Dict[str, np.ndarray] = {}
    for k, parts in vals.items():
        arr = np.concatenate(parts, axis=0).reshape(T, B)
        if k == "s":
            out_np[k] = arr.astype(np.int64)
        else:
            out_np[k] = arr.astype(np.float64)
    return out_np


def _to_author_defs_shaped(
    sim: Dict[str, np.ndarray],
    params: ModelParams,
    *,
    compute_i_flex: bool,
    cons_mode: str = "paper",
) -> Dict[str, np.ndarray]:
    mode = _resolve_cons_mode(cons_mode)
    c = np.asarray(sim["c"], dtype=np.float64)
    shp = c.shape
    flat = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in sim.items() if k != "s"}
    s = np.asarray(sim["s"], dtype=np.int64).reshape(-1)

    one_plus_pi = np.clip(1.0 + flat["pi"], 1e-12, None)
    eps = float(params.eps)
    pi_aux = one_plus_pi ** eps
    p_star_aux = np.clip(flat["pstar"], 1e-12, None) ** (-eps)
    r_real = flat["i"] - flat["pi"]

    if mode == "author":
        cons_flex = _cons_flex_author(params, flat["A"])
    else:
        cons_flex = _cons_flex_paper(params, flat["A"], flat["g"], flat["tau"])
    out_gap = np.log(np.clip(flat["c"], 1e-12, None) / np.clip(cons_flex, 1e-12, None))

    if compute_i_flex:
        if mode == "author":
            i_flex = _i_flex_author_like(
                params,
                cons_flex=cons_flex,
                A=flat["A"],
                n_quad_pts=3,
            )
        else:
            i_flex = _i_flex_paper_like(
                params,
                cons_flex=cons_flex,
                A=flat["A"],
                g=flat["g"],
                s=s,
                xi=flat["xi"],
                p21_state=flat.get("p21", None),
                n_quad_pts=3,
            )
    else:
        i_flex = np.full_like(cons_flex, np.nan, dtype=np.float64)

    out = {
        "cons_y": flat["c"].reshape(shp),
        "disp_y": flat["Delta"].reshape(shp),
        "i_nom_y": flat["i"].reshape(shp),
        "h_work_y": flat["h"].reshape(shp),
        "pi_tot_y": flat["pi"].reshape(shp),
        "pi_aux_y": pi_aux.reshape(shp),
        "p_star_y": flat["pstar"].reshape(shp),
        "p_star_aux_y": p_star_aux.reshape(shp),
        "r_real_y": r_real.reshape(shp),
        "tau_x": flat["tau"].reshape(shp),
        "a_x": flat["A"].reshape(shp),
        "xi_x": flat["xi"].reshape(shp),
        "g_x": flat["g"].reshape(shp),
        "y_tot_y": flat["y"].reshape(shp),
        "out_gap_y": out_gap.reshape(shp),
        "i_flex_y": i_flex.reshape(shp),
        "cons_flex_y": cons_flex.reshape(shp),
        "regime_x": s.reshape(shp),
    }
    if "p21" in sim:
        out["p_21_x"] = np.asarray(sim["p21"], dtype=np.float64).reshape(shp)
    if "p12_eff" in sim:
        out["p_12_eff_x"] = np.asarray(sim["p12_eff"], dtype=np.float64).reshape(shp)
    if "p21_eff" in sim:
        out["p_21_eff_x"] = np.asarray(sim["p21_eff"], dtype=np.float64).reshape(shp)
    if "sigma_tau" in sim:
        out["sigma_tau_x"] = np.asarray(sim["sigma_tau"], dtype=np.float64).reshape(shp)
    return out


def _author_state_names_for_policy(policy: str, state_dim: int) -> list[tuple[str, int]]:
    if policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb", "discretion", "discretion_zlb"):
        return [
            ("disp_old_x", 0),
            ("log_a_x", 1),
            ("log_xi_x", 3),
            ("log_g_x", 2),
            ("regime_x", 4),
        ]
    if policy == "taylor_para":
        out = [
            ("disp_old_x", 0),
            ("log_a_x", 1),
            ("log_xi_x", 3),
            ("log_g_x", 2),
            ("regime_x", 4),
        ]
        if state_dim >= 6:
            out.append(("i_old_x", 5))
        if state_dim >= 7:
            out.append(("p_21_x", 6))
        return out
    if policy == "commitment":
        out = [
            ("disp_old_x", 0),
            ("log_a_x", 1),
            ("log_xi_x", 3),
            ("log_g_x", 2),
            ("regime_x", 4),
            ("vartheta_old_x", 5),
            ("rho_old_x", 6),
        ]
        if state_dim >= 8:
            out.append(("c_old_x", 7))
        return out
    if policy == "commitment_zlb":
        return [
            ("disp_old_x", 0),
            ("log_a_x", 1),
            ("log_xi_x", 3),
            ("log_g_x", 2),
            ("regime_x", 4),
            ("vartheta_old_x", 5),
            ("rho_old_x", 6),
            ("c_old_x", 7),
            ("i_nom_old_x", 8),
            ("varphi_old_x", 9),
        ]
    raise ValueError(f"Unsupported policy for state export: {policy!r}")


def _export_ir_npz(
    *,
    run_dir: str,
    policy: PolicyName,
    states: torch.Tensor,
    sim: Dict[str, np.ndarray],
    params: ModelParams,
    out_dir: str,
    use_para_grid: bool,
    n_grid: int,
    compute_i_flex: bool,
    cons_mode: str,
    save_states: bool,
) -> None:
    labels = _flatten_shock_labels()
    T, B, D = states.shape
    states_np = states.cpu().numpy().astype(np.float64, copy=False)
    defs_npz: Dict[str, Dict[str, np.ndarray]] = {}
    states_npz: Dict[str, Dict[str, np.ndarray]] = {}

    if not use_para_grid:
        if int(B) != len(labels):
            raise RuntimeError(f"Expected {len(labels)} IR scenarios, got B={B}.")
        for b, label in enumerate(labels):
            sim_b = {k: v[:, b : b + 1] for k, v in sim.items()}
            defs = _to_author_defs_shaped(sim_b, params, compute_i_flex=compute_i_flex, cons_mode=cons_mode)
            defs_npz[label] = {k: np.asarray(v).reshape(-1) for k, v in defs.items()}
            if save_states:
                names = _author_state_names_for_policy(str(policy), int(D))
                states_npz[label] = {nm: states_np[:, b, idx].reshape(-1) for nm, idx in names}
    else:
        if int(B) != len(labels) * int(n_grid):
            raise RuntimeError(f"Expected B={len(labels)}*{n_grid} in para mode, got B={B}.")
        for i, label in enumerate(labels):
            sl = slice(i * int(n_grid), (i + 1) * int(n_grid))
            sim_blk = {k: v[:, sl] for k, v in sim.items()}
            defs_npz[label] = _to_author_defs_shaped(sim_blk, params, compute_i_flex=compute_i_flex, cons_mode=cons_mode)
            if save_states:
                names = _author_state_names_for_policy(str(policy), int(D))
                states_npz[label] = {nm: states_np[:, sl, idx] for nm, idx in names}

    np.savez_compressed(os.path.join(out_dir, "IR_definitions.npz"), **defs_npz)
    if save_states:
        np.savez_compressed(os.path.join(out_dir, "IR_states.npz"), **states_npz)


def _infer_para_mode(policy: str, x0: torch.Tensor, use_para_grid: bool) -> bool:
    if not bool(use_para_grid):
        return False
    return _policy_supports_para_grid(policy) and int(x0.shape[-1]) >= 7


def run_export(
    *,
    artifacts_root: str,
    policy: PolicyName,
    run_dir: str | None,
    use_selected: bool,
    device: str,
    dtype: torch.dtype,
    out_subdir: str,
    no_steps: int,
    no_steps_decay: int,
    presteps: int,
    gh_n: int,
    compute_i_flex_mode: str,
    use_para_grid: bool,
    save_states: bool,
    clean_out_dir: bool,
    params_override: Dict[str, float] | None = None,
    cons_mode: str = "paper",
) -> str:
    mode = _resolve_cons_mode(cons_mode)
    rd = run_dir
    if rd is None:
        rd = _load_run_dir(artifacts_root, policy, use_selected=use_selected, required_files=())
    params = _load_params_from_run(rd, device=device, dtype=dtype)
    if params_override:
        allowed = set(ModelParams.__dataclass_fields__.keys()) - {"device", "dtype"}
        p_new = {k: getattr(params, k) for k in ModelParams.__dataclass_fields__.keys()}
        for k, v in params_override.items():
            if k not in allowed:
                raise ValueError(
                    f"Unsupported override key={k!r}. "
                    f"Allowed keys: {sorted(list(allowed))}"
                )
            p_new[k] = float(v)
        p_new["device"] = params.device
        p_new["dtype"] = params.dtype
        params = ModelParams(**p_new).to_torch()
    net = _load_net_from_run(rd, params, policy)

    rbar_by_regime = None
    if policy in ("mod_taylor", "mod_taylor_zlb"):
        flex = solve_flexprice_sss(params)
        rbar_by_regime = export_rbar_tensor(params, flex)

    cfg_sim = TrainConfig.author_like(policy=policy)  # type: ignore[arg-type]
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, gh_n=int(gh_n), rbar_by_regime=rbar_by_regime)
    x0 = _load_starting_state(trainer, rd)
    try:
        _ = trainer._policy_outputs(x0)
    except Exception as e:
        raise RuntimeError(
            f"Run '{rd}' for policy='{policy}' is incompatible with current decoder/state layout: {e}. "
            "Use a compatible run (or retrain this policy), then re-run this script."
        ) from e

    para_mode = _infer_para_mode(str(policy), x0, use_para_grid)
    para_grid = _PARA_GRID_AUTHOR.copy() if para_mode else None
    compute_i_flex = _resolve_compute_i_flex(compute_i_flex_mode, para_mode=para_mode)

    states = _run_author_ir_episode(
        trainer,
        x0_single=x0,
        no_steps=int(no_steps),
        no_steps_decay=int(no_steps_decay),
        presteps=int(presteps),
        use_para_grid=para_mode,
        para_grid=para_grid,
    )
    sim = _compute_sim_arrays_from_states(trainer, states, gh_n=int(gh_n))

    out_dir = os.path.join(rd, str(out_subdir))
    if clean_out_dir and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _export_ir_npz(
        run_dir=rd,
        policy=policy,
        states=states,
        sim=sim,
        params=params,
        out_dir=out_dir,
        use_para_grid=para_mode,
        n_grid=1 if para_grid is None else int(para_grid.shape[0]),
        compute_i_flex=compute_i_flex,
        cons_mode=mode,
        save_states=bool(save_states),
    )
    with open(os.path.join(out_dir, "author_ir_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"cons_mode": mode}, f, ensure_ascii=True, indent=2, sort_keys=True)
    return out_dir


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build author-style IR files (IR_definitions.npz/IR_states.npz) from a trained run."
    )
    ap.add_argument("--artifacts-root", default=os.path.join("artifacts", "runs"))
    ap.add_argument(
        "--policy",
        required=True,
        choices=[
            "taylor",
            "taylor_para",
            "mod_taylor",
            "discretion",
            "commitment",
            "taylor_zlb",
            "mod_taylor_zlb",
            "discretion_zlb",
            "commitment_zlb",
        ],
    )
    ap.add_argument("--run-dir", default=None, help="Direct run directory. If omitted, auto-resolve from artifacts.")
    ap.add_argument(
        "--use-selected",
        type=str,
        default="true",
        help="Use artifacts/selected_runs.json when resolving run_dir (true/false).",
    )
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:N")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--out-subdir", default="IRS", help="Output subdir inside run dir.")
    ap.add_argument("--no-steps", type=int, default=1000, help="Burn-in steps before pulse (author default: 1000).")
    ap.add_argument("--no-steps-decay", type=int, default=400, help="Steps after pulse (author default: 400).")
    ap.add_argument("--presteps", type=int, default=5, help="Pre-shock points kept in output (author default: 5).")
    ap.add_argument("--gh-n", type=int, default=3, help="GH points for implied i_t in Euler-based policies.")
    ap.add_argument(
        "--compute-i-flex",
        default="auto",
        choices=["auto", "on", "off"],
        help="Compute author i_flex_y exactly. auto=on for non-para, off for para.",
    )
    ap.add_argument(
        "--use-para-grid",
        type=str,
        default="true",
        help="For policy=taylor_para with extended state, use author p21 grid (true/false).",
    )
    ap.add_argument(
        "--save-states",
        type=str,
        default="true",
        help="Also save IR_states.npz (true/false).",
    )
    ap.add_argument(
        "--clean-out-dir",
        type=str,
        default="true",
        help="Delete output folder before writing files (true/false).",
    )
    ap.add_argument(
        "--cons-mode",
        default="paper",
        choices=["author", "paper"],
        help="Output-gap consumption mode for saved definitions (default: paper/effective).",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Parameter override in form key=value (repeatable). Example: --override rho_tau=0.99",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    dtype = _parse_dtype(args.dtype)
    overrides: Dict[str, float] = {}
    for item in args.override:
        s = str(item).strip()
        if "=" not in s:
            raise ValueError(f"Invalid --override value {item!r}; expected key=value.")
        k, v = s.split("=", 1)
        overrides[k.strip()] = float(v)
    out_dir = run_export(
        artifacts_root=args.artifacts_root,
        policy=args.policy,  # type: ignore[arg-type]
        run_dir=args.run_dir,
        use_selected=_parse_bool(args.use_selected),
        device=args.device,
        dtype=dtype,
        out_subdir=args.out_subdir,
        no_steps=int(args.no_steps),
        no_steps_decay=int(args.no_steps_decay),
        presteps=int(args.presteps),
        gh_n=int(args.gh_n),
        compute_i_flex_mode=args.compute_i_flex,
        use_para_grid=_parse_bool(args.use_para_grid),
        save_states=_parse_bool(args.save_states),
        clean_out_dir=_parse_bool(args.clean_out_dir),
        params_override=overrides if overrides else None,
        cons_mode=str(args.cons_mode),
    )
    print(f"[build_author_ir_like] policy={args.policy} ({args.cons_mode}) written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
