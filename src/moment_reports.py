from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from .config import ModelParams
from .io_utils import load_json
from .metrics import output_gap_from_consumption
from .steady_states import solve_efficient_sss


def _regime_name(reg: int) -> str:
    r = int(reg)
    if r == 0:
        return "normal"
    if r == 1:
        return "bad"
    if r == 2:
        return "severe"
    return f"regime_{r}"


def _to_display(df: pd.DataFrame) -> None:
    try:
        from IPython.display import display  # type: ignore

        display(df)
    except Exception:
        print(df.to_string(index=False))


def _load_params_from_run(run_dir: str, *, device: str, dtype: torch.dtype) -> ModelParams:
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


def _cons_flex_author_series(params: ModelParams, A: np.ndarray) -> np.ndarray:
    expo = (1.0 + float(params.omega)) / (float(params.omega) + float(params.gamma))
    return np.clip(np.asarray(A, dtype=np.float64).reshape(-1), 1e-12, None) ** expo


def _author_postprocess_ready(run_dir: str, *, cons_mode: str) -> bool:
    out_dir = os.path.join(run_dir, "author_postprocess")
    req = [
        os.path.join(out_dir, "simulated_definitions.npz"),
        os.path.join(out_dir, "simulated_definitions_NT.npz"),
        os.path.join(out_dir, "simulated_definitions_SS.npz"),
        os.path.join(out_dir, "simulated_definitions_ss_only.npz"),
        os.path.join(out_dir, "simulated_definitions_xi_only.npz"),
    ]
    if not all(os.path.exists(p) for p in req):
        return False
    meta_path = os.path.join(out_dir, "author_postprocess_meta.json")
    if not os.path.exists(meta_path):
        return False
    try:
        meta = load_json(meta_path)
        return str(meta.get("cons_mode", "")).strip().lower() == str(cons_mode).strip().lower()
    except Exception:
        return False


def _author_nt_ss_moments(run_dir: str) -> pd.DataFrame:
    out_dir = os.path.join(run_dir, "author_postprocess")
    rows = []
    for reg, fname in ((0, "simulated_definitions_NT.npz"), (1, "simulated_definitions_SS.npz")):
        p = os.path.join(out_dir, fname)
        with np.load(p) as z:
            pi = np.asarray(z["pi_tot_y"]).reshape(-1)
            x = np.asarray(z["out_gap_y"]).reshape(-1)
            i = np.asarray(z["i_nom_y"]).reshape(-1)
            if "r_real_y" in z.files:
                r = np.asarray(z["r_real_y"]).reshape(-1)
            else:
                r = i - pi
            rows.append(
                {
                    "regime": _regime_name(reg),
                    "n": int(pi.size),
                    "pi_mean_pct": 400.0 * float(np.mean(pi)),
                    "x_mean_pct": 100.0 * float(np.mean(x)),
                    "i_mean_pct": 400.0 * float(np.mean(i)),
                    "r_mean_pct": 400.0 * float(np.mean(r)),
                }
            )
    return pd.DataFrame(rows)


def _sim_paths_conditional_moments(run_dir: str, *, params: ModelParams) -> pd.DataFrame:
    p = os.path.join(run_dir, "sim_paths.npz")
    if not os.path.exists(p):
        return pd.DataFrame(
            columns=[
                "regime",
                "n",
                "pi_mean_pct",
                "x_mean_author_pct",
                "x_mean_paper_pct",
                "i_mean_pct",
                "r_mean_same_pct",
                "r_mean_next_pct",
            ]
        )

    with np.load(p) as z:
        sim = {k: z[k] for k in z.files}

    s = np.asarray(sim["s"]).reshape(-1).astype(np.int64)
    pi = np.asarray(sim["pi"]).reshape(-1)
    c = np.asarray(sim["c"]).reshape(-1)

    if "A" in sim:
        A = np.asarray(sim["A"]).reshape(-1)
    else:
        A = np.exp(np.asarray(sim["logA"]).reshape(-1))

    x_author = np.log(np.clip(c, 1e-12, None)) - np.log(np.clip(_cons_flex_author_series(params, A), 1e-12, None))

    eff_ss = solve_efficient_sss(params)
    x_paper = np.asarray(output_gap_from_consumption(sim, eff_ss, params=params, time_varying=True)).reshape(-1)

    has_i = "i" in sim
    if has_i:
        i = np.asarray(sim["i"]).reshape(-1)
        # Author/table convention for real rate:
        #   r_t = i_t - pi_t
        r_same = i - pi
        # Old ex-post alternative (not used):
        # r_next = (1.0 + i[:-1]) / (1.0 + pi[1:]) - 1.0
        # s_next = s[:-1]
        r_next = None
        s_next = None
    else:
        i = None
        r_same = None
        r_next = None
        s_next = None

    rows = []
    reg_ids = sorted({int(v) for v in np.unique(s)})
    for reg in reg_ids:
        m = s == reg
        row = {
            "regime": _regime_name(reg),
            "n": int(np.sum(m)),
            "pi_mean_pct": 400.0 * float(np.mean(pi[m])) if np.any(m) else float("nan"),
            "x_mean_author_pct": 100.0 * float(np.mean(x_author[m])) if np.any(m) else float("nan"),
            "x_mean_paper_pct": 100.0 * float(np.mean(x_paper[m])) if np.any(m) else float("nan"),
            "i_mean_pct": float("nan"),
            "r_mean_same_pct": float("nan"),
            "r_mean_next_pct": float("nan"),
        }
        if has_i and i is not None:
            row["i_mean_pct"] = 400.0 * float(np.mean(i[m])) if np.any(m) else float("nan")
            row["r_mean_same_pct"] = 400.0 * float(np.mean(r_same[m])) if np.any(m) else float("nan")
            if r_next is not None and s_next is not None:
                m2 = s_next == reg
                row["r_mean_next_pct"] = 400.0 * float(np.mean(r_next[m2])) if np.any(m2) else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def build_and_print_policy_reports(
    *,
    run_dir: str,
    policy: str,
    artifacts_root: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    cons_mode: str = "paper",
    rebuild_author_postprocess: bool = False,
) -> Dict[str, Any]:
    """
    Build/ensure author postprocess files and print two summary blocks:
      1) author NT/SS moments (table-comparison object)
      2) long sim_paths conditional moments (our diagnostic object)
    """
    from scripts.build_author_postprocess_like import _export_for_policy

    mode = str(cons_mode).strip().lower()
    if mode not in {"author", "paper"}:
        raise ValueError(f"cons_mode must be 'author' or 'paper' (got {cons_mode!r})")

    need_build = rebuild_author_postprocess or (not _author_postprocess_ready(run_dir, cons_mode=mode))
    if need_build:
        _export_for_policy(
            artifacts_root,
            str(policy),
            run_dir=run_dir,
            device=device,
            dtype=dtype,
            use_selected=False,
            T=10_000,
            cons_mode=mode,
        )

    params = _load_params_from_run(run_dir, device=device, dtype=dtype)
    author_df = _author_nt_ss_moments(run_dir)
    sim_df = _sim_paths_conditional_moments(run_dir, params=params)

    print("\n[author_postprocess NT/SS] (table-comparison object)")
    _to_display(author_df)
    print("\n[sim_paths conditional by regime] (our long-sim diagnostic object)")
    _to_display(sim_df)

    return {
        "run_dir": run_dir,
        "policy": str(policy),
        "cons_mode": mode,
        "author_nt_ss": author_df,
        "sim_paths": sim_df,
    }


def print_sim_paths_moments_report(
    *,
    run_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    """
    Print moments from long saved sim_paths by regime (diagnostic object).
    """
    params = _load_params_from_run(run_dir, device=device, dtype=dtype)
    sim_df = _sim_paths_conditional_moments(run_dir, params=params)
    print("\n[sim_paths conditional by regime] (our long-sim diagnostic object)")
    _to_display(sim_df)
    return sim_df


def build_and_print_author_nt_ss_report(
    *,
    run_dir: str,
    policy: str,
    artifacts_root: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    cons_mode: str = "paper",
    rebuild_author_postprocess: bool = False,
) -> pd.DataFrame:
    """
    Ensure author postprocess files and print NT/SS moments (table object).
    """
    from scripts.build_author_postprocess_like import _export_for_policy

    mode = str(cons_mode).strip().lower()
    if mode not in {"author", "paper"}:
        raise ValueError(f"cons_mode must be 'author' or 'paper' (got {cons_mode!r})")

    need_build = rebuild_author_postprocess or (not _author_postprocess_ready(run_dir, cons_mode=mode))
    if need_build:
        _export_for_policy(
            artifacts_root,
            str(policy),
            run_dir=run_dir,
            device=device,
            dtype=dtype,
            use_selected=False,
            T=10_000,
            cons_mode=mode,
        )

    author_df = _author_nt_ss_moments(run_dir)
    print("\n[author_postprocess NT/SS] (table-comparison object)")
    _to_display(author_df)
    return author_df
