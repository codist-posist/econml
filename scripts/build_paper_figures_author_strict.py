#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams, PolicyName
from src.io_utils import _normalize_artifacts_root, list_run_dirs, load_json, load_selected_run
from src.steady_states import solve_efficient_sss, solve_flexprice_sss, solve_taylor_sss
from src.table2_builder import _load_net_from_run
from scripts.build_author_postprocess_like import _export_for_policy as _build_author_postprocess
from scripts.build_author_ir_like import run_export as _build_author_ir


def _parse_bool(s: str) -> bool:
    x = str(s).strip().lower()
    if x in {"1", "true", "yes", "on"}:
        return True
    if x in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean string, got: {s!r}")


def _parse_dtype(s: str) -> torch.dtype:
    ss = str(s).strip().lower()
    if ss in ("float64", "torch.float64"):
        return torch.float64
    if ss in ("float32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s!r}")


def _resolve_cons_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in {"author", "paper"}:
        raise ValueError(f"Unsupported cons_mode={mode!r}; expected 'author' or 'paper'")
    return m


def _ann(x: np.ndarray) -> np.ndarray:
    return 400.0 * np.asarray(x, dtype=np.float64)


def _regime_name(reg: int) -> str:
    r = int(reg)
    if r == 0:
        return "normal"
    if r == 1:
        return "bad"
    if r == 2:
        return "severe"
    return f"regime_{r}"


def _hist_by_regime(
    ax,
    values: np.ndarray,
    regimes: np.ndarray,
    *,
    bins: int,
    alpha: float,
    colors: list[str],
    label_prefix: str | None = None,
) -> None:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    regs = np.asarray(regimes, dtype=np.int64).reshape(-1)
    reg_ids = sorted({int(v) for v in np.unique(regs)})
    for j, r in enumerate(reg_ids):
        m = regs == int(r)
        if not np.any(m):
            continue
        kw = {}
        if label_prefix is not None:
            kw["label"] = f"{label_prefix} {_regime_name(r)}"
        ax.hist(vals[m], bins=int(bins), alpha=float(alpha), color=colors[j % len(colors)], **kw)


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


def _resolve_run_dir(
    artifacts_root: str,
    policy: PolicyName,
    *,
    use_selected: bool,
    device: str,
    dtype: torch.dtype,
) -> Tuple[str, ModelParams]:
    ar = _normalize_artifacts_root(artifacts_root)
    cands: list[str] = []
    if use_selected:
        sel = load_selected_run(ar, str(policy))
        if sel is not None:
            cands.append(sel)
    for rd in list_run_dirs(ar, str(policy)):
        if rd not in cands:
            cands.append(rd)
    if not cands:
        raise FileNotFoundError(f"No runs found for policy={policy!r} under {ar}")
    errs: list[str] = []
    for rd in cands:
        try:
            params = _load_params_from_run(rd, device=device, dtype=dtype)
            _ = _load_net_from_run(rd, params, policy)
            return rd, params
        except Exception as e:
            errs.append(f"{rd}: {e}")
    raise RuntimeError(
        f"No compatible run found for policy={policy!r}. Tried:\n  " + "\n  ".join(errs[:6])
    )


def _read_mode_from_meta(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = obj.get("cons_mode", None)
        if v is None:
            return None
        return str(v).strip().lower()
    except Exception:
        return None


def _ensure_author_postprocess(
    *,
    artifacts_root: str,
    policy: PolicyName,
    run_dir: str,
    device: str,
    dtype: torch.dtype,
    use_selected: bool,
    enabled: bool,
    force_rebuild: bool = False,
    cons_mode: str = "paper",
) -> str:
    mode = _resolve_cons_mode(cons_mode)
    out_dir = os.path.join(run_dir, "author_postprocess")
    meta_path = os.path.join(out_dir, "author_postprocess_meta.json")
    required = [
        os.path.join(out_dir, "simulated_definitions.npz"),
        os.path.join(out_dir, "simulated_definitions_NT.npz"),
        os.path.join(out_dir, "simulated_definitions_SS.npz"),
        os.path.join(out_dir, "simulated_definitions_SEV.npz"),
        os.path.join(out_dir, "simulated_definitions_ss_only.npz"),
        os.path.join(out_dir, "simulated_definitions_xi_only.npz"),
    ]
    existing_mode = _read_mode_from_meta(meta_path)
    mode_ok = existing_mode == mode
    if (not force_rebuild) and all(os.path.exists(p) for p in required) and mode_ok:
        return out_dir
    if not enabled:
        missing = [p for p in required if not os.path.exists(p)]
        if (not missing) and (not mode_ok):
            raise FileNotFoundError(
                f"Postprocess files exist but mode mismatch: requested {mode!r}, found {existing_mode!r}"
            )
        raise FileNotFoundError(
            "Missing author postprocess file(s): " + ", ".join(missing)
        )
    _build_author_postprocess(
        artifacts_root,
        str(policy),
        run_dir=run_dir,
        device=device,
        dtype=dtype,
        use_selected=bool(use_selected),
        T=10_000,
        cons_mode=mode,
    )
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Postprocess generation failed for {policy!r}: missing {', '.join(missing)}"
        )
    rebuilt_mode = _read_mode_from_meta(meta_path)
    if rebuilt_mode != mode:
        raise FileNotFoundError(
            f"Postprocess mode mismatch after generation: requested {mode!r}, found {rebuilt_mode!r}"
        )
    return out_dir


def _ensure_author_ir(
    *,
    artifacts_root: str,
    policy: PolicyName,
    run_dir: str,
    device: str,
    dtype: torch.dtype,
    use_selected: bool,
    enabled: bool,
    out_subdir: str = "IRS",
    params_override: Dict[str, float] | None = None,
    force_rebuild: bool = False,
    cons_mode: str = "paper",
) -> str:
    mode = _resolve_cons_mode(cons_mode)
    out_dir = os.path.join(run_dir, out_subdir)
    target = os.path.join(out_dir, "IR_definitions.npz")
    meta_path = os.path.join(out_dir, "author_ir_meta.json")
    existing_mode = _read_mode_from_meta(meta_path)
    mode_ok = existing_mode == mode
    local_force_rebuild = bool(force_rebuild or (params_override is not None))
    if (not local_force_rebuild) and os.path.exists(target) and mode_ok:
        return out_dir
    if not enabled:
        if os.path.exists(target) and (not mode_ok):
            raise FileNotFoundError(
                f"IR files exist but mode mismatch: requested {mode!r}, found {existing_mode!r}"
            )
        raise FileNotFoundError(f"Missing author IR file: {target}")
    _build_author_ir(
        artifacts_root=artifacts_root,
        policy=policy,
        run_dir=run_dir,
        use_selected=bool(use_selected),
        device=device,
        dtype=dtype,
        out_subdir=out_subdir,
        no_steps=1000,
        no_steps_decay=400,
        presteps=5,
        gh_n=3,
        compute_i_flex_mode="on",
        use_para_grid=False,
        save_states=True,
        clean_out_dir=local_force_rebuild,
        params_override=params_override,
        cons_mode=mode,
    )
    if not os.path.exists(target):
        raise FileNotFoundError(f"IR generation failed for {policy!r}: {target}")
    rebuilt_mode = _read_mode_from_meta(meta_path)
    if rebuilt_mode != mode:
        raise FileNotFoundError(
            f"IR mode mismatch after generation: requested {mode!r}, found {rebuilt_mode!r}"
        )
    return out_dir


def _load_defs_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    out = {k: np.asarray(z[k]) for k in z.files}
    z.close()
    return out


def _load_ir_label(path: str, label: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    if label not in z.files:
        raise KeyError(f"IR label {label!r} not found in {path}. Available: {list(z.files)}")
    raw = z[label]
    z.close()
    if isinstance(raw, dict):
        d = raw
    elif isinstance(raw, np.ndarray) and raw.dtype == object:
        d = raw.item()
    else:
        d = raw.item()
    return {k: np.asarray(v) for k, v in d.items()}


def _vec(d: Dict[str, np.ndarray], k: str) -> np.ndarray:
    x = np.asarray(d[k], dtype=np.float64)
    if x.ndim == 2 and x.shape[1] >= 1:
        x = x[:, 0]
    return x.reshape(-1)


def _ergodic_series(defs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    s = np.asarray(defs["regime_x"]).reshape(-1).astype(np.int64)
    return {
        "s": s,
        "s_r": s,
        "pi": np.asarray(defs["pi_tot_y"]).reshape(-1),
        "x": np.asarray(defs["out_gap_y"]).reshape(-1),
        "i": np.asarray(defs["i_nom_y"]).reshape(-1),
        "r": np.asarray(defs["r_real_y"]).reshape(-1),
        "Delta": np.asarray(defs["disp_y"]).reshape(-1),
    }


def _transition_pack(ir: Dict[str, np.ndarray], *, pre: int, n_post: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = int(pre) + int(n_post)
    pi = _vec(ir, "pi_tot_y")[:h]
    x = _vec(ir, "out_gap_y")[:h]
    r = _vec(ir, "r_real_y")[:h]
    D = _vec(ir, "disp_y")[:h]
    return pi, x, r, D


def _irf_pack(ir: Dict[str, np.ndarray], *, pre: int, ir_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = min(int(pre) + int(ir_h), _vec(ir, "pi_tot_y").shape[0])
    pi = _vec(ir, "pi_tot_y")[:h]
    x = _vec(ir, "out_gap_y")[:h]
    r = _vec(ir, "r_real_y")[:h]
    P = np.cumprod(np.clip(1.0 + pi, 1e-12, None))
    idx0 = max(0, int(pre) - 1)
    base = P[idx0] if P.size > 0 else 1.0
    if base != 0:
        P = P / base
    return pi, x, r, P


def _savefig(fig_dir: str, name: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    p = os.path.join(fig_dir, f"{name}.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    print(f"[saved] {p}")


def _clone_params(p: ModelParams, **overrides: float) -> ModelParams:
    kwargs = {k: getattr(p, k) for k in ModelParams.__dataclass_fields__.keys()}
    kwargs.update(overrides)
    return ModelParams(**kwargs).to_torch()


def build_figures(
    *,
    artifacts_root: str,
    fig_dir: str,
    device: str,
    dtype: torch.dtype,
    use_selected: bool,
    ensure_postprocess: bool,
    ensure_ir: bool,
    force_rebuild_postprocess: bool,
    force_rebuild_ir: bool,
    cons_mode: str,
    pre: int,
    n_post: int,
    ir_h: int,
    shock_size_mult: float,
) -> None:
    mode = _resolve_cons_mode(cons_mode)
    run_t, p_t = _resolve_run_dir(artifacts_root, "taylor", use_selected=use_selected, device=device, dtype=dtype)
    run_m, p_m = _resolve_run_dir(artifacts_root, "mod_taylor", use_selected=use_selected, device=device, dtype=dtype)
    run_d, p_d = _resolve_run_dir(artifacts_root, "discretion", use_selected=use_selected, device=device, dtype=dtype)
    run_c, p_c = _resolve_run_dir(artifacts_root, "commitment", use_selected=use_selected, device=device, dtype=dtype)
    run_tz, p_tz = _resolve_run_dir(artifacts_root, "taylor_zlb", use_selected=use_selected, device=device, dtype=dtype)
    run_mz, p_mz = _resolve_run_dir(artifacts_root, "mod_taylor_zlb", use_selected=use_selected, device=device, dtype=dtype)
    run_dz, p_dz = _resolve_run_dir(artifacts_root, "discretion_zlb", use_selected=use_selected, device=device, dtype=dtype)
    run_cz, p_cz = _resolve_run_dir(artifacts_root, "commitment_zlb", use_selected=use_selected, device=device, dtype=dtype)

    pp_t = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="taylor", run_dir=run_t, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_m = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="mod_taylor", run_dir=run_m, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_d = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="discretion", run_dir=run_d, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_c = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="commitment", run_dir=run_c, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_tz = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="taylor_zlb", run_dir=run_tz, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_mz = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="mod_taylor_zlb", run_dir=run_mz, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_dz = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="discretion_zlb", run_dir=run_dz, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )
    pp_cz = _ensure_author_postprocess(
        artifacts_root=artifacts_root, policy="commitment_zlb", run_dir=run_cz, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_postprocess, force_rebuild=force_rebuild_postprocess, cons_mode=mode
    )

    ir_t = _ensure_author_ir(
        artifacts_root=artifacts_root, policy="taylor", run_dir=run_t, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_ir, out_subdir="IRS", force_rebuild=force_rebuild_ir, cons_mode=mode
    )
    ir_m = _ensure_author_ir(
        artifacts_root=artifacts_root, policy="mod_taylor", run_dir=run_m, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_ir, out_subdir="IRS", force_rebuild=force_rebuild_ir, cons_mode=mode
    )
    ir_d = _ensure_author_ir(
        artifacts_root=artifacts_root, policy="discretion", run_dir=run_d, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_ir, out_subdir="IRS", force_rebuild=force_rebuild_ir, cons_mode=mode
    )
    ir_c = _ensure_author_ir(
        artifacts_root=artifacts_root, policy="commitment", run_dir=run_c, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_ir, out_subdir="IRS", force_rebuild=force_rebuild_ir, cons_mode=mode
    )
    ir_cz = _ensure_author_ir(
        artifacts_root=artifacts_root, policy="commitment_zlb", run_dir=run_cz, device=device, dtype=dtype,
        use_selected=use_selected, enabled=ensure_ir, out_subdir="IRS", force_rebuild=force_rebuild_ir, cons_mode=mode
    )
    ir_c_fig9_base = _ensure_author_ir(
        artifacts_root=artifacts_root,
        policy="commitment",
        run_dir=run_c,
        device=device,
        dtype=dtype,
        use_selected=use_selected,
        enabled=ensure_ir,
        out_subdir="IRS_fig9_base",
        params_override={"eta_bar": 0.0, "p12": 0.0, "p21": 1.0},
        force_rebuild=force_rebuild_ir,
        cons_mode=mode,
    )
    ir_c_fig9_hi = _ensure_author_ir(
        artifacts_root=artifacts_root,
        policy="commitment",
        run_dir=run_c,
        device=device,
        dtype=dtype,
        use_selected=use_selected,
        enabled=ensure_ir,
        out_subdir="IRS_fig9_hi",
        params_override={"eta_bar": 0.0, "p12": 0.0, "p21": 1.0, "rho_tau": 0.99},
        force_rebuild=force_rebuild_ir,
        cons_mode=mode,
    )
    ir_c_fig13_big = _ensure_author_ir(
        artifacts_root=artifacts_root,
        policy="commitment",
        run_dir=run_c,
        device=device,
        dtype=dtype,
        use_selected=use_selected,
        enabled=ensure_ir,
        out_subdir="IRS_fig13_big",
        params_override={"eta_bar": float(p_c.eta_bar) * float(shock_size_mult)},
        force_rebuild=force_rebuild_ir,
        cons_mode=mode,
    )

    s_t = _ergodic_series(_load_defs_npz(os.path.join(pp_t, "simulated_definitions.npz")))
    s_m = _ergodic_series(_load_defs_npz(os.path.join(pp_m, "simulated_definitions.npz")))
    s_d = _ergodic_series(_load_defs_npz(os.path.join(pp_d, "simulated_definitions.npz")))
    s_c = _ergodic_series(_load_defs_npz(os.path.join(pp_c, "simulated_definitions.npz")))
    s_tz = _ergodic_series(_load_defs_npz(os.path.join(pp_tz, "simulated_definitions.npz")))
    s_mz = _ergodic_series(_load_defs_npz(os.path.join(pp_mz, "simulated_definitions.npz")))
    s_dz = _ergodic_series(_load_defs_npz(os.path.join(pp_dz, "simulated_definitions.npz")))
    s_cz = _ergodic_series(_load_defs_npz(os.path.join(pp_cz, "simulated_definitions.npz")))

    ir_t_npz = os.path.join(ir_t, "IR_definitions.npz")
    ir_m_npz = os.path.join(ir_m, "IR_definitions.npz")
    ir_d_npz = os.path.join(ir_d, "IR_definitions.npz")
    ir_c_npz = os.path.join(ir_c, "IR_definitions.npz")
    ir_cz_npz = os.path.join(ir_cz, "IR_definitions.npz")
    tr_t = _load_ir_label(ir_t_npz, "NT")
    tr_m = _load_ir_label(ir_m_npz, "NT")
    tr_d = _load_ir_label(ir_d_npz, "NT")
    tr_c = _load_ir_label(ir_c_npz, "NT")
    tr_cz = _load_ir_label(ir_cz_npz, "NT")
    ir7_c = _load_ir_label(ir_c_npz, "1sigT_SS")
    ir7_m = _load_ir_label(ir_m_npz, "1sigT_SS")
    ir9_b = _load_ir_label(os.path.join(ir_c_fig9_base, "IR_definitions.npz"), "1sigT_NT")
    ir9_h = _load_ir_label(os.path.join(ir_c_fig9_hi, "IR_definitions.npz"), "1sigT_NT")
    ir13_b = _load_ir_label(ir_c_npz, "NT")
    ir13_g = _load_ir_label(os.path.join(ir_c_fig13_big, "IR_definitions.npz"), "NT")

    cols_t = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    cols_m = ["tab:green", "tab:red", "tab:olive", "tab:pink", "tab:cyan"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    _hist_by_regime(ax[0, 0], _ann(s_t["pi"]), s_t["s"], bins=60, alpha=0.45, colors=cols_t, label_prefix="Taylor")
    _hist_by_regime(ax[0, 0], _ann(s_m["pi"]), s_m["s"], bins=60, alpha=0.45, colors=cols_m, label_prefix="Mod Taylor")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].set_xlabel("Ann. perc.")
    ax[0, 0].legend(fontsize=8)
    _hist_by_regime(ax[0, 1], 100.0 * s_t["x"], s_t["s"], bins=60, alpha=0.45, colors=cols_t)
    _hist_by_regime(ax[0, 1], 100.0 * s_m["x"], s_m["s"], bins=60, alpha=0.45, colors=cols_m)
    ax[0, 1].set_title("(b) Output gap")
    ax[0, 1].set_xlabel("Perc. of log")
    _hist_by_regime(ax[1, 0], _ann(s_t["i"]), s_t["s"], bins=60, alpha=0.45, colors=cols_t)
    _hist_by_regime(ax[1, 0], _ann(s_m["i"]), s_m["s"], bins=60, alpha=0.45, colors=cols_m)
    ax[1, 0].set_title("(c) Nominal interest rate")
    ax[1, 0].set_xlabel("Ann. perc.")
    _hist_by_regime(ax[1, 1], _ann(s_t["r"]), s_t["s_r"], bins=60, alpha=0.45, colors=cols_t)
    _hist_by_regime(ax[1, 1], _ann(s_m["r"]), s_m["s_r"], bins=60, alpha=0.45, colors=cols_m)
    ax[1, 1].set_title("(d) Real interest rate")
    ax[1, 1].set_xlabel("Ann. perc.")
    fig.suptitle("Figure 2: Ergodic distribution", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure2")
    plt.close(fig)

    pi_ta, x_ta, r_ta, D_ta = _transition_pack(tr_t, pre=pre, n_post=n_post)
    pi_ma, x_ma, r_ma, D_ma = _transition_pack(tr_m, pre=pre, n_post=n_post)
    t = np.arange(-pre, n_post)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t, _ann(pi_ta), label="Taylor")
    ax[0, 0].plot(t, _ann(pi_ma), "--", label="Modified Taylor")
    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 1].plot(t, 100.0 * x_ta, label="Taylor")
    ax[0, 1].plot(t, 100.0 * x_ma, "--", label="Modified Taylor")
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[0, 1].set_title("(b) Output gap")
    ax[1, 0].plot(t, _ann(r_ta), label="Taylor")
    ax[1, 0].plot(t, _ann(r_ma), "--", label="Modified Taylor")
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 0].set_title("(c) Real interest rate")
    ax[1, 1].plot(t, D_ta, label="Taylor")
    ax[1, 1].plot(t, D_ma, "--", label="Modified Taylor")
    ax[1, 1].set_title("(d) Price dispersion")
    for a in ax.ravel():
        a.set_xlabel("Time in quarters")
    ax[0, 0].legend()
    fig.suptitle("Figure 3: Response to a regime change (Taylor rule)", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure3")
    plt.close(fig)

    params0 = p_t
    grid_p21 = np.linspace(0.02, 0.98, 40)
    dur_bad = 1.0 / grid_p21
    bad_reg = int(params0.bad_state) if int(params0.n_regimes) > 1 else 0
    bad_lbl = _regime_name(bad_reg)
    pi_normal, r_normal, pi_bad_cf, r_bad_cf = [], [], [], []
    for p21 in grid_p21:
        pb = _clone_params(params0, p12=float(params0.p12), p21=float(p21))
        flex_b = solve_flexprice_sss(pb)
        tay_b = solve_taylor_sss(pb, flex_b)
        n_key = 0 if 0 in tay_b.by_regime else sorted(tay_b.by_regime.keys())[0]
        pi_normal.append(float(tay_b.by_regime[n_key]["pi"]))
        r_normal.append(float(tay_b.by_regime[n_key]["r"]))
        pc = _clone_params(params0, p12=1.0, p21=float(p21))
        flex_c = solve_flexprice_sss(pc)
        tay_c = solve_taylor_sss(pc, flex_c)
        b_key = bad_reg if bad_reg in tay_c.by_regime else sorted(tay_c.by_regime.keys())[-1]
        pi_bad_cf.append(float(tay_c.by_regime[b_key]["pi"]))
        r_bad_cf.append(float(tay_c.by_regime[b_key]["r"]))
    x = dur_bad
    idx = np.argsort(x)
    eff = solve_efficient_sss(params0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x[idx], _ann(np.array(pi_normal)[idx]), label="Baseline (normal SSS)")
    ax[0].plot(x[idx], _ann(np.array(pi_bad_cf)[idx]), "--", label=f"Counterfactual p12->1 ({bad_lbl} SSS)")
    ax[0].axhline(0.0, color="gray", ls=":")
    ax[0].set_title("(a) Inflation")
    ax[0].set_xlabel("Average bad-times duration (quarters)")
    ax[0].set_ylabel("Ann. perc.")
    ax[0].legend()
    ax[1].plot(x[idx], _ann(np.array(r_normal)[idx]), label="Baseline (normal SSS)")
    ax[1].plot(x[idx], _ann(np.array(r_bad_cf)[idx]), "--", label=f"Counterfactual p12->1 ({bad_lbl} SSS)")
    ax[1].axhline(_ann(np.array([float(eff["r_hat"])]))[0], color="gray", ls=":")
    ax[1].set_title("(b) Real interest rate")
    ax[1].set_xlabel("Average bad-times duration (quarters)")
    ax[1].set_ylabel("Ann. perc.")
    ax[1].legend()
    fig.suptitle("Figure 4: Sensitivity to regime length (Taylor rule)", y=1.03)
    plt.tight_layout()
    _savefig(fig_dir, "figure4")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cols_d = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    _hist_by_regime(ax[0, 0], _ann(s_d["pi"]), s_d["s"], bins=60, alpha=0.6, colors=cols_d, label_prefix="discretion")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].legend()
    _hist_by_regime(ax[0, 1], 100.0 * s_d["x"], s_d["s"], bins=60, alpha=0.6, colors=cols_d)
    ax[0, 1].set_title("(b) Output gap")
    _hist_by_regime(ax[1, 0], _ann(s_d["r"]), s_d["s_r"], bins=60, alpha=0.6, colors=cols_d)
    ax[1, 0].set_title("(c) Real interest rate")
    _hist_by_regime(ax[1, 1], s_d["Delta"], s_d["s"], bins=60, alpha=0.6, colors=cols_d)
    ax[1, 1].set_title("(d) Price dispersion")
    for a in ax.ravel():
        a.set_xlabel("model units")
    fig.suptitle("Figure 5: Ergodic distribution: discretion", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure5")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cols_c = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    _hist_by_regime(ax[0, 0], _ann(s_c["pi"]), s_c["s"], bins=60, alpha=0.6, colors=cols_c, label_prefix="commitment")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].legend()
    _hist_by_regime(ax[0, 1], 100.0 * s_c["x"], s_c["s"], bins=60, alpha=0.6, colors=cols_c)
    ax[0, 1].set_title("(b) Output gap")
    _hist_by_regime(ax[1, 0], _ann(s_c["r"]), s_c["s_r"], bins=60, alpha=0.6, colors=cols_c)
    ax[1, 0].set_title("(c) Real interest rate")
    _hist_by_regime(ax[1, 1], s_c["Delta"], s_c["s"], bins=60, alpha=0.6, colors=cols_c)
    ax[1, 1].set_title("(d) Price dispersion")
    for a in ax.ravel():
        a.set_xlabel("model units")
    fig.suptitle("Figure 6: Ergodic distribution: commitment", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure6")
    plt.close(fig)

    pi_c7, x_c7, r_c7, P_c7 = _irf_pack(ir7_c, pre=pre, ir_h=ir_h)
    pi_m7, x_m7, r_m7, P_m7 = _irf_pack(ir7_m, pre=pre, ir_h=ir_h)
    t7 = np.arange(-pre, -pre + len(pi_c7))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t7, _ann(pi_c7), label="Commitment")
    ax[0, 0].plot(t7, _ann(pi_m7), "--", label="Modified Taylor")
    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 1].plot(t7, 100.0 * x_c7, label="Commitment")
    ax[0, 1].plot(t7, 100.0 * x_m7, "--", label="Modified Taylor")
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[0, 1].set_title("(b) Output gap")
    ax[1, 0].plot(t7, _ann(r_c7), label="Commitment")
    ax[1, 0].plot(t7, _ann(r_m7), "--", label="Modified Taylor")
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 0].set_title("(c) Real interest rate")
    ax[1, 1].plot(t7, P_c7, label="Commitment")
    ax[1, 1].plot(t7, P_m7, "--", label="Modified Taylor")
    ax[1, 1].set_title("(d) Price level")
    for a in ax.ravel():
        a.set_xlabel("Time in quarters")
    ax[0, 0].legend()
    fig.suptitle("Figure 7: Impulse response to a transitory cost-push shock: commitment versus Taylor rule.", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure7")
    plt.close(fig)

    pi_c8, x_c8, r_c8, D_c8 = _transition_pack(tr_c, pre=pre, n_post=n_post)
    pi_d8, x_d8, r_d8, D_d8 = _transition_pack(tr_d, pre=pre, n_post=n_post)
    t8 = np.arange(-pre, n_post)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t8, _ann(pi_c8), label="Commitment")
    ax[0, 0].plot(t8, _ann(pi_d8), "--", label="Discretion")
    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 1].plot(t8, 100.0 * x_c8, label="Commitment")
    ax[0, 1].plot(t8, 100.0 * x_d8, "--", label="Discretion")
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[0, 1].set_title("(b) Output gap")
    ax[1, 0].plot(t8, _ann(r_c8), label="Commitment")
    ax[1, 0].plot(t8, _ann(r_d8), "--", label="Discretion")
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 0].set_title("(c) Real interest rate")
    ax[1, 1].plot(t8, D_c8, label="Commitment")
    ax[1, 1].plot(t8, D_d8, "--", label="Discretion")
    ax[1, 1].set_title("(d) Price dispersion")
    for a in ax.ravel():
        a.set_xlabel("Time in quarters")
    ax[0, 0].legend()
    fig.suptitle("Figure 8: Response to a regime change: commitment versus discretion.", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure8")
    plt.close(fig)

    pi_b9, x_b9, r_b9, P_b9 = _irf_pack(ir9_b, pre=pre, ir_h=ir_h)
    pi_h9, x_h9, r_h9, P_h9 = _irf_pack(ir9_h, pre=pre, ir_h=ir_h)
    t9 = np.arange(-pre, -pre + len(pi_b9))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t9, _ann(pi_b9), label="baseline rho_tau")
    ax[0, 0].plot(t9, _ann(pi_h9), "--", label="rho_tau=0.99")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 1].plot(t9, 100.0 * x_b9, label="baseline rho_tau")
    ax[0, 1].plot(t9, 100.0 * x_h9, "--", label="rho_tau=0.99")
    ax[0, 1].set_title("(b) Output gap")
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[1, 0].plot(t9, _ann(r_b9), label="baseline rho_tau")
    ax[1, 0].plot(t9, _ann(r_h9), "--", label="rho_tau=0.99")
    ax[1, 0].set_title("(c) Real interest rate")
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 1].plot(t9, P_b9, label="baseline rho_tau")
    ax[1, 1].plot(t9, P_h9, "--", label="rho_tau=0.99")
    ax[1, 1].set_title("(d) Price level")
    for a in ax.ravel():
        a.set_xlabel("Time in quarters")
    ax[0, 0].legend()
    fig.suptitle("Figure 9: Impulse response to a transitory cost-push shock with different levels of persistence.", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure9")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cols_tz = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    cols_mz = ["tab:green", "tab:red", "tab:olive", "tab:pink", "tab:cyan"]
    _hist_by_regime(ax[0, 0], _ann(s_tz["pi"]), s_tz["s"], bins=60, alpha=0.45, colors=cols_tz, label_prefix="Taylor ZLB")
    _hist_by_regime(ax[0, 0], _ann(s_mz["pi"]), s_mz["s"], bins=60, alpha=0.45, colors=cols_mz, label_prefix="Mod Taylor ZLB")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].legend(fontsize=8)
    _hist_by_regime(ax[0, 1], 100.0 * s_tz["x"], s_tz["s"], bins=60, alpha=0.45, colors=cols_tz)
    _hist_by_regime(ax[0, 1], 100.0 * s_mz["x"], s_mz["s"], bins=60, alpha=0.45, colors=cols_mz)
    ax[0, 1].set_title("(b) Output gap")
    _hist_by_regime(ax[1, 0], _ann(s_tz["i"]), s_tz["s"], bins=60, alpha=0.45, colors=cols_tz)
    _hist_by_regime(ax[1, 0], _ann(s_mz["i"]), s_mz["s"], bins=60, alpha=0.45, colors=cols_mz)
    ax[1, 0].set_title("(c) Nominal interest rate")
    _hist_by_regime(ax[1, 1], _ann(s_tz["r"]), s_tz["s_r"], bins=60, alpha=0.45, colors=cols_tz)
    _hist_by_regime(ax[1, 1], _ann(s_mz["r"]), s_mz["s_r"], bins=60, alpha=0.45, colors=cols_mz)
    ax[1, 1].set_title("(d) Real interest rate")
    for a in ax.ravel():
        a.set_xlabel("model units")
    fig.suptitle("Figure 10: Ergodic distribution: Taylor rules with a ZLB", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure10")
    plt.close(fig)

    pi_c11, x_c11, r_c11, D_c11 = _transition_pack(tr_c, pre=pre, n_post=n_post)
    pi_z11, x_z11, r_z11, D_z11 = _transition_pack(tr_cz, pre=pre, n_post=n_post)
    t11 = np.arange(-pre, n_post)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t11, _ann(pi_c11), label="Commitment (no ZLB)")
    ax[0, 0].plot(t11, _ann(pi_z11), "--", label="Commitment ZLB")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 1].plot(t11, 100.0 * x_c11, label="Commitment (no ZLB)")
    ax[0, 1].plot(t11, 100.0 * x_z11, "--", label="Commitment ZLB")
    ax[0, 1].set_title("(b) Output gap")
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[1, 0].plot(t11, _ann(r_c11), label="Commitment (no ZLB)")
    ax[1, 0].plot(t11, _ann(r_z11), "--", label="Commitment ZLB")
    ax[1, 0].set_title("(c) Real interest rate")
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 1].plot(t11, D_c11, label="Commitment (no ZLB)")
    ax[1, 1].plot(t11, D_z11, "--", label="Commitment ZLB")
    ax[1, 1].set_title("(d) Price dispersion")
    for a in ax.ravel():
        a.set_xlabel("Time in quarters")
    ax[0, 0].legend()
    fig.suptitle("Figure 11: Response to a regime change: commitment with ZLB.", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure11")
    plt.close(fig)

    defs_c_full = _load_defs_npz(os.path.join(pp_c, "simulated_definitions.npz"))
    pi_sim = np.asarray(defs_c_full["pi_tot_y"], dtype=np.float64).reshape(-1)
    t_show = min(10_000, pi_sim.shape[0])
    P = np.cumprod(np.clip(1.0 + pi_sim[:t_show], 1e-12, None))
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(np.arange(t_show), P)
    ax.set_title("Figure 12: Price level dynamics under commitment")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Price level index")
    plt.tight_layout()
    _savefig(fig_dir, "figure12")
    plt.close(fig)

    pi_b13, x_b13, r_b13, D_b13 = _transition_pack(ir13_b, pre=pre, n_post=n_post)
    pi_g13, x_g13, r_g13, D_g13 = _transition_pack(ir13_g, pre=pre, n_post=n_post)
    t13 = np.arange(-pre, n_post)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(t13, _ann(pi_b13), label="baseline")
    ax[0, 0].plot(t13, _ann(pi_g13), "--", label="larger shock")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 1].plot(t13, 100.0 * x_b13, label="baseline")
    ax[0, 1].plot(t13, 100.0 * x_g13, "--", label="larger shock")
    ax[0, 1].set_title("(b) Output gap")
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[1, 0].plot(t13, _ann(r_b13), label="baseline")
    ax[1, 0].plot(t13, _ann(r_g13), "--", label="larger shock")
    ax[1, 0].set_title("(c) Real interest rate")
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 1].plot(t13, D_b13, label="baseline")
    ax[1, 1].plot(t13, D_g13, "--", label="larger shock")
    ax[1, 1].set_title("(d) Price dispersion")
    for a in ax.ravel():
        a.set_xlabel("Time in quarters")
    ax[0, 0].legend()
    fig.suptitle("Figure 13: Response to a regime change: shock size", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure13")
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Paper note for Fig. 14 color mapping:
    # commitment -> blue/orange, discretion -> green/red.
    cols_cz = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    cols_dz = ["tab:green", "tab:red", "tab:olive", "tab:pink", "tab:cyan"]
    _hist_by_regime(ax[0, 0], _ann(s_cz["pi"]), s_cz["s"], bins=60, alpha=0.45, colors=cols_cz, label_prefix="Commitment ZLB")
    _hist_by_regime(ax[0, 0], _ann(s_dz["pi"]), s_dz["s"], bins=60, alpha=0.45, colors=cols_dz, label_prefix="Discretion ZLB")
    ax[0, 0].set_title("(a) Inflation")
    ax[0, 0].legend(fontsize=8)
    _hist_by_regime(ax[0, 1], 100.0 * s_cz["x"], s_cz["s"], bins=60, alpha=0.45, colors=cols_cz)
    _hist_by_regime(ax[0, 1], 100.0 * s_dz["x"], s_dz["s"], bins=60, alpha=0.45, colors=cols_dz)
    ax[0, 1].set_title("(b) Output gap")
    _hist_by_regime(ax[1, 0], _ann(s_cz["i"]), s_cz["s"], bins=60, alpha=0.45, colors=cols_cz)
    _hist_by_regime(ax[1, 0], _ann(s_dz["i"]), s_dz["s"], bins=60, alpha=0.45, colors=cols_dz)
    ax[1, 0].set_title("(c) Nominal interest rate")
    _hist_by_regime(ax[1, 1], _ann(s_cz["r"]), s_cz["s_r"], bins=60, alpha=0.45, colors=cols_cz)
    _hist_by_regime(ax[1, 1], _ann(s_dz["r"]), s_dz["s_r"], bins=60, alpha=0.45, colors=cols_dz)
    ax[1, 1].set_title("(d) Real interest rate")
    for a in ax.ravel():
        a.set_xlabel("model units")
    fig.suptitle("Figure 14: Ergodic distribution: optimal policy at the ZLB.", y=1.02)
    plt.tight_layout()
    _savefig(fig_dir, "figure14")
    plt.close(fig)

    note = os.path.join(fig_dir, "figure1_note.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write("Paper Figure 1 is a schematic neural-network diagram and is not generated from simulation code.\n")
    print(f"[saved] {note}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build paper figures (2..14) using strict author-style postprocess/IR data."
    )
    ap.add_argument("--artifacts-root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "artifacts", "paper_figures"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--use-selected", type=str, default="true")
    ap.add_argument("--ensure-postprocess", type=str, default="true")
    ap.add_argument("--ensure-ir", type=str, default="true")
    ap.add_argument("--cons-mode", type=str, default="paper", choices=["author", "paper"])
    ap.add_argument("--force-rebuild-postprocess", type=str, default="false")
    ap.add_argument("--force-rebuild-ir", type=str, default="false")
    ap.add_argument("--pre", type=int, default=5)
    ap.add_argument("--n-post", type=int, default=20)
    ap.add_argument("--ir-h", type=int, default=120)
    ap.add_argument("--shock-size-mult", type=float, default=1.5)
    args = ap.parse_args(list(argv) if argv is not None else None)

    build_figures(
        artifacts_root=args.artifacts_root,
        fig_dir=args.fig_dir,
        device=args.device,
        dtype=_parse_dtype(args.dtype),
        use_selected=_parse_bool(args.use_selected),
        ensure_postprocess=_parse_bool(args.ensure_postprocess),
        ensure_ir=_parse_bool(args.ensure_ir),
        force_rebuild_postprocess=_parse_bool(args.force_rebuild_postprocess),
        force_rebuild_ir=_parse_bool(args.force_rebuild_ir),
        cons_mode=str(args.cons_mode),
        pre=int(args.pre),
        n_post=int(args.n_post),
        ir_h=int(args.ir_h),
        shock_size_mult=float(args.shock_size_mult),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
