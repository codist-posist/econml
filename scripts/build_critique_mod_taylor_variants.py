#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams, TrainConfig
from src.critique_adaptation import (
    CritiqueAugmentedTrainer,
    CritiqueCurriculumConfig,
    RobustModTaylorRuleConfig,
)
from src.deqn import Trainer, _transition_probs_to_next
from src.io_utils import load_json
from src.metrics import output_gap_from_consumption
from src.model_common import identities, shock_laws_of_motion, unpack_state
from src.sanity_checks import _state_from_policy_sss
from src.sss_from_policy import switching_policy_sss_by_regime_from_policy
from src.steady_states import solve_efficient_sss, solve_flexprice_sss
from src.table2_builder import _load_net_from_run
from scripts.build_paper_figures_author_strict import (
    _ann,
    _ensure_author_ir,
    _ensure_author_postprocess,
    _parse_bool,
    _parse_dtype,
    _resolve_cons_mode,
    _resolve_run_dir,
)


def _savefig(fig_dir: str, name: str) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    p = os.path.join(fig_dir, name)
    plt.savefig(p, dpi=160, bbox_inches="tight")
    print(f"[saved] {p}")
    return p


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s)).strip("_") or "variant"


def _regime_label(reg: int) -> str:
    r = int(reg)
    if r == 0:
        return "normal"
    if r == 1:
        return "bad"
    if r == 2:
        return "severe"
    return f"regime_{r}"


def _bad_regime(params: ModelParams) -> int:
    n_reg = max(1, int(params.n_regimes))
    if n_reg <= 1:
        return 0
    br = int(params.bad_state)
    if br < 1 or br >= n_reg:
        return min(1, n_reg - 1)
    return br


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


def _maybe_rbar(params: ModelParams) -> torch.Tensor:
    flex = solve_flexprice_sss(params)
    return torch.tensor(
        [flex.by_regime[s]["r_star"] for s in range(int(params.n_regimes))],
        device=params.device,
        dtype=params.dtype,
    )


@dataclass
class VariantBundle:
    label: str
    run_dir: str
    params: ModelParams
    net: torch.nn.Module
    rbar_by_regime: torch.Tensor
    x0: torch.Tensor
    x0_by_regime: Dict[int, torch.Tensor]
    robust_rule: RobustModTaylorRuleConfig | None = None


def _load_variant_bundle(
    *,
    artifacts_root: str,
    label: str,
    run_dir_override: str | None,
    use_selected: bool,
    device: str,
    dtype: torch.dtype,
    ensure_postprocess: bool,
    ensure_ir: bool,
    force_rebuild_postprocess: bool,
    force_rebuild_ir: bool,
    cons_mode: str,
    robust_rule: RobustModTaylorRuleConfig | None,
) -> VariantBundle:
    if run_dir_override:
        run_dir = os.path.abspath(str(run_dir_override))
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        params = _load_params_from_run(run_dir, device=device, dtype=dtype)
    else:
        run_dir, params = _resolve_run_dir(
            artifacts_root,
            "mod_taylor",
            use_selected=use_selected,
            device=device,
            dtype=dtype,
        )

    if bool(ensure_postprocess) or bool(force_rebuild_postprocess):
        _ensure_author_postprocess(
            artifacts_root=artifacts_root,
            policy="mod_taylor",
            run_dir=run_dir,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            enabled=ensure_postprocess,
            force_rebuild=force_rebuild_postprocess,
            cons_mode=cons_mode,
        )
    if bool(ensure_ir) or bool(force_rebuild_ir):
        _ensure_author_ir(
            artifacts_root=artifacts_root,
            policy="mod_taylor",
            run_dir=run_dir,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            enabled=ensure_ir,
            out_subdir="IRS",
            force_rebuild=force_rebuild_ir,
            cons_mode=cons_mode,
        )

    net = _load_net_from_run(run_dir, params, "mod_taylor")
    rbar = _maybe_rbar(params)
    sss = switching_policy_sss_by_regime_from_policy(
        params,
        net,
        policy="mod_taylor",
        rbar_by_regime=rbar,
    ).by_regime
    x0_by_regime: Dict[int, torch.Tensor] = {}
    for r in range(int(params.n_regimes)):
        base = sss.get(r, sss[0])
        x0_by_regime[r] = _state_from_policy_sss(
            params,
            "mod_taylor",
            base,
            regime=r,
            commitment_state_dim=None,
        )
    x0 = x0_by_regime.get(0, x0_by_regime[min(x0_by_regime.keys())])
    return VariantBundle(
        label=str(label),
        run_dir=str(run_dir),
        params=params,
        net=net,
        rbar_by_regime=rbar,
        x0=x0,
        x0_by_regime=x0_by_regime,
        robust_rule=robust_rule,
    )


def _make_trainer(bundle: VariantBundle, *, gh_n: int = 3):
    params = bundle.params
    if params.device == "cpu":
        cfg_sim = TrainConfig.dev(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    else:
        cfg_sim = TrainConfig.full(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)

    if bundle.robust_rule is not None and bool(bundle.robust_rule.enabled):
        return CritiqueAugmentedTrainer(
            params=params,
            cfg=cfg_sim,
            policy="mod_taylor",
            net=bundle.net,
            gh_n=int(gh_n),
            rbar_by_regime=bundle.rbar_by_regime,
            curriculum=CritiqueCurriculumConfig(enabled=False),
            robust_rule=bundle.robust_rule,
        )

    return Trainer(
        params=params,
        cfg=cfg_sim,
        policy="mod_taylor",
        net=bundle.net,
        gh_n=int(gh_n),
        rbar_by_regime=bundle.rbar_by_regime,
    )


@torch.inference_mode()
def _simulate_custom(
    bundle: VariantBundle,
    *,
    T: int,
    x0: torch.Tensor | None = None,
    regime_path: Sequence[int] | None = None,
    epsA_path: np.ndarray | None = None,
    epsg_path: np.ndarray | None = None,
    epst_path: np.ndarray | None = None,
    noise_scale: float = 0.0,
    noise_mult: float = 1.0,
    seed: int = 123,
    gh_n: int = 3,
) -> Dict[str, np.ndarray]:
    params = bundle.params
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    if epsA_path is None:
        epsA_path = np.zeros(T, dtype=np.float64)
    if epsg_path is None:
        epsg_path = np.zeros(T, dtype=np.float64)
    if epst_path is None:
        epst_path = np.zeros(T, dtype=np.float64)
    epsA_path = np.asarray(epsA_path, dtype=np.float64).reshape(-1)
    epsg_path = np.asarray(epsg_path, dtype=np.float64).reshape(-1)
    epst_path = np.asarray(epst_path, dtype=np.float64).reshape(-1)
    if not (len(epsA_path) == len(epsg_path) == len(epst_path) == int(T)):
        raise ValueError("eps paths must all have length T")

    if regime_path is not None:
        reg = [int(v) for v in regime_path]
        if len(reg) != int(T) + 1:
            raise ValueError("regime_path must have length T+1")
        regime_path = reg

    tr = _make_trainer(bundle, gh_n=int(gh_n))

    x_base = bundle.x0 if x0 is None else x0
    x = x_base.to(device=params.device, dtype=params.dtype)
    B = int(x.shape[0])
    store: Dict[str, np.ndarray] = {
        "c": np.zeros((T + 1, B), dtype=np.float64),
        "pi": np.zeros((T + 1, B), dtype=np.float64),
        "i": np.zeros((T + 1, B), dtype=np.float64),
        "Delta": np.zeros((T + 1, B), dtype=np.float64),
        "y": np.zeros((T + 1, B), dtype=np.float64),
        "h": np.zeros((T + 1, B), dtype=np.float64),
        "g": np.zeros((T + 1, B), dtype=np.float64),
        "A": np.zeros((T + 1, B), dtype=np.float64),
        "tau": np.zeros((T + 1, B), dtype=np.float64),
        "s": np.zeros((T + 1, B), dtype=np.int64),
    }

    for t in range(T + 1):
        out = tr._policy_outputs(x)
        st = unpack_state(x, "mod_taylor")
        ids = identities(params, st, out)

        store["c"][t] = out["c"].detach().cpu().numpy()
        store["pi"][t] = out["pi"].detach().cpu().numpy()
        store["i"][t] = out["i_nom"].detach().cpu().numpy()
        store["Delta"][t] = out["Delta"].detach().cpu().numpy()
        store["y"][t] = ids["y"].detach().cpu().numpy()
        store["h"][t] = ids["h"].detach().cpu().numpy()
        store["g"][t] = ids["g"].detach().cpu().numpy()
        store["A"][t] = ids["A"].detach().cpu().numpy()
        store["tau"][t] = (ids["one_plus_tau"] - 1.0).detach().cpu().numpy()
        store["s"][t] = st.s.detach().cpu().numpy()
        if t == T:
            break

        epsA = torch.full((B,), float(epsA_path[t]), device=params.device, dtype=params.dtype)
        epsg = torch.full((B,), float(epsg_path[t]), device=params.device, dtype=params.dtype)
        epst = torch.full((B,), float(epst_path[t]), device=params.device, dtype=params.dtype)

        if float(noise_scale) != 0.0:
            z = torch.randn((B,), device=params.device, dtype=params.dtype)
            epst = epst + float(noise_scale) * float(noise_mult) * z

        if regime_path is not None:
            s_next = torch.full((B,), int(regime_path[t + 1]), device=params.device, dtype=torch.long)
        else:
            u = torch.rand((B,), device=params.device, dtype=params.dtype)
            probs = _transition_probs_to_next(params, st)
            cdf = torch.cumsum(probs, dim=-1)
            cdf[..., -1] = 1.0
            s_next = torch.sum(u.view(-1, 1) > cdf, dim=-1).to(torch.long)

        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, epsA, epsg, epst, s_next)
        x = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(params.dtype)], dim=-1)

    return store


def _add_output_gap(sim: Dict[str, np.ndarray], params: ModelParams) -> Dict[str, np.ndarray]:
    eff = solve_efficient_sss(params)
    shp = np.asarray(sim["c"]).shape
    x = output_gap_from_consumption(sim, eff, params=params, time_varying=True)
    sim["x"] = np.asarray(x, dtype=np.float64).reshape(shp)
    return sim


def _regime_moments(
    sim: Dict[str, np.ndarray],
    key: str,
    *,
    burn_in: int,
) -> Dict[str, Dict[str, float]]:
    a = np.asarray(sim[key], dtype=np.float64)[burn_in:, :].reshape(-1)
    s = np.asarray(sim["s"], dtype=np.int64)[burn_in:, :].reshape(-1)
    out: Dict[str, Dict[str, float]] = {}
    for reg in sorted({int(v) for v in np.unique(s)}):
        label = _regime_label(reg)
        v = a[s == int(reg)]
        if v.size == 0:
            out[label] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[label] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    return out


def _qband(a: np.ndarray, q: Sequence[float] = (0.1, 0.5, 0.9)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(a, dtype=np.float64)
    return tuple(np.quantile(arr, qq, axis=1) for qq in q)  # type: ignore[return-value]


def _save_compare_vs_baseline(df: pd.DataFrame, *, baseline_label: str, out_csv: str, cols: Sequence[str]) -> None:
    if df.empty or baseline_label not in set(df["variant"].astype(str)):
        df.to_csv(out_csv, index=False)
        print("[saved]", out_csv)
        return
    base = df[df["variant"].astype(str) == str(baseline_label)].iloc[0]
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"d_vs_{baseline_label}__{c}"] = out[c].astype(float) - float(base[c])
    out.to_csv(out_csv, index=False)
    print("[saved]", out_csv)


def _build_shock_train(
    variants: Sequence[VariantBundle],
    *,
    fig_dir: str,
    baseline_label: str,
    T: int,
    shock_times: Sequence[int],
    shock_size: float,
) -> None:
    base_epst = np.zeros(T, dtype=np.float64)
    train_epst = np.zeros(T, dtype=np.float64)
    for k in shock_times:
        if 0 <= int(k) < int(T):
            train_epst[int(k)] = float(shock_size)

    rows: List[Dict[str, float | str]] = []
    rows_all: List[Dict[str, float | str]] = []
    for v in variants:
        normal_reg = 0 if 0 in v.x0_by_regime else min(v.x0_by_regime.keys())
        for start_reg in sorted(v.x0_by_regime.keys()):
            start_label = _regime_label(start_reg)
            regime_path = [int(start_reg)] + [int(_bad_regime(v.params))] * int(T)
            sim_base = _simulate_custom(
                v,
                T=T,
                x0=v.x0_by_regime[start_reg],
                regime_path=regime_path,
                epst_path=base_epst,
                noise_scale=0.0,
                seed=123,
            )
            sim_train = _simulate_custom(
                v,
                T=T,
                x0=v.x0_by_regime[start_reg],
                regime_path=regime_path,
                epst_path=train_epst,
                noise_scale=0.0,
                seed=123,
            )
            sim_base = _add_output_gap(sim_base, v.params)
            sim_train = _add_output_gap(sim_train, v.params)

            t = np.arange(T + 1)
            pi_b = _ann(sim_base["pi"][:, 0])
            pi_t = _ann(sim_train["pi"][:, 0])
            i_b = _ann(sim_base["i"][:, 0])
            i_t = _ann(sim_train["i"][:, 0])
            x_b = 100.0 * sim_base["x"][:, 0]
            x_t = 100.0 * sim_train["x"][:, 0]

            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ax[0, 0].plot(t, pi_b, label="baseline")
            ax[0, 0].plot(t, pi_t, "--", label="shock train")
            ax[0, 0].set_title("(a) Inflation")
            ax[0, 0].set_ylabel("ann. %")
            ax[0, 0].legend()

            ax[0, 1].plot(t, i_b, label="baseline")
            ax[0, 1].plot(t, i_t, "--", label="shock train")
            ax[0, 1].set_title("(b) Nominal interest rate")
            ax[0, 1].set_ylabel("ann. %")
            ax[0, 1].legend()

            ax[1, 0].plot(t, x_b, label="baseline")
            ax[1, 0].plot(t, x_t, "--", label="shock train")
            ax[1, 0].set_title("(c) Output gap")
            ax[1, 0].set_ylabel("%")
            ax[1, 0].set_xlabel("quarter")
            ax[1, 0].legend()

            p_b = np.cumprod(1.0 + sim_base["pi"][:, 0])
            p_t = np.cumprod(1.0 + sim_train["pi"][:, 0])
            ax[1, 1].plot(t, p_b, label="baseline")
            ax[1, 1].plot(t, p_t, "--", label="shock train")
            ax[1, 1].set_title("(d) Price level index")
            ax[1, 1].set_xlabel("quarter")
            ax[1, 1].legend()

            fig.suptitle(f"Critique Shock Train: {v.label} (start={start_label})", y=1.02)
            plt.tight_layout()
            _savefig(fig_dir, f"critique_shock_train_{_slug(v.label)}_start_{start_label}.png")
            plt.close(fig)

            row = {
                "variant": v.label,
                "start_regime": start_label,
                "start_regime_id": int(start_reg),
                "peak_pi_baseline_ann_pct": float(np.max(pi_b[1:])),
                "peak_pi_shock_train_ann_pct": float(np.max(pi_t[1:])),
                "avg_i_baseline_ann_pct": float(np.mean(i_b[1:])),
                "avg_i_shock_train_ann_pct": float(np.mean(i_t[1:])),
                "avg_x_baseline_pct": float(np.mean(x_b[1:])),
                "avg_x_shock_train_pct": float(np.mean(x_t[1:])),
            }
            rows_all.append(row)
            if int(start_reg) == int(normal_reg):
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(fig_dir, "critique_shock_train_summary.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_shock_train_summary.csv"))
    pd.DataFrame(rows_all).to_csv(
        os.path.join(fig_dir, "critique_shock_train_summary_by_start.csv"),
        index=False,
    )
    print("[saved]", os.path.join(fig_dir, "critique_shock_train_summary_by_start.csv"))

    delta = summary.copy()
    delta["d_peak_pi_ann_pct"] = delta["peak_pi_shock_train_ann_pct"] - delta["peak_pi_baseline_ann_pct"]
    delta["d_avg_i_ann_pct"] = delta["avg_i_shock_train_ann_pct"] - delta["avg_i_baseline_ann_pct"]
    delta["d_avg_x_pct"] = delta["avg_x_shock_train_pct"] - delta["avg_x_baseline_pct"]
    delta.to_csv(os.path.join(fig_dir, "critique_shock_train_delta.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_shock_train_delta.csv"))

    delta_all = pd.DataFrame(rows_all)
    if not delta_all.empty:
        delta_all = delta_all.copy()
        delta_all["d_peak_pi_ann_pct"] = delta_all["peak_pi_shock_train_ann_pct"] - delta_all["peak_pi_baseline_ann_pct"]
        delta_all["d_avg_i_ann_pct"] = delta_all["avg_i_shock_train_ann_pct"] - delta_all["avg_i_baseline_ann_pct"]
        delta_all["d_avg_x_pct"] = delta_all["avg_x_shock_train_pct"] - delta_all["avg_x_baseline_pct"]
    delta_all.to_csv(os.path.join(fig_dir, "critique_shock_train_delta_by_start.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_shock_train_delta_by_start.csv"))

    _save_compare_vs_baseline(
        delta,
        baseline_label=baseline_label,
        out_csv=os.path.join(fig_dir, "compare_shock_train_vs_baseline.csv"),
        cols=["d_peak_pi_ann_pct", "d_avg_i_ann_pct", "d_avg_x_pct"],
    )


def _build_bad_uncertainty(
    variants: Sequence[VariantBundle],
    *,
    fig_dir: str,
    baseline_label: str,
    T: int,
    B: int,
    burn_in: int,
    noise_mult: float,
    noise_scale: float,
) -> None:
    rows: List[Dict[str, str | float]] = []
    stress_rows: List[Dict[str, str | float]] = []
    for v in variants:
        x0 = v.x0.repeat(int(B), 1)
        bad_reg = int(_bad_regime(v.params))
        sim_base = _simulate_custom(
            v,
            T=T,
            x0=x0,
            regime_path=None,
            epst_path=np.zeros(T, dtype=np.float64),
            noise_scale=float(noise_scale),
            noise_mult=1.0,
            seed=123,
        )
        sim_hi = _simulate_custom(
            v,
            T=T,
            x0=x0,
            regime_path=None,
            epst_path=np.zeros(T, dtype=np.float64),
            noise_scale=float(noise_scale),
            noise_mult=float(noise_mult),
            seed=123,
        )
        sim_base = _add_output_gap(sim_base, v.params)
        sim_hi = _add_output_gap(sim_hi, v.params)

        m_base_pi = _regime_moments(
            {"pi": _ann(sim_base["pi"]), "s": sim_base["s"]},
            "pi",
            burn_in=int(burn_in),
        )
        m_hi_pi = _regime_moments(
            {"pi": _ann(sim_hi["pi"]), "s": sim_hi["s"]},
            "pi",
            burn_in=int(burn_in),
        )

        for reg in sorted(set(m_base_pi.keys()) | set(m_hi_pi.keys())):
            rows.append(
                {
                    "variant": v.label,
                    "scenario": "baseline_uncertainty",
                    "regime": reg,
                    "pi_mean_ann_pct": m_base_pi[reg]["mean"],
                    "pi_std_ann_pct": m_base_pi[reg]["std"],
                }
            )
            rows.append(
                {
                    "variant": v.label,
                    "scenario": f"high_uncertainty_x{noise_mult:g}",
                    "regime": reg,
                    "pi_mean_ann_pct": m_hi_pi[reg]["mean"],
                    "pi_std_ann_pct": m_hi_pi[reg]["std"],
                }
            )

        s_base = np.asarray(sim_base["s"][burn_in:, :], dtype=np.int64)
        s_hi = np.asarray(sim_hi["s"][burn_in:, :], dtype=np.int64)
        m_stress_base = s_base >= int(bad_reg)
        m_stress_hi = s_hi >= int(bad_reg)
        pi_b_base = _ann(sim_base["pi"])[burn_in:, :][m_stress_base]
        pi_b_hi = _ann(sim_hi["pi"])[burn_in:, :][m_stress_hi]
        stress_rows.append(
            {
                "variant": v.label,
                "scenario": "baseline_uncertainty",
                "regime": "stress",
                "pi_mean_ann_pct": float(np.nanmean(pi_b_base)) if pi_b_base.size else float("nan"),
                "pi_std_ann_pct": float(np.nanstd(pi_b_base)) if pi_b_base.size else float("nan"),
            }
        )
        stress_rows.append(
            {
                "variant": v.label,
                "scenario": f"high_uncertainty_x{noise_mult:g}",
                "regime": "stress",
                "pi_mean_ann_pct": float(np.nanmean(pi_b_hi)) if pi_b_hi.size else float("nan"),
                "pi_std_ann_pct": float(np.nanstd(pi_b_hi)) if pi_b_hi.size else float("nan"),
            }
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(pi_b_base, bins=60, alpha=0.55, label="baseline")
        ax[0].hist(pi_b_hi, bins=60, alpha=0.55, label=f"high uncertainty x{noise_mult:g}")
        ax[0].set_title("(a) Stress-regime inflation distribution")
        ax[0].set_xlabel("annualized inflation, %")
        ax[0].legend()

        ax[1].bar(
            ["baseline", f"x{noise_mult:g}"],
            [
                float(np.nanstd(pi_b_base)) if pi_b_base.size else float("nan"),
                float(np.nanstd(pi_b_hi)) if pi_b_hi.size else float("nan"),
            ],
        )
        ax[1].set_title("(b) std(inflation | stress)")
        ax[1].set_ylabel("annualized pp")
        fig.suptitle(f"Critique Bad-Uncertainty: {v.label}", y=1.03)
        plt.tight_layout()
        _savefig(fig_dir, f"critique_bad_uncertainty_{_slug(v.label)}.png")
        plt.close(fig)

    moments = pd.DataFrame(rows + stress_rows)
    moments.to_csv(os.path.join(fig_dir, "critique_bad_uncertainty_moments.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_bad_uncertainty_moments.csv"))

    piv = moments.pivot_table(index=["variant", "regime"], columns="scenario", values=["pi_mean_ann_pct", "pi_std_ann_pct"])
    piv.to_csv(os.path.join(fig_dir, "critique_bad_uncertainty_pivot.csv"))
    print("[saved]", os.path.join(fig_dir, "critique_bad_uncertainty_pivot.csv"))

    hi_key = f"high_uncertainty_x{noise_mult:g}"
    bad_hi = moments[(moments["scenario"] == hi_key) & (moments["regime"] == "stress")].copy()
    _save_compare_vs_baseline(
        bad_hi,
        baseline_label=baseline_label,
        out_csv=os.path.join(fig_dir, "compare_bad_uncertainty_stress_regime_vs_baseline.csv"),
        cols=["pi_mean_ann_pct", "pi_std_ann_pct"],
    )


def _build_combined(
    variants: Sequence[VariantBundle],
    *,
    fig_dir: str,
    baseline_label: str,
    T: int,
    B: int,
    shock_times: Sequence[int],
    shock_size: float,
    noise_mult: float,
    noise_scale: float,
) -> None:
    base_epst = np.zeros(T, dtype=np.float64)
    comb_epst = np.zeros(T, dtype=np.float64)
    for k in shock_times:
        if 0 <= int(k) < int(T):
            comb_epst[int(k)] = float(shock_size)

    rows: List[Dict[str, str | float]] = []
    rows_all: List[Dict[str, str | float]] = []
    for v in variants:
        normal_reg = 0 if 0 in v.x0_by_regime else min(v.x0_by_regime.keys())
        for start_reg in sorted(v.x0_by_regime.keys()):
            start_label = _regime_label(start_reg)
            regime_path = [int(start_reg)] + [int(_bad_regime(v.params))] * int(T)
            x0 = v.x0_by_regime[start_reg].repeat(int(B), 1)
            sim_base = _simulate_custom(
                v,
                T=T,
                x0=x0,
                regime_path=regime_path,
                epst_path=base_epst,
                noise_scale=float(noise_scale),
                noise_mult=1.0,
                seed=123,
            )
            sim_comb = _simulate_custom(
                v,
                T=T,
                x0=x0,
                regime_path=regime_path,
                epst_path=comb_epst,
                noise_scale=float(noise_scale),
                noise_mult=float(noise_mult),
                seed=123,
            )
            sim_base = _add_output_gap(sim_base, v.params)
            sim_comb = _add_output_gap(sim_comb, v.params)

            t = np.arange(T + 1)
            pi_b_q10, pi_b_q50, pi_b_q90 = _qband(_ann(sim_base["pi"]))
            pi_c_q10, pi_c_q50, pi_c_q90 = _qband(_ann(sim_comb["pi"]))
            x_b_q10, x_b_q50, x_b_q90 = _qband(100.0 * sim_base["x"])
            x_c_q10, x_c_q50, x_c_q90 = _qband(100.0 * sim_comb["x"])

            fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            ax[0].plot(t, pi_b_q50, color="tab:blue", label="baseline median")
            ax[0].fill_between(t, pi_b_q10, pi_b_q90, color="tab:blue", alpha=0.2, label="baseline 10-90")
            ax[0].plot(t, pi_c_q50, color="tab:red", label="combined median")
            ax[0].fill_between(t, pi_c_q10, pi_c_q90, color="tab:red", alpha=0.2, label="combined 10-90")
            ax[0].set_title("(a) Inflation (ann. %)")
            ax[0].legend(ncol=2)

            ax[1].plot(t, x_b_q50, color="tab:blue", label="baseline median")
            ax[1].fill_between(t, x_b_q10, x_b_q90, color="tab:blue", alpha=0.2, label="baseline 10-90")
            ax[1].plot(t, x_c_q50, color="tab:red", label="combined median")
            ax[1].fill_between(t, x_c_q10, x_c_q90, color="tab:red", alpha=0.2, label="combined 10-90")
            ax[1].set_title("(b) Output gap (%)")
            ax[1].set_xlabel("quarter")
            ax[1].legend(ncol=2)

            fig.suptitle(f"Critique Combined Scenario: {v.label} (start={start_label})", y=1.02)
            plt.tight_layout()
            _savefig(fig_dir, f"critique_combined_{_slug(v.label)}_start_{start_label}.png")
            plt.close(fig)

            row = {
                "variant": v.label,
                "start_regime": start_label,
                "start_regime_id": int(start_reg),
                "median_peak_pi_base_ann_pct": float(np.max(pi_b_q50[1:])),
                "median_peak_pi_comb_ann_pct": float(np.max(pi_c_q50[1:])),
                "median_min_x_base_pct": float(np.min(x_b_q50[1:])),
                "median_min_x_comb_pct": float(np.min(x_c_q50[1:])),
            }
            rows_all.append(row)
            if int(start_reg) == int(normal_reg):
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(fig_dir, "critique_combined_summary.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_combined_summary.csv"))
    pd.DataFrame(rows_all).to_csv(
        os.path.join(fig_dir, "critique_combined_summary_by_start.csv"),
        index=False,
    )
    print("[saved]", os.path.join(fig_dir, "critique_combined_summary_by_start.csv"))

    delta = summary.copy()
    delta["d_peak_pi_ann_pct"] = delta["median_peak_pi_comb_ann_pct"] - delta["median_peak_pi_base_ann_pct"]
    delta["d_min_x_pct"] = delta["median_min_x_comb_pct"] - delta["median_min_x_base_pct"]
    delta.to_csv(os.path.join(fig_dir, "critique_combined_delta.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_combined_delta.csv"))

    delta_all = pd.DataFrame(rows_all)
    if not delta_all.empty:
        delta_all = delta_all.copy()
        delta_all["d_peak_pi_ann_pct"] = delta_all["median_peak_pi_comb_ann_pct"] - delta_all["median_peak_pi_base_ann_pct"]
        delta_all["d_min_x_pct"] = delta_all["median_min_x_comb_pct"] - delta_all["median_min_x_base_pct"]
    delta_all.to_csv(os.path.join(fig_dir, "critique_combined_delta_by_start.csv"), index=False)
    print("[saved]", os.path.join(fig_dir, "critique_combined_delta_by_start.csv"))

    _save_compare_vs_baseline(
        delta,
        baseline_label=baseline_label,
        out_csv=os.path.join(fig_dir, "compare_combined_vs_baseline.csv"),
        cols=["d_peak_pi_ann_pct", "d_min_x_pct"],
    )


def _parse_int_list(s: str) -> List[int]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return [int(x) for x in items]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def build_compare(
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
    scenarios: Sequence[str],
    baseline_label: str,
    baseline_run_dir: str | None,
    include_robust: bool,
    robust_label: str,
    robust_run_dir: str | None,
    robust_rule: RobustModTaylorRuleConfig,
    shock_train_T: int,
    shock_train_times: Sequence[int],
    shock_train_size: float,
    bad_uncert_T: int,
    bad_uncert_B: int,
    bad_uncert_burn_in: int,
    bad_uncert_mult: float,
    bad_uncert_noise: float,
    combined_T: int,
    combined_B: int,
    combined_times: Sequence[int],
    combined_size: float,
    combined_mult: float,
    combined_noise: float,
) -> None:
    mode = _resolve_cons_mode(cons_mode)
    variants: List[VariantBundle] = []

    variants.append(
        _load_variant_bundle(
            artifacts_root=artifacts_root,
            label=baseline_label,
            run_dir_override=baseline_run_dir,
            use_selected=use_selected,
            device=device,
            dtype=dtype,
            ensure_postprocess=ensure_postprocess,
            ensure_ir=ensure_ir,
            force_rebuild_postprocess=force_rebuild_postprocess,
            force_rebuild_ir=force_rebuild_ir,
            cons_mode=mode,
            robust_rule=None,
        )
    )

    if bool(include_robust):
        variants.append(
            _load_variant_bundle(
                artifacts_root=artifacts_root,
                label=robust_label,
                run_dir_override=robust_run_dir,
                use_selected=use_selected,
                device=device,
                dtype=dtype,
                ensure_postprocess=ensure_postprocess,
                ensure_ir=ensure_ir,
                force_rebuild_postprocess=force_rebuild_postprocess,
                force_rebuild_ir=force_rebuild_ir,
                cons_mode=mode,
                robust_rule=robust_rule,
            )
        )

    wanted = {str(s).strip().lower() for s in scenarios}
    valid = {"shock_train", "bad_uncertainty", "combined", "all"}
    if not wanted.issubset(valid):
        raise ValueError(f"Unknown scenario(s): {sorted(wanted.difference(valid))}")
    if "all" in wanted:
        wanted = {"shock_train", "bad_uncertainty", "combined"}

    os.makedirs(fig_dir, exist_ok=True)
    if "shock_train" in wanted:
        _build_shock_train(
            variants,
            fig_dir=fig_dir,
            baseline_label=baseline_label,
            T=int(shock_train_T),
            shock_times=[int(v) for v in shock_train_times],
            shock_size=float(shock_train_size),
        )
    if "bad_uncertainty" in wanted:
        _build_bad_uncertainty(
            variants,
            fig_dir=fig_dir,
            baseline_label=baseline_label,
            T=int(bad_uncert_T),
            B=int(bad_uncert_B),
            burn_in=int(bad_uncert_burn_in),
            noise_mult=float(bad_uncert_mult),
            noise_scale=float(bad_uncert_noise),
        )
    if "combined" in wanted:
        _build_combined(
            variants,
            fig_dir=fig_dir,
            baseline_label=baseline_label,
            T=int(combined_T),
            B=int(combined_B),
            shock_times=[int(v) for v in combined_times],
            shock_size=float(combined_size),
            noise_mult=float(combined_mult),
            noise_scale=float(combined_noise),
        )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare mod_taylor variants (baseline vs robust) on critique scenarios using 120/121/122 metrics."
    )
    ap.add_argument("--artifacts-root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "artifacts", "critique_mod_taylor_variants"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--use-selected", type=str, default="true")
    ap.add_argument("--ensure-postprocess", type=str, default="true")
    ap.add_argument("--ensure-ir", type=str, default="true")
    ap.add_argument("--cons-mode", type=str, default="author", choices=["author", "paper"])
    ap.add_argument("--force-rebuild-postprocess", type=str, default="false")
    ap.add_argument("--force-rebuild-ir", type=str, default="false")

    ap.add_argument("--scenarios", type=str, default="all", help="comma-separated: all,shock_train,bad_uncertainty,combined")

    ap.add_argument("--baseline-label", type=str, default="baseline_curriculum")
    ap.add_argument("--baseline-run-dir", type=str, default="")
    ap.add_argument("--include-robust", type=str, default="true")
    ap.add_argument("--robust-label", type=str, default="robust_curriculum")
    ap.add_argument("--robust-run-dir", type=str, default="")

    ap.add_argument("--robust-kappa", type=float, default=0.02)
    ap.add_argument("--robust-uncert-ref", type=float, default=0.0)
    ap.add_argument("--robust-max-premium", type=float, default=0.03)
    ap.add_argument("--robust-bad-only", type=str, default="true")
    ap.add_argument(
        "--robust-bad-state",
        type=int,
        default=1,
        help="Stress-state threshold for robust premium (applies to regimes s >= this index).",
    )

    ap.add_argument("--shock-train-T", type=int, default=64)
    ap.add_argument("--shock-train-times", type=str, default="0,4,8,12")
    ap.add_argument("--shock-train-size", type=float, default=1.25)

    ap.add_argument("--bad-uncert-T", type=int, default=320)
    ap.add_argument("--bad-uncert-B", type=int, default=512)
    ap.add_argument("--bad-uncert-burn-in", type=int, default=80)
    ap.add_argument("--bad-uncert-mult", type=float, default=2.0)
    ap.add_argument("--bad-uncert-noise", type=float, default=1.0)

    ap.add_argument("--combined-T", type=int, default=80)
    ap.add_argument("--combined-B", type=int, default=512)
    ap.add_argument("--combined-times", type=str, default="0,4,8,12")
    ap.add_argument("--combined-size", type=float, default=1.25)
    ap.add_argument("--combined-mult", type=float, default=2.0)
    ap.add_argument("--combined-noise", type=float, default=1.0)

    args = ap.parse_args(list(argv) if argv is not None else None)

    robust_rule = RobustModTaylorRuleConfig(
        enabled=True,
        bad_state=int(args.robust_bad_state),
        kappa=float(args.robust_kappa),
        uncertainty_ref=float(args.robust_uncert_ref),
        max_premium=float(args.robust_max_premium),
        bad_only=bool(_parse_bool(args.robust_bad_only)),
    )

    build_compare(
        artifacts_root=str(args.artifacts_root),
        fig_dir=str(args.fig_dir),
        device=str(args.device),
        dtype=_parse_dtype(str(args.dtype)),
        use_selected=bool(_parse_bool(args.use_selected)),
        ensure_postprocess=bool(_parse_bool(args.ensure_postprocess)),
        ensure_ir=bool(_parse_bool(args.ensure_ir)),
        force_rebuild_postprocess=bool(_parse_bool(args.force_rebuild_postprocess)),
        force_rebuild_ir=bool(_parse_bool(args.force_rebuild_ir)),
        cons_mode=str(args.cons_mode),
        scenarios=_parse_str_list(str(args.scenarios)),
        baseline_label=str(args.baseline_label),
        baseline_run_dir=(str(args.baseline_run_dir).strip() or None),
        include_robust=bool(_parse_bool(args.include_robust)),
        robust_label=str(args.robust_label),
        robust_run_dir=(str(args.robust_run_dir).strip() or None),
        robust_rule=robust_rule,
        shock_train_T=int(args.shock_train_T),
        shock_train_times=_parse_int_list(args.shock_train_times),
        shock_train_size=float(args.shock_train_size),
        bad_uncert_T=int(args.bad_uncert_T),
        bad_uncert_B=int(args.bad_uncert_B),
        bad_uncert_burn_in=int(args.bad_uncert_burn_in),
        bad_uncert_mult=float(args.bad_uncert_mult),
        bad_uncert_noise=float(args.bad_uncert_noise),
        combined_T=int(args.combined_T),
        combined_B=int(args.combined_B),
        combined_times=_parse_int_list(args.combined_times),
        combined_size=float(args.combined_size),
        combined_mult=float(args.combined_mult),
        combined_noise=float(args.combined_noise),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
