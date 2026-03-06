from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Mapping, Iterable, Sequence

import json
import os

import numpy as np
import pandas as pd

from .paper_targets import TABLE2_TARGETS

TargetMap = Dict[Tuple[str, str], Dict[str, float]]

DEFAULT_TARGET_METRICS: Tuple[str, ...] = (
    "pi_sss_pct",
    "pi_mean_pct",
    "pi_std_pct",
    "pi_skew",
    "x_sss_pct",
    "x_mean_pct",
    "x_std_pct",
    "x_skew",
    "r_sss_pct",
    "r_mean_pct",
    "r_std_pct",
    "r_skew",
    "i_sss_pct",
    "i_mean_pct",
    "i_std_pct",
    "i_skew",
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    max_abs_dev: float
    failures: pd.DataFrame


def _ordered_unique(items: Iterable[str], *, preferred: Sequence[str] = ()) -> List[str]:
    out: List[str] = []
    seen = set()
    pref = [str(x) for x in preferred]
    vals = [str(x) for x in items]
    for p in pref:
        if p in vals and p not in seen:
            out.append(p)
            seen.add(p)
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _normalize_targets_map(targets: Mapping[Tuple[str, str], Mapping[str, float]]) -> TargetMap:
    out: TargetMap = {}
    for key, vals in targets.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "targets keys must be 2-tuples (policy, regime), "
                f"got key={key!r}"
            )
        pol = str(key[0]).strip()
        reg = str(key[1]).strip()
        vv: Dict[str, float] = {}
        for m, x in vals.items():
            try:
                vv[str(m)] = float(x)
            except Exception:
                continue
        out[(pol, reg)] = vv
    return out


def check_table_against_targets(
    df: pd.DataFrame,
    *,
    targets: Mapping[Tuple[str, str], Mapping[str, float]],
    policies: Tuple[str, ...] | None = None,
    regimes: Tuple[str, ...] | None = None,
    tol_abs: float = 0.15,
    # separate tolerance for skew (often noisier)
    tol_abs_skew: float = 0.5,
) -> CheckResult:
    """Compare a reproduced table dataframe against an explicit target map.

    `targets` format:
      {(policy, regime): {metric: value, ...}, ...}
    """
    targets_norm = _normalize_targets_map(targets)

    rows: List[Dict[str, Any]] = []
    max_dev = 0.0

    df2 = df.copy()
    # normalize
    if "policy" not in df2.columns or "regime" not in df2.columns:
        raise KeyError("df must contain columns 'policy' and 'regime'.")
    df2["policy"] = df2["policy"].astype(str).str.strip()
    df2["regime"] = df2["regime"].astype(str).str.strip()

    if policies is None:
        pol_set = [str(pol) for pol, _ in targets_norm.keys()]
        policies = tuple(_ordered_unique(pol_set))
    else:
        policies = tuple(str(p).strip() for p in policies)

    if regimes is None:
        reg_set = {
            str(reg)
            for pol, reg in targets_norm.keys()
            if str(pol) in set(str(p) for p in policies)
        }
        pref = ["normal", "bad", "severe"]
        reg_order = _ordered_unique(sorted(reg_set), preferred=pref)
        regimes = tuple(reg_order)
    else:
        regimes = tuple(str(r).strip() for r in regimes)

    for pol in policies:
        for reg in regimes:
            tgt = targets_norm.get((pol, reg))
            if tgt is None:
                continue
            sub = df2[(df2.policy == pol) & (df2.regime == reg)]
            if sub.empty:
                rows.append({"policy": pol, "regime": reg, "metric": "__row__", "paper": np.nan, "model": np.nan, "abs_dev": np.inf, "tol": 0.0, "ok": False})
                max_dev = float("inf")
                continue
            rec = sub.iloc[0].to_dict()
            for m, paper_val in tgt.items():
                if m not in rec:
                    continue
                if not np.isfinite(float(paper_val)):
                    # Allow placeholders (NaN/Inf) in custom target maps.
                    continue
                model_val = float(rec[m])
                dev = abs(model_val - float(paper_val))
                tol = tol_abs_skew if m.endswith("_skew") else tol_abs
                ok = dev <= tol
                rows.append({
                    "policy": pol,
                    "regime": reg,
                    "metric": m,
                    "paper": float(paper_val),
                    "model": model_val,
                    "abs_dev": dev,
                    "tol": tol,
                    "ok": ok,
                })
                if np.isfinite(dev):
                    max_dev = max(max_dev, dev)

    out = pd.DataFrame(rows, columns=["policy", "regime", "metric", "paper", "model", "abs_dev", "tol", "ok"])
    failures = out[~out["ok"]].sort_values(["policy", "regime", "abs_dev"], ascending=[True, True, False])
    ok_all = failures.empty
    return CheckResult(ok=ok_all, max_abs_dev=float(max_dev), failures=failures)


def check_table2(
    df: pd.DataFrame,
    *,
    policies: Tuple[str, ...] = ("flex", "commitment", "discretion"),
    regimes: Tuple[str, ...] | None = None,
    tol_abs: float = 0.15,
    # separate tolerance for skew (often noisier)
    tol_abs_skew: float = 0.5,
) -> CheckResult:
    """Legacy paper Table-2 check (normal/bad targets from paper_targets)."""
    return check_table_against_targets(
        df,
        targets=TABLE2_TARGETS,
        policies=policies,
        regimes=regimes,
        tol_abs=tol_abs,
        tol_abs_skew=tol_abs_skew,
    )


def build_targets_from_table(
    df: pd.DataFrame,
    *,
    policies: Tuple[str, ...] | None = None,
    regimes: Tuple[str, ...] | None = None,
    metrics: Tuple[str, ...] = DEFAULT_TARGET_METRICS,
) -> TargetMap:
    """Extract a target map from a produced table frame (supports 3+ regimes)."""
    if "policy" not in df.columns or "regime" not in df.columns:
        raise KeyError("df must contain columns 'policy' and 'regime'.")
    df2 = df.copy()
    df2["policy"] = df2["policy"].astype(str).str.strip()
    df2["regime"] = df2["regime"].astype(str).str.strip()

    if policies is None:
        policies = tuple(_ordered_unique(df2["policy"].tolist()))
    if regimes is None:
        regimes = tuple(_ordered_unique(df2["regime"].tolist(), preferred=("normal", "bad", "severe")))

    out: TargetMap = {}
    for pol in policies:
        for reg in regimes:
            sub = df2[(df2["policy"] == str(pol)) & (df2["regime"] == str(reg))]
            if sub.empty:
                continue
            row = sub.iloc[0]
            vals: Dict[str, float] = {}
            for m in metrics:
                if m not in row.index:
                    continue
                x = float(row[m])
                if np.isfinite(x):
                    vals[str(m)] = x
            if vals:
                out[(str(pol), str(reg))] = vals
    return out


def targets_to_nested_dict(targets: Mapping[Tuple[str, str], Mapping[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Convert tuple-key target map to JSON-friendly nested dict."""
    norm = _normalize_targets_map(targets)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (pol, reg), vals in norm.items():
        out.setdefault(str(pol), {})
        out[str(pol)][str(reg)] = {str(k): float(v) for k, v in vals.items()}
    return out


def targets_from_nested_dict(blob: Mapping[str, Any]) -> TargetMap:
    """Load tuple-key target map from nested dict.

    Accepted shapes:
      - {"targets": {policy: {regime: {metric: value}}}}
      - {policy: {regime: {metric: value}}}
    """
    core: Any = blob.get("targets", blob) if isinstance(blob, Mapping) else blob
    if not isinstance(core, Mapping):
        raise ValueError("targets blob must be a mapping.")
    out: TargetMap = {}
    for pol, reg_map in core.items():
        if not isinstance(reg_map, Mapping):
            continue
        for reg, metrics in reg_map.items():
            if not isinstance(metrics, Mapping):
                continue
            vals: Dict[str, float] = {}
            for m, x in metrics.items():
                try:
                    vals[str(m)] = float(x)
                except Exception:
                    continue
            out[(str(pol), str(reg))] = vals
    return out


def save_targets_json(
    path: str,
    targets: Mapping[Tuple[str, str], Mapping[str, float]],
    *,
    meta: Mapping[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "format": "econml.targets.v1",
        "targets": targets_to_nested_dict(targets),
    }
    if meta is not None:
        payload["meta"] = dict(meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def load_targets_json(path: str) -> TargetMap:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return targets_from_nested_dict(blob)
