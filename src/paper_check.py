from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from .paper_targets import TABLE2_TARGETS


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    max_abs_dev: float
    failures: pd.DataFrame


def check_table2(
    df: pd.DataFrame,
    *,
    policies: Tuple[str, ...] = ("flex", "commitment", "discretion"),
    regimes: Tuple[str, ...] = ("normal", "bad"),
    tol_abs: float = 0.15,
    # separate tolerance for skew (often noisier)
    tol_abs_skew: float = 0.5,
) -> CheckResult:
    """Compare a reproduced Table-2 dataframe against the paper's Table 2.

    Parameters
    ----------
    df:
        Output of table2_builder.build_table2.
    tol_abs:
        Absolute tolerance for levels in percent.
    tol_abs_skew:
        Absolute tolerance for skewness.
    """

    rows: List[Dict[str, Any]] = []
    max_dev = 0.0

    df2 = df.copy()
    # normalize
    df2["policy"] = df2["policy"].astype(str)
    df2["regime"] = df2["regime"].astype(str)

    key_cols = [c for c in df2.columns if c not in ("policy", "regime")]

    for pol in policies:
        for reg in regimes:
            tgt = TABLE2_TARGETS.get((pol, reg))
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

    out = pd.DataFrame(rows)
    failures = out[~out["ok"]].sort_values(["policy", "regime", "abs_dev"], ascending=[True, True, False])
    ok_all = failures.empty
    return CheckResult(ok=ok_all, max_abs_dev=float(max_dev), failures=failures)
