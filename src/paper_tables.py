from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import ModelParams
from .table2_builder import build_table2


def build_table1_calibration(params: ModelParams | None = None) -> pd.DataFrame:
    """Build a compact calibration table (paper-style Table 1 helper)."""
    p = params if params is not None else ModelParams().to_torch()
    rows = [
        ("Discount factor", "beta", float(p.beta)),
        ("Risk aversion", "gamma", float(p.gamma)),
        ("Inverse Frisch", "omega", float(p.omega)),
        ("Calvo parameter", "theta", float(p.theta)),
        ("Elasticity of substitution", "eps", float(p.eps)),
        ("Tax wedge level", "tau_bar", float(p.tau_bar)),
        ("Government purchases level", "g_bar", float(p.g_bar)),
        ("Persistent supply-shock level", "eta_bar", float(p.eta_bar)),
        ("TFP AR(1)", "rho_A", float(p.rho_A)),
        ("Cost-push AR(1)", "rho_tau", float(p.rho_tau)),
        ("Gov. spending AR(1)", "rho_g", float(p.rho_g)),
        ("TFP shock st.dev", "sigma_A", float(p.sigma_A)),
        ("Cost-push shock st.dev", "sigma_tau", float(p.sigma_tau)),
        ("Gov. spending shock st.dev", "sigma_g", float(p.sigma_g)),
        ("Transition prob. normal->bad", "p12", float(p.p12)),
        ("Transition prob. bad->normal", "p21", float(p.p21)),
        ("Inflation target (quarterly net)", "pi_bar", float(p.pi_bar)),
        ("Taylor inflation response", "psi", float(p.psi)),
    ]
    return pd.DataFrame(rows, columns=["Description", "Symbol", "Value"])


def _paper_sss_wide_table(
    df: pd.DataFrame,
    *,
    policies: List[Tuple[str, str]],
) -> pd.DataFrame:
    metrics = [
        ("Inflation", "pi_sss_pct"),
        ("Output gap", "x_sss_pct"),
        ("Real interest rates", "r_sss_pct"),
        ("Nominal interest rates", "i_sss_pct"),
    ]
    regimes = [("normal", "normal times"), ("bad", "persistent supply shock")]

    rows: List[Dict[str, object]] = []
    for metric_name, col in metrics:
        for reg_key, reg_label in regimes:
            row: Dict[str, object] = {
                "Variable": metric_name,
                "Regime": reg_label,
            }
            for pkey, plabel in policies:
                sub = df[(df["policy"] == pkey) & (df["regime"] == reg_key)]
                row[plabel] = float(sub.iloc[0][col]) if len(sub) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def build_paper_tables_2_4(
    artifacts_root: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    use_selected: bool = True,
    sss_source: str = "sim_conditional",
) -> Dict[str, pd.DataFrame]:
    """
    Build paper-style Table 2/3/4 using the common table builder and reshape to article layout.

    Recommended:
      - `sss_source='sim_conditional'` for paper/author table values
        (conditional means from long simulation by regime).
      - `sss_source='fixed_point'` only as a diagnostic comparison.
    """
    try:
        df = build_table2(
            artifacts_root,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            include_rules=True,
            include_zlb=True,
            sss_source=sss_source,
        )
    except FileNotFoundError as e:
        msg = str(e).lower()
        if "zlb" not in msg:
            raise
        # Fallback: build non-ZLB part and append NaN placeholders for ZLB rows.
        df = build_table2(
            artifacts_root,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            include_rules=True,
            include_zlb=False,
            sss_source=sss_source,
        )
        cols = list(df.columns)
        extras = []
        for pol in ("taylor_zlb", "mod_taylor_zlb", "discretion_zlb", "commitment_zlb"):
            for reg in ("normal", "bad"):
                row = {c: np.nan for c in cols}
                row["policy"] = pol
                row["regime"] = reg
                if "sss_source" in row:
                    row["sss_source"] = sss_source
                extras.append(row)
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)
        print("[build_paper_tables_2_4] WARNING: missing ZLB runs; Table 4 contains NaN placeholders.")

    table2 = _paper_sss_wide_table(
        df,
        policies=[
            ("flex", "Flex. prices"),
            ("taylor", "Taylor rule"),
            ("mod_taylor", "Mod. Taylor rule"),
        ],
    )

    table3 = _paper_sss_wide_table(
        df,
        policies=[
            ("discretion", "Discretion"),
            ("commitment", "Commitment"),
        ],
    )

    table4 = _paper_sss_wide_table(
        df,
        policies=[
            ("taylor_zlb", "Taylor rule"),
            ("mod_taylor_zlb", "Mod. Taylor rule"),
            ("discretion_zlb", "Discretion"),
            ("commitment_zlb", "Commitment"),
        ],
    )

    return {
        "table2": table2,
        "table3": table3,
        "table4": table4,
    }
