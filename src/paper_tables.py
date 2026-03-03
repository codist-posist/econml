from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import ModelParams
from .table2_builder import build_table0


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


def _paper_moments_wide_table(
    df: pd.DataFrame,
    *,
    policies: List[Tuple[str, str]],
) -> pd.DataFrame:
    metrics = [
        ("Inflation (SSS)", "pi_sss_pct"),
        ("Inflation (Mean)", "pi_mean_pct"),
        ("Inflation (Std)", "pi_std_pct"),
        ("Inflation (Skew)", "pi_skew"),
        ("Output gap (SSS)", "x_sss_pct"),
        ("Output gap (Mean)", "x_mean_pct"),
        ("Output gap (Std)", "x_std_pct"),
        ("Output gap (Skew)", "x_skew"),
        ("Real interest rates (SSS)", "r_sss_pct"),
        ("Real interest rates (Mean)", "r_mean_pct"),
        ("Real interest rates (Std)", "r_std_pct"),
        ("Real interest rates (Skew)", "r_skew"),
        ("Nominal interest rates (SSS)", "i_sss_pct"),
        ("Nominal interest rates (Mean)", "i_mean_pct"),
        ("Nominal interest rates (Std)", "i_std_pct"),
        ("Nominal interest rates (Skew)", "i_skew"),
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
    strict_selected: bool = False,
    weights_source: str = "auto",
    sss_source: str = "sim_conditional",
    strict_author_table2: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build paper-style Table 2/3/4 using the common table builder and reshape to article layout.

    Paper tables are always produced from `sim_conditional` moments.
    """
    if str(sss_source).strip().lower() != "sim_conditional":
        print(
            "[build_paper_tables_2_4] WARNING: paper tables enforce "
            "sss_source='sim_conditional'; overriding requested source."
        )
    sss_source_used = "sim_conditional"

    try:
        df = build_table0(
            artifacts_root,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            strict_selected=strict_selected,
            weights_source=weights_source,
            include_rules=True,
            include_zlb=True,
            sss_source=sss_source_used,
            strict_author_table2=strict_author_table2,
        )
    except FileNotFoundError as e:
        msg = str(e).lower()
        if "zlb" not in msg:
            raise
        # Fallback: build non-ZLB part and append NaN placeholders for ZLB rows.
        df = build_table0(
            artifacts_root,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            strict_selected=strict_selected,
            weights_source=weights_source,
            include_rules=True,
            include_zlb=False,
            sss_source=sss_source_used,
            strict_author_table2=strict_author_table2,
        )
        cols = list(df.columns)
        extras = []
        for pol in ("taylor_zlb", "mod_taylor_zlb", "discretion_zlb", "commitment_zlb"):
            for reg in ("normal", "bad"):
                row = {c: np.nan for c in cols}
                row["policy"] = pol
                row["regime"] = reg
                if "sss_source" in row:
                    row["sss_source"] = sss_source_used
                extras.append(row)
        df = pd.concat([df, pd.DataFrame(extras)], ignore_index=True)
        print("[build_paper_tables_2_4] WARNING: missing ZLB runs; Table 4 contains NaN placeholders.")

    table2 = _paper_moments_wide_table(
        df,
        policies=[
            ("flex", "Flex. prices"),
            ("taylor", "Taylor rule"),
            ("mod_taylor", "Mod. Taylor rule"),
        ],
    )

    table3 = _paper_moments_wide_table(
        df,
        policies=[
            ("discretion", "Discretion"),
            ("commitment", "Commitment"),
        ],
    )

    table4 = _paper_moments_wide_table(
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


def build_taylor_para_robustness_table(
    artifacts_root: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    use_selected: bool = True,
    strict_selected: bool = False,
    weights_source: str = "auto",
    sss_source: str = "sim_conditional",
    strict_author_table2: bool = True,
) -> pd.DataFrame:
    """
    Robustness-only table: compare Taylor variants, including taylor_para
    (rate as explicit network output). Not part of the main paper tables.
    """
    if str(sss_source).strip().lower() != "sim_conditional":
        print(
            "[build_taylor_para_robustness_table] WARNING: enforcing "
            "sss_source='sim_conditional' for robustness table."
        )
    df = build_table0(
        artifacts_root,
        device=device,
        dtype=dtype,
        use_selected=use_selected,
        strict_selected=strict_selected,
        weights_source=weights_source,
        include_rules=True,
        include_para=True,
        include_zlb=False,
        sss_source="sim_conditional",
        strict_author_table2=strict_author_table2,
    )
    keep = ["taylor", "taylor_para", "mod_taylor"]
    out = df[df["policy"].isin(keep)].copy()
    out["policy"] = pd.Categorical(out["policy"], categories=keep, ordered=True)
    out["regime"] = pd.Categorical(out["regime"], categories=["normal", "bad"], ordered=True)
    return out.sort_values(["policy", "regime"]).reset_index(drop=True)
