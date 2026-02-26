#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch

# Allow running from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import ModelParams
from src.table2_builder import build_table2


def _dtype_from_name(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}. Use float32 or float64.")


def _load_variant_tables(
    *,
    artifacts_roots: Dict[str, str],
    params_by_variant: Dict[str, ModelParams],
    device: str,
    dtype: torch.dtype,
    include_rules: bool,
    use_selected: bool,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for variant, root in artifacts_roots.items():
        if not os.path.isdir(root):
            raise FileNotFoundError(f"artifacts root does not exist for variant '{variant}': {root}")
        df = build_table2(
            root,
            device=device,
            dtype=dtype,
            use_selected=use_selected,
            include_rules=include_rules,
            params=params_by_variant[variant],
        ).copy()
        df.insert(0, "variant", variant)
        out[variant] = df
    return out


def _delta_table(df_a: pd.DataFrame, df_b: pd.DataFrame, *, label: str = "B_minus_A") -> pd.DataFrame:
    key = ["policy", "regime"]
    metrics = [c for c in df_a.columns if c not in {"variant", *key}]
    a = df_a.set_index(key)[metrics]
    b = df_b.set_index(key)[metrics]
    common_index = a.index.intersection(b.index)
    a = a.loc[common_index]
    b = b.loc[common_index]
    d = (b - a).reset_index()
    d.insert(0, "delta", label)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare critique scenarios with regime-dependent volatility: "
            "A (sigma_tau by regime) vs B (all sigmas by regime)."
        )
    )
    ap.add_argument("--artifacts_a", required=True, help="Artifacts root for scenario A runs.")
    ap.add_argument("--artifacts_b", required=True, help="Artifacts root for scenario B runs.")
    ap.add_argument("--artifacts_baseline", default=None, help="Optional artifacts root for baseline runs.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--bad_multiplier", type=float, default=2.0)
    ap.add_argument("--normal_multiplier", type=float, default=1.0)
    ap.add_argument(
        "--include_rules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Taylor and modified Taylor rows (use --no-include_rules to disable).",
    )
    ap.add_argument(
        "--use_selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer selected_runs.json over latest complete run (use --no-use_selected to ignore).",
    )
    ap.add_argument("--out_dir", default=os.path.join(_ROOT, "artifacts", "comparisons"))
    args = ap.parse_args()

    dtype = _dtype_from_name(args.dtype)
    base = ModelParams(device=args.device, dtype=dtype).to_torch()
    params_a = base.with_sigma_tau_regime(
        bad_multiplier=float(args.bad_multiplier),
        normal_multiplier=float(args.normal_multiplier),
    )
    params_b = base.with_all_sigma_regime(
        bad_multiplier=float(args.bad_multiplier),
        normal_multiplier=float(args.normal_multiplier),
    )

    artifacts_roots: Dict[str, str] = {
        "variant_A_sigma_tau": os.path.abspath(args.artifacts_a),
        "variant_B_all_sigma": os.path.abspath(args.artifacts_b),
    }
    params_by_variant: Dict[str, ModelParams] = {
        "variant_A_sigma_tau": params_a,
        "variant_B_all_sigma": params_b,
    }
    if args.artifacts_baseline:
        artifacts_roots["baseline"] = os.path.abspath(args.artifacts_baseline)
        params_by_variant["baseline"] = base

    tables = _load_variant_tables(
        artifacts_roots=artifacts_roots,
        params_by_variant=params_by_variant,
        device=args.device,
        dtype=dtype,
        include_rules=bool(args.include_rules),
        use_selected=bool(args.use_selected),
    )

    out_all = pd.concat([tables[k] for k in tables.keys()], axis=0, ignore_index=True)
    delta = _delta_table(tables["variant_A_sigma_tau"], tables["variant_B_all_sigma"])

    os.makedirs(args.out_dir, exist_ok=True)
    path_all = os.path.join(args.out_dir, "table2_variants_combined.csv")
    path_delta = os.path.join(args.out_dir, "table2_variantB_minus_variantA.csv")
    out_all.to_csv(path_all, index=False)
    delta.to_csv(path_delta, index=False)

    print("Saved:")
    print(path_all)
    print(path_delta)
    print()
    print("Rows by variant:")
    for variant, df in tables.items():
        print(f"  {variant}: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

