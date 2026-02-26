"""Strict Table 2 replication check.

Usage:
  python paper_check_table2.py --artifacts ./artifacts --device cpu

This script:
  1) builds the Table-2 dataframe from your artifacts,
  2) compares it to the paper's Table 2 numbers,
  3) prints failures (if any) and exits with non-zero status.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.table2_builder import build_table2
from src.paper_check import check_table2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Artifacts root (contains run dirs and flex/)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device")
    ap.add_argument("--tol", type=float, default=0.15, help="Abs tolerance for percent quantities")
    ap.add_argument("--tol-skew", type=float, default=0.5, help="Abs tolerance for skewness")
    args = ap.parse_args()

    df = build_table2(args.artifacts, device=args.device, use_selected=True, include_rules=False)
    chk = check_table2(df, tol_abs=args.tol, tol_abs_skew=args.tol_skew)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)

    if chk.ok:
        print("OK: Table 2 matches paper within tolerances.")
        print(f"max_abs_dev = {chk.max_abs_dev:.4f}")
        return 0

    print("FAIL: Table 2 deviates from paper beyond tolerances.")
    print(f"max_abs_dev = {chk.max_abs_dev:.4f}")
    print("\nFailures:")
    print(chk.failures.to_string(index=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
