#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.paper_check import (
    DEFAULT_TARGET_METRICS,
    build_targets_from_table,
    check_table_against_targets,
    load_targets_json,
    save_targets_json,
)


def _parse_csv_list(s: str | None) -> Tuple[str, ...] | None:
    if s is None:
        return None
    vals = [x.strip() for x in str(s).split(",")]
    vals = [x for x in vals if x]
    return tuple(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "3-regime target layer: export target maps from an existing table CSV "
            "and validate new tables against those targets."
        )
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_export = sub.add_parser("export", help="Export targets JSON from a table CSV.")
    ap_export.add_argument("--table-csv", required=True, help="Input table CSV (e.g., table0_reproduced_*.csv).")
    ap_export.add_argument("--out-json", required=True, help="Output targets JSON path.")
    ap_export.add_argument(
        "--policies",
        default="flex,commitment,discretion,taylor,mod_taylor",
        help="Comma-separated policy ids to include.",
    )
    ap_export.add_argument(
        "--regimes",
        default="normal,bad,severe",
        help="Comma-separated regime ids to include.",
    )
    ap_export.add_argument(
        "--metrics",
        default=",".join(DEFAULT_TARGET_METRICS),
        help="Comma-separated metric columns to include.",
    )

    ap_check = sub.add_parser("check", help="Check a table CSV against targets JSON.")
    ap_check.add_argument("--table-csv", required=True, help="Input table CSV to validate.")
    ap_check.add_argument("--targets-json", required=True, help="Targets JSON path.")
    ap_check.add_argument("--policies", default=None, help="Optional comma-separated policy filter.")
    ap_check.add_argument("--regimes", default=None, help="Optional comma-separated regime filter.")
    ap_check.add_argument("--tol-abs", type=float, default=0.15, help="Absolute tolerance for non-skew metrics.")
    ap_check.add_argument("--tol-abs-skew", type=float, default=0.5, help="Absolute tolerance for skew metrics.")
    ap_check.add_argument("--failures-csv", default=None, help="Optional path to save failure rows as CSV.")

    args = ap.parse_args()

    if args.cmd == "export":
        df = pd.read_csv(str(args.table_csv))
        policies = _parse_csv_list(args.policies)
        regimes = _parse_csv_list(args.regimes)
        metrics = _parse_csv_list(args.metrics)
        if metrics is None:
            metrics = DEFAULT_TARGET_METRICS
        targets = build_targets_from_table(
            df,
            policies=policies,
            regimes=regimes,
            metrics=tuple(metrics),
        )
        save_targets_json(
            str(args.out_json),
            targets,
            meta={
                "source_table_csv": str(args.table_csv),
                "policies": list(policies) if policies is not None else None,
                "regimes": list(regimes) if regimes is not None else None,
                "metrics": list(metrics),
            },
        )
        print(f"saved targets: {args.out_json}")
        print(f"entries: {len(targets)}")
        return 0

    if args.cmd == "check":
        df = pd.read_csv(str(args.table_csv))
        targets = load_targets_json(str(args.targets_json))
        policies = _parse_csv_list(args.policies)
        regimes = _parse_csv_list(args.regimes)
        res = check_table_against_targets(
            df,
            targets=targets,
            policies=policies,
            regimes=regimes,
            tol_abs=float(args.tol_abs),
            tol_abs_skew=float(args.tol_abs_skew),
        )
        print(f"ok: {res.ok}")
        print(f"max_abs_dev: {res.max_abs_dev:.6g}")
        print(f"n_failures: {len(res.failures)}")
        if len(res.failures) > 0:
            print(res.failures.head(25).to_string(index=False))
            if args.failures_csv:
                os.makedirs(os.path.dirname(str(args.failures_csv)) or ".", exist_ok=True)
                res.failures.to_csv(str(args.failures_csv), index=False)
                print(f"saved failures: {args.failures_csv}")
            return 1
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
