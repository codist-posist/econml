#!/usr/bin/env python
from __future__ import annotations

import os
import sys

# Allow running this script from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, argparse
import sys
# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from src.table2_builder import build_table0, save_table0_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", default=os.path.join(_ROOT, "artifacts"))
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device: auto -> CUDA if available, else CPU.",
    )
    ap.add_argument(
        "--include_rules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Taylor and modified Taylor rows (use --no-include_rules to disable).",
    )
    ap.add_argument(
        "--include_zlb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include ZLB variants when include_rules=True (use --no-include_zlb to disable).",
    )
    ap.add_argument(
        "--include_para",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include taylor_para (rate as network output) robustness row.",
    )
    ap.add_argument(
        "--sss_source",
        default="deterministic_no_innovation",
        choices=["fixed_point", "sim_conditional", "deterministic_no_innovation"],
        help=(
            "Table source mode: deterministic_no_innovation (table0 default), "
            "sim_conditional (paper-author conditional moments), or fixed_point (diagnostic)."
        ),
    )
    ap.add_argument(
        "--use_selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer selected_runs.json over latest complete run (use --no-use_selected to ignore it).",
    )
    ap.add_argument(
        "--strict_selected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require selected_runs.json entries to exist and be complete (no fallback to latest run).",
    )
    ap.add_argument(
        "--weights_source",
        default="auto",
        choices=["auto", "canonical", "best", "last"],
        help="Which checkpoint to load from each run.",
    )
    ap.add_argument(
        "--strict_author_table2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require author by-regime simulation files (NT/SS/SEV/...) for sim_conditional mode (no fallback).",
    )
    # Backward compatibility with previous CLI.
    ap.add_argument("--no_selected", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    use_selected = bool(args.use_selected) and (not bool(args.no_selected))
    df = build_table0(
        args.artifacts_root,
        device=args.device,
        include_rules=args.include_rules,
        include_para=args.include_para,
        include_zlb=args.include_zlb,
        sss_source=args.sss_source,
        use_selected=use_selected,
        strict_selected=bool(args.strict_selected),
        weights_source=args.weights_source,
        strict_author_table2=bool(args.strict_author_table2),
    )
    print(df.to_string(index=False))
    path = save_table0_csv(df, args.artifacts_root)
    print("Saved:", path)

if __name__ == "__main__":
    main()
