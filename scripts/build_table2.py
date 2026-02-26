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
from src.table2_builder import build_table2, save_table2_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", default=os.path.join(_ROOT, "artifacts"))
    ap.add_argument("--device", default="cpu")
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
        help="Prefer selected_runs.json over latest complete run (use --no-use_selected to ignore it).",
    )
    # Backward compatibility with previous CLI.
    ap.add_argument("--no_selected", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    use_selected = bool(args.use_selected) and (not bool(args.no_selected))
    df = build_table2(args.artifacts_root, device=args.device, include_rules=args.include_rules, use_selected=use_selected)
    print(df.to_string(index=False))
    path = save_table2_csv(df, args.artifacts_root)
    print("Saved:", path)

if __name__ == "__main__":
    main()
