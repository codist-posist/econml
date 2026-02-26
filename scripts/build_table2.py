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
    ap.add_argument("--include_rules", action="store_true", default=True)
    ap.add_argument("--no_selected", action="store_true")
    args = ap.parse_args()
    df = build_table2(args.artifacts_root, device=args.device, include_rules=args.include_rules, use_selected=not args.no_selected)
    print(df.to_string(index=False))
    path = save_table2_csv(df, args.artifacts_root)
    print("Saved:", path)

if __name__ == "__main__":
    main()
