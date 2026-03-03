#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.notebook_checklist import build_notebook_run_checklist, save_notebook_run_checklist


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a unified checklist for 8 training notebooks and their run artifacts.")
    ap.add_argument("--artifacts_root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument(
        "--prefer_selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer selected_runs.json entries over latest run directories.",
    )
    args = ap.parse_args()

    df = build_notebook_run_checklist(
        args.artifacts_root,
        prefer_selected=bool(args.prefer_selected),
        project_root=ROOT,
    )
    print(df.to_string(index=False))

    csv_path, md_path = save_notebook_run_checklist(df, args.artifacts_root)
    print("Saved CSV:", csv_path)
    print("Saved MD :", md_path)
    return 0 if bool(df["ok"].all()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
