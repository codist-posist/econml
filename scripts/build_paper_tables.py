#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.paper_tables import (
    build_table1_calibration,
    build_paper_tables_2_4,
    build_taylor_para_robustness_table,
)
from src.config import ModelParams


def main() -> int:
    ap = argparse.ArgumentParser(description="Build paper-style Table 1/2/3/4 from current artifacts.")
    ap.add_argument("--artifacts_root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--sss_source",
        default="sim_conditional",
        choices=["fixed_point", "sim_conditional", "deterministic_no_innovation"],
        help=(
            "Paper tables are built with sim_conditional; this flag is kept for "
            "backward compatibility and will be ignored if different."
        ),
    )
    ap.add_argument(
        "--use_selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer selected runs over latest runs.",
    )
    ap.add_argument(
        "--strict_selected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require selected runs to exist and be complete (no fallback).",
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
    ap.add_argument(
        "--include_taylor_para_robustness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save a robustness table comparing taylor / mod_taylor / taylor_para.",
    )
    args = ap.parse_args()

    os.makedirs(args.artifacts_root, exist_ok=True)

    p = ModelParams(device=args.device, dtype=torch.float64).to_torch()
    table1 = build_table1_calibration(p)
    table1_path = os.path.join(args.artifacts_root, "table1_calibration.csv")
    table1.to_csv(table1_path, index=False)
    print("Saved:", table1_path)

    tables = build_paper_tables_2_4(
        args.artifacts_root,
        device=args.device,
        dtype=torch.float64,
        use_selected=bool(args.use_selected),
        strict_selected=bool(args.strict_selected),
        weights_source=args.weights_source,
        sss_source=args.sss_source,
        strict_author_table2=bool(args.strict_author_table2),
    )

    for key, df in tables.items():
        out = os.path.join(args.artifacts_root, f"{key}_{args.sss_source}.csv")
        df.to_csv(out, index=False)
        print(f"\n{key.upper()}")
        print(df.to_string(index=False))
        print("Saved:", out)

    if bool(args.include_taylor_para_robustness):
        rob = build_taylor_para_robustness_table(
            args.artifacts_root,
            device=args.device,
            dtype=torch.float64,
            use_selected=bool(args.use_selected),
            strict_selected=bool(args.strict_selected),
            weights_source=args.weights_source,
            sss_source=args.sss_source,
            strict_author_table2=bool(args.strict_author_table2),
        )
        out = os.path.join(args.artifacts_root, f"table_taylor_para_compare_{args.sss_source}.csv")
        rob.to_csv(out, index=False)
        print("\nTAYLOR_PARA_ROBUSTNESS")
        print(rob.to_string(index=False))
        print("Saved:", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
