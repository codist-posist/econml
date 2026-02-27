#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


_ROOT = Path(__file__).resolve().parents[1]
_NB_DIR = _ROOT / "notebooks"


def _slug_multiplier(x: float) -> str:
    return f"{float(x):g}"


def _scenario_artifacts_root(base_root: Path, scenario: str, bad_multiplier: float) -> Path:
    s = scenario.strip().upper()
    if s == "BASELINE":
        return base_root
    if s in {"A", "B"}:
        return base_root / "critique" / f"{s}_bm_{_slug_multiplier(bad_multiplier)}"
    raise ValueError(f"Unknown scenario: {scenario}. Expected baseline/A/B.")


def _run(cmd: List[str], *, env: Dict[str, str], dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def _run_nb(nb_name: str, *, env: Dict[str, str], dry_run: bool) -> int:
    nb_path = _NB_DIR / nb_name
    if not nb_path.exists():
        print(f"[skip] missing notebook: {nb_path}")
        return 0
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(nb_path),
    ]
    return _run(cmd, env=env, dry_run=dry_run)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Execute figure notebooks for baseline/A/B without manual env switching, "
            "then build A-vs-B comparison outputs."
        )
    )
    ap.add_argument("--base_artifacts_root", default=str(_ROOT / "artifacts"))
    ap.add_argument("--scenarios", default="baseline,A,B", help="Comma-separated: baseline,A,B")
    ap.add_argument("--bad_multiplier", type=float, default=2.0)
    ap.add_argument("--normal_multiplier", type=float, default=1.0)
    ap.add_argument(
        "--figure_notebooks",
        default="90_results_analysis.ipynb,91_fig1_ergodic_distributions.ipynb,92_fig2_transition_commitment_vs_discretion.ipynb,93_fig3_persistent_vs_temporary.ipynb,94_fig4_persistence_sensitivity.ipynb,96_fig6_asymmetry.ipynb,99_fig9_taylor_vs_modtaylor.ipynb,100_fig10_sensitivity_p21.ipynb",
    )
    ap.add_argument("--run_compare", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--run_new_figures", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--critique_sweep_root", default=str(_ROOT / "artifacts" / "critique_discretion"))
    ap.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    base_root = Path(args.base_artifacts_root).resolve()
    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    nbs = [s.strip() for s in str(args.figure_notebooks).split(",") if s.strip()]

    env_base = os.environ.copy()
    scenario_roots: Dict[str, Path] = {}
    failures: List[str] = []

    for sc in scenarios:
        root = _scenario_artifacts_root(base_root, sc, float(args.bad_multiplier))
        scenario_roots[sc.upper()] = root
        env = dict(env_base)
        env["ARTIFACTS_ROOT"] = str(root)
        print(f"\n=== Scenario {sc.upper()} | ARTIFACTS_ROOT={root} ===")
        for nb in nbs:
            rc = _run_nb(nb, env=env, dry_run=bool(args.dry_run))
            if rc != 0:
                failures.append(f"{sc.upper()}::{nb}")
                break

    if args.run_compare and {"A", "B"}.issubset(set(scenario_roots.keys())):
        print("\n=== Build A-vs-B comparison ===")
        compare_out = base_root / "comparisons"
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "compare_uncertainty_variants.py"),
            "--artifacts_a",
            str(scenario_roots["A"]),
            "--artifacts_b",
            str(scenario_roots["B"]),
            "--bad_multiplier",
            str(float(args.bad_multiplier)),
            "--normal_multiplier",
            str(float(args.normal_multiplier)),
            "--out_dir",
            str(compare_out),
        ]
        if "BASELINE" in scenario_roots:
            cmd.extend(["--artifacts_baseline", str(scenario_roots["BASELINE"])])
        rc = _run(cmd, env=env_base, dry_run=bool(args.dry_run))
        if rc != 0:
            failures.append("compare_uncertainty_variants.py")

        if args.run_new_figures:
            env_new = dict(env_base)
            env_new["CRITIQUE_COMPARE_DIR"] = str(compare_out)
            rc = _run_nb("101_fig11_critique_variants.ipynb", env=env_new, dry_run=bool(args.dry_run))
            if rc != 0:
                failures.append("101_fig11_critique_variants.ipynb")

            sweep_root = Path(args.critique_sweep_root).resolve()
            env_new["CRITIQUE_SWEEP_ROOT"] = str(sweep_root)
            if (sweep_root / "discretion_sweep_summary.csv").exists() or bool(args.dry_run):
                rc = _run_nb("102_fig12_critique_sensitivity.ipynb", env=env_new, dry_run=bool(args.dry_run))
                if rc != 0:
                    failures.append("102_fig12_critique_sensitivity.ipynb")
            else:
                print(f"[skip] 102_fig12_critique_sensitivity.ipynb (missing {sweep_root / 'discretion_sweep_summary.csv'})")

    if failures:
        print("\nFAILED:")
        for x in failures:
            print(" -", x)
        return 2

    print("\nAll requested figure runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

