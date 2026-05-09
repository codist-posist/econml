from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .experiments import EXPERIMENTS, TABLE_EXPERIMENTS


def _unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


SUITES = {
    "baseline": ["baseline"],
    "core_irf": ["baseline", "price_only", "quantity_only"],
    "counterfactual": TABLE_EXPERIMENTS["counterfactual"],
    "sensitivity": TABLE_EXPERIMENTS["sensitivity"],
    "all_registered": list(EXPERIMENTS),
}


def experiments_for_suite(suite: str, explicit: str | None = None) -> list[str]:
    if explicit:
        names = [name.strip() for name in explicit.split(",") if name.strip()]
        missing = sorted(set(names) - set(EXPERIMENTS))
        if missing:
            raise ValueError(f"Unknown experiment(s): {missing}. Available: {sorted(EXPERIMENTS)}")
        return _unique(names)
    return _unique(SUITES[suite])


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a suite of critical-input DEQN experiments. Unknown trailing "
            "arguments are forwarded to run_experiment, so training dimensions "
            "can be controlled from one command."
        )
    )
    parser.add_argument("--suite", default="baseline", choices=sorted(SUITES))
    parser.add_argument("--experiments", default=None, help="Comma-separated explicit experiment list.")
    parser.add_argument("--base-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument(
        "--stage",
        default="all",
        choices=("natural", "fixed", "ba", "rules", "discretion", "commitment", "postprocess", "all"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--table-policy", default="ba", choices=("fixed", "ba", "discretion", "commitment"))
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    experiments = experiments_for_suite(args.suite, args.experiments)
    for experiment in experiments:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.critical_input_deqn.run_experiment",
            "--base-root",
            str(args.base_root),
            "--experiment",
            experiment,
            "--stage",
            args.stage,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(forwarded)
        _run(cmd, dry_run=args.dry_run)

    if args.stage in {"postprocess", "all"} and not args.skip_tables:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.critical_input_deqn.make_tables",
            "--base-root",
            str(args.base_root),
            "--output-dir",
            str(args.base_root / "tables"),
            "--policy",
            args.table_policy,
        ]
        _run(cmd, dry_run=args.dry_run)

    if args.stage in {"postprocess", "all"} and not args.skip_figures:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.critical_input_deqn.make_figures",
            "--base-root",
            str(args.base_root),
            "--output-dir",
            str(args.base_root / "figures"),
            "--policy",
            args.table_policy,
        ]
        _run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
