from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROBUSTNESS_SPECS: dict[str, list[str]] = {
    "seed_321": ["--seed", "321"],
    "seed_777": ["--seed", "777"],
    "width_128": ["--hidden-width", "128"],
    "width_256": ["--hidden-width", "256"],
    "qmc_256": ["--qmc-train", "256", "--qmc-val", "2048"],
    "qmc_1024": ["--qmc-train", "1024", "--qmc-val", "8192"],
}


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run numerical robustness checks for the baseline economy. These "
            "checks change numerical approximation settings, not model primitives."
        )
    )
    parser.add_argument("--base-root", type=Path, default=Path("baseline_artifacts/critical_input_deqn"))
    parser.add_argument("--specs", default=",".join(ROBUSTNESS_SPECS), help="Comma-separated robustness specs.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=("natural", "fixed", "ba", "rules", "discretion", "commitment", "postprocess", "all"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    names = [name.strip() for name in args.specs.split(",") if name.strip()]
    missing = sorted(set(names) - set(ROBUSTNESS_SPECS))
    if missing:
        raise ValueError(f"Unknown robustness spec(s): {missing}. Available: {sorted(ROBUSTNESS_SPECS)}")

    for name in names:
        root = args.base_root / "numerical_robustness" / name
        cmd = [
            sys.executable,
            "-m",
            "src.critical_input_deqn.run_experiment",
            "--base-root",
            str(root),
            "--experiment",
            "baseline",
            "--stage",
            args.stage,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(ROBUSTNESS_SPECS[name])
        cmd.extend(forwarded)
        _run(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
