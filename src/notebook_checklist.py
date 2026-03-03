from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from .io_utils import find_latest_run_dir, load_selected_run


TRAIN_NOTEBOOK_POLICIES: List[Tuple[str, str]] = [
    ("10_train_taylor.ipynb", "taylor"),
    ("11_train_mod_taylor.ipynb", "mod_taylor"),
    ("12_train_discretion.ipynb", "discretion"),
    ("13_train_commitment.ipynb", "commitment"),
    ("14_train_taylor_zlb.ipynb", "taylor_zlb"),
    ("15_train_mod_taylor_zlb.ipynb", "mod_taylor_zlb"),
    ("16_train_discretion_zlb.ipynb", "discretion_zlb"),
    ("17_train_commitment_zlb.ipynb", "commitment_zlb"),
]


REQUIRED_RUN_FILES: List[str] = [
    "config.json",
    "train_log.csv",
    "train_metrics.csv",
    "train_quality.json",
    "sss_policy_fixed_point.json",
    "sanity_checks.json",
    "sim_paths.npz",
]


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_weights(run_dir: str) -> bool:
    if not os.path.isdir(run_dir):
        return False
    return os.path.exists(os.path.join(run_dir, "weights.pt")) or os.path.exists(os.path.join(run_dir, "weights_best.pt"))


def _select_run_dir(artifacts_root: str, policy: str, prefer_selected: bool) -> Tuple[str | None, str]:
    selected = load_selected_run(artifacts_root, policy)
    latest = find_latest_run_dir(artifacts_root, policy)
    if prefer_selected and selected is not None:
        return selected, "selected"
    if latest is not None:
        return latest, "latest"
    if selected is not None:
        return selected, "selected"
    return None, "none"


def _missing_required(run_dir: str) -> List[str]:
    if not os.path.isdir(run_dir):
        return REQUIRED_RUN_FILES + ["weights.pt|weights_best.pt"]
    miss = [name for name in REQUIRED_RUN_FILES if not os.path.exists(os.path.join(run_dir, name))]
    if not _has_weights(run_dir):
        miss.append("weights.pt|weights_best.pt")
    return miss


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def build_notebook_run_checklist(
    artifacts_root: str,
    *,
    prefer_selected: bool = True,
    project_root: str | None = None,
) -> pd.DataFrame:
    root = os.path.abspath(project_root or _project_root())
    rows: List[Dict[str, object]] = []

    for nb_name, policy in TRAIN_NOTEBOOK_POLICIES:
        nb_path = os.path.join(root, "notebooks", nb_name)
        notebook_exists = os.path.exists(nb_path)

        run_dir, run_source = _select_run_dir(artifacts_root, policy, prefer_selected=prefer_selected)
        run_exists = bool(run_dir and os.path.isdir(run_dir))
        missing = _missing_required(run_dir) if run_dir is not None else (REQUIRED_RUN_FILES + ["weights.pt|weights_best.pt"])

        run_mtime = None
        if run_exists and run_dir is not None:
            try:
                run_mtime = os.path.getmtime(run_dir)
            except Exception:
                run_mtime = None

        ok = bool(notebook_exists and run_exists and len(missing) == 0)
        rows.append(
            {
                "policy": policy,
                "notebook": nb_name,
                "notebook_exists": notebook_exists,
                "run_source": run_source,
                "run_dir": run_dir,
                "run_exists": run_exists,
                "missing_required": "; ".join(missing),
                "last_modified": _fmt_time(run_mtime),
                "ok": ok,
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["ok", "policy"], ascending=[True, True]).reset_index(drop=True)
    return df


def _escape_md_cell(v: object) -> str:
    s = "" if v is None else str(v)
    return s.replace("|", "\\|")


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_escape_md_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_notebook_run_checklist(
    df: pd.DataFrame,
    artifacts_root: str,
    *,
    csv_name: str = "train8_run_checklist.csv",
    md_name: str = "train8_run_checklist.md",
) -> Tuple[str, str]:
    os.makedirs(artifacts_root, exist_ok=True)
    csv_path = os.path.join(artifacts_root, csv_name)
    md_path = os.path.join(artifacts_root, md_name)

    df.to_csv(csv_path, index=False)

    total = int(len(df))
    ok_count = int(df["ok"].sum()) if total else 0
    failed = df[df["ok"] == False]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Train Notebooks Checklist\n\n")
        f.write(f"- total: {total}\n")
        f.write(f"- ok: {ok_count}\n")
        f.write(f"- failed: {total - ok_count}\n\n")
        if total:
            f.write(_to_markdown_table(df))
            f.write("\n")
        if len(failed):
            f.write("\n## Missing Or Incomplete\n\n")
            f.write(_to_markdown_table(failed))
            f.write("\n")

    return csv_path, md_path
