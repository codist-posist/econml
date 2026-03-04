from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJECT_ROOT / "notebooks"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "_notebook_light_run_report.txt"
EXEC_DIR = PROJECT_ROOT / "artifacts" / "_notebook_light_exec"
EXEC_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

ORDER = [
    "00_main_pipeline.ipynb",
    "10_train_taylor.ipynb",
    "11_train_mod_taylor.ipynb",
    "12_train_discretion.ipynb",
    "13_train_commitment.ipynb",
    "14_train_taylor_zlb.ipynb",
    "15_train_mod_taylor_zlb.ipynb",
    "16_train_discretion_zlb.ipynb",
    "17_train_commitment_zlb.ipynb",
    "18_train_taylor_para.ipynb",
    "90_results_analysis.ipynb",
    "100_fig10_sensitivity_p21.ipynb",
    "smoke_test_minimal.ipynb",
]

SUBS: list[tuple[str, str]] = [
    (r"(\bT\s*=\s*)20000\b", r"\g<1>120"),
    (r"(\bburn_in\s*=\s*)2000\b", r"\g<1>20"),
    (r"(\bT_sim\s*=\s*)6000\b", r"\g<1>120"),
    (r"(\bburn_in_sim\s*=\s*)1000\b", r"\g<1>20"),
    (r"(\bB_sim\s*=\s*)2048\b", r"\g<1>64"),
    (r"simulate_initial_state\(\s*512\b", "simulate_initial_state(64"),
    (r"(\bgh_n\s*=\s*)7\b", r"\g<1>2"),
    (r"(\bgh_n\s*=\s*)3\b", r"\g<1>2"),
    (r"(\bthin\s*=\s*)10\b", r"\g<1>1"),
    (r"np\.linspace\(\s*0\.02\s*,\s*0\.98\s*,\s*40\s*\)", "np.linspace(0.02, 0.08, 8)"),
    (r"(\bT\s*=\s*)40\b", r"\g<1>20"),
]

NB_LITERAL_SUBS: dict[str, list[tuple[str, str]]] = {
    "11_train_mod_taylor.ipynb": [
        (
            "from src.sss_from_policy import mod_taylor_sss_by_regime_from_policy",
            "from src.sss_from_policy import switching_policy_sss_by_regime_from_policy",
        ),
        (
            "mt_sss_pol = mod_taylor_sss_by_regime_from_policy(params, trainer.net)",
            "mt_sss_pol = switching_policy_sss_by_regime_from_policy(params, trainer.net, policy='mod_taylor')",
        ),
    ],
}

PRELUDE = f"""
import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(r\"{PROJECT_ROOT}\")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use(\"Agg\")
except Exception:
    pass

from dataclasses import replace
from src.config import TrainConfig

_ORIG_DEV = TrainConfig.dev
_ORIG_MID = TrainConfig.mid
_ORIG_FULL = TrainConfig.full


def _lighten_cfg(cfg):
    p1 = cfg.phase1
    p2 = cfg.phase2
    return replace(
        cfg,
        n_path=min(int(cfg.n_path), 8),
        n_paths_per_step=1,
        val_size=min(int(cfg.val_size), 64),
        log_every=1,
        phase1=replace(
            p1,
            steps=min(int(p1.steps), 3),
            batch_size=min(int(p1.batch_size), 32),
            gh_n_train=min(int(p1.gh_n_train), 2),
            use_float64=False,
            eps_stop=None,
        ),
        phase2=replace(
            p2,
            steps=min(int(p2.steps), 1),
            batch_size=min(int(p2.batch_size), 32),
            gh_n_train=min(int(p2.gh_n_train), 2),
            use_float64=False,
            eps_stop=None,
        ),
    )


def _wrap(orig):
    def _f(**overrides):
        return _lighten_cfg(orig(**overrides))

    return _f


TrainConfig.dev = staticmethod(_wrap(_ORIG_DEV))
TrainConfig.mid = staticmethod(_wrap(_ORIG_MID))
TrainConfig.full = staticmethod(_wrap(_ORIG_FULL))
print(\"[light-check] TrainConfig presets patched to ultra-light.\")
""".strip()


def patch_notebook(nb: nbformat.NotebookNode, nb_name: str) -> tuple[nbformat.NotebookNode, int]:
    patched = nbformat.from_dict(nb)
    patched.cells.insert(0, nbformat.v4.new_code_cell(PRELUDE))

    n_subs = 0
    for cell in patched.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        for old, new in NB_LITERAL_SUBS.get(nb_name, []):
            c = src.count(old)
            if c:
                src = src.replace(old, new)
                n_subs += c
        for pat, repl in SUBS:
            src, c = re.subn(pat, repl, src)
            n_subs += c
        cell["source"] = src
    return patched, n_subs


def run_one(path: Path) -> tuple[str, float, int, str]:
    nb = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    nb, n_subs = patch_notebook(nb, path.name)

    start = time.time()
    status = "ok"
    err = ""

    client = NotebookClient(
        nb,
        timeout=7200,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )

    try:
        client.execute()
    except Exception:
        status = "fail"
        err = traceback.format_exc(limit=5)
    elapsed = time.time() - start

    out_nb = EXEC_DIR / path.name
    nbformat.write(nb, out_nb.open("w", encoding="utf-8"))

    return status, elapsed, n_subs, err


def main() -> int:
    requested = {arg.strip() for arg in sys.argv[1:] if arg.strip()}
    if requested:
        targets = [NB_DIR / name for name in ORDER if name in requested and (NB_DIR / name).exists()]
    else:
        targets = [NB_DIR / name for name in ORDER if (NB_DIR / name).exists()]
    if not targets:
        print("No notebooks found.")
        return 1

    lines: list[str] = []
    lines.append(f"project_root={PROJECT_ROOT}")
    lines.append(f"executed_notebooks_dir={NB_DIR}")
    lines.append(f"patched_outputs={EXEC_DIR}")

    ok = 0
    fail = 0

    for nb_path in targets:
        print(f"[run] {nb_path.name}")
        status, elapsed, n_subs, err = run_one(nb_path)
        if status == "ok":
            ok += 1
        else:
            fail += 1
        lines.append(f"{status}\t{nb_path.name}\t{elapsed:.1f}s\tpatches={n_subs}")
        if err:
            lines.append(err.strip())

    lines.append(f"summary\tok={ok}\tfail={fail}\ttotal={len(targets)}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] ok={ok}, fail={fail}, total={len(targets)}")
    print(f"report={REPORT_PATH}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
