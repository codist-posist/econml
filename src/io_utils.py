from __future__ import annotations
import os, json, sys, platform
from datetime import datetime
from uuid import uuid4
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, List

import torch
import numpy as np
import pandas as pd


def make_run_dir(artifacts_root: str, policy: str, *, tag: str = "", seed: int | None = None) -> str:
    """Create a unique, collision-resistant run directory.

    Structure:
      <artifacts_root>/runs/<policy>/<timestamp>_<tag>_seed<seed>_<uuid>
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = f"seed{seed}" if seed is not None else "seedNA"
    tag_part = (tag.strip().replace(" ", "_")[:32]) if tag else ""
    uid = uuid4().hex[:10]
    name = "_".join([p for p in [ts, tag_part, seed_part, uid] if p])
    run_dir = os.path.join(artifacts_root, "runs", policy, name)
    ensure_dir(run_dir)
    return run_dir


def save_run_metadata(run_dir: str, config: Dict[str, Any], *, notes: str | None = None) -> None:
    """Save configuration + environment snapshot for reproducibility."""
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "config.json"), config)
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
    }
    if notes:
        env["notes"] = notes
    save_json(os.path.join(run_dir, "environment.json"), env)


def _project_root() -> str:
    """Absolute path to the project root (directory containing 'src')."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_artifacts_root(artifacts_root: str) -> str:
    """Normalize artifacts_root in a way that is stable across working directories.

    Accept either '<...>/artifacts' or legacy '<...>/artifacts/runs'.
    Also accepts legacy relative defaults like '../artifacts' coming from notebooks/scripts;
    these are interpreted relative to the project root (not the current working directory).
    """
    if artifacts_root is None:
        raise ValueError("artifacts_root must be a non-empty string")
    # Expand env vars and ~
    p = os.path.expandvars(os.path.expanduser(str(artifacts_root)))

    # Interpret relative paths relative to project root (not CWD).
    if not os.path.isabs(p):
        # Backward compatibility: many entrypoints used '../artifacts' assuming they were run
        # from within project/notebooks or project/scripts. Anchor these paths inside the project.
        # NOTE: normalize Windows separators first; need a double backslash in the string literal.
        parts = [q for q in p.replace("\\", "/").split("/") if q not in ("", ".")]
        while parts and parts[0] == "..":
            parts = parts[1:]
        p = os.path.join(_project_root(), *parts)

    p = os.path.normpath(p)

    # Legacy '<...>/artifacts/runs' -> '<...>/artifacts'
    if os.path.basename(p) == "runs":
        p = os.path.dirname(p)

    return p


def find_latest_run_dir(artifacts_root: str, policy: str) -> str | None:
    """Return most recent run directory for a policy, or None if none exist."""
    artifacts_root = _normalize_artifacts_root(artifacts_root)
    root = os.path.join(artifacts_root, "runs", policy)
    if not os.path.isdir(root):
        return None
    cand = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not cand:
        return None
    return sorted(cand)[-1]


def list_run_dirs(artifacts_root: str, policy: str) -> List[str]:
    """Return run directories for a policy sorted by mtime (newest first)."""
    artifacts_root = _normalize_artifacts_root(artifacts_root)
    root = os.path.join(artifacts_root, "runs", policy)
    if not os.path.isdir(root):
        return []
    cand = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand


def _run_config_safe(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "config.json")
    if not os.path.exists(p):
        return {}
    try:
        return load_json(p)
    except Exception:
        return {}


def _infer_net_output_dim_from_weights(run_dir: str) -> int | None:
    for wname in ("weights.pt", "weights_best.pt"):
        wp = os.path.join(run_dir, wname)
        if not os.path.exists(wp):
            continue
        try:
            state = load_torch(wp, map_location="cpu")
            if isinstance(state, dict):
                w = state.get("net.4.weight")
                if isinstance(w, torch.Tensor) and w.ndim == 2:
                    return int(w.shape[0])
        except Exception:
            pass
    return None


def infer_mod_taylor_variant(run_dir: str) -> str | None:
    """Infer mod_taylor variant from metadata, with checkpoint-shape fallback."""
    cfg = _run_config_safe(run_dir)
    extra = cfg.get("extra", {}) if isinstance(cfg, dict) else {}
    v = extra.get("mod_taylor_variant")
    if v:
        return str(v)
    d_out = _infer_net_output_dim_from_weights(run_dir)
    if d_out == 9:
        return "author_repo_param_i"
    if d_out == 8:
        # Legacy local variant kept for backward compatibility with old artifacts.
        return "legacy_rule_rbar"
    return None


def is_paper_mod_taylor_run(run_dir: str) -> bool:
    """Backward-compatible helper for old artifacts."""
    v = infer_mod_taylor_variant(run_dir)
    return v == "legacy_rule_rbar"


def resolve_analysis_run_dir(
    artifacts_root: str,
    policy: str,
    *,
    prefer_selected: bool = True,
    require_paper_mod_taylor: bool = False,
    mod_taylor_variant: str | None = None,
) -> str | None:
    """
    Resolve the *correct* run directory for analysis/figures.

    For mod_taylor, project default is author_repo_param_i.
    """
    artifacts_root = _normalize_artifacts_root(artifacts_root)
    cands: List[str] = []
    if prefer_selected:
        sel = load_selected_run(artifacts_root, policy)
        if sel is not None:
            cands.append(sel)
    for rd in list_run_dirs(artifacts_root, policy):
        if rd not in cands:
            cands.append(rd)

    if not cands:
        return None

    effective_variant = mod_taylor_variant
    if policy == "mod_taylor" and effective_variant is None:
        effective_variant = "author_repo_param_i"

    for rd in cands:
        if policy == "mod_taylor" and effective_variant is not None:
            if infer_mod_taylor_variant(rd) != str(effective_variant):
                continue
        return rd
    return None


def save_selected_run(artifacts_root: str, policy: str, run_dir: str) -> str:
    """Persist a pointer to the run_dir to make results reproducible.

    This lets notebooks and scripts refer to a specific run (instead of "latest"),
    which is important when running multiple experiments in parallel.
    """
    artifacts_root = _normalize_artifacts_root(artifacts_root)
    path = os.path.join(artifacts_root, "selected_runs.json")
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            data = load_json(path)
        except Exception:
            data = {}
    # Store relative paths when possible so selected_runs.json is portable
    # across machines (e.g., when copying the artifacts folder).
    ar = os.path.abspath(_normalize_artifacts_root(artifacts_root))
    rd = os.path.abspath(str(run_dir))
    if rd.startswith(ar + os.sep):
        rd_store = os.path.relpath(rd, ar)
    else:
        rd_store = rd
    data[str(policy)] = rd_store
    save_json(path, data)
    return path


def load_selected_run(artifacts_root: str, policy: str) -> str | None:
    artifacts_root = _normalize_artifacts_root(artifacts_root)
    path = os.path.join(artifacts_root, "selected_runs.json")
    if not os.path.exists(path):
        return None
    try:
        data = load_json(path)
        val = data.get(str(policy))
        if not val:
            return None
        # Accept absolute paths (legacy) or relative paths (portable).
        if os.path.isabs(val):
            cand = val
        else:
            cand = os.path.join(artifacts_root, val)
        if os.path.isdir(cand):
            return os.path.normpath(str(cand))
    except Exception:
        return None
    return None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_torch(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)

def load_torch(path: str, map_location: str | None = None) -> Any:
    return torch.load(path, map_location=map_location)

def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)


def _json_default(o: Any) -> Any:
    """Best-effort JSON serializer.

    This is used only for metadata/config logging, not for model math.
    We keep it conservative: convert dtypes/devices/scalars to strings,
    numpy scalars to Python scalars, tensors/arrays to lightweight summaries.
    """
    # dataclasses
    if is_dataclass(o):
        return asdict(o)

    # torch
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, torch.device):
        return str(o)
    if isinstance(o, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(o.shape),
            "dtype": str(o.dtype),
            "device": str(o.device),
        }

    # numpy
    if isinstance(o, np.dtype):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}

    # containers
    if isinstance(o, (set, tuple)):
        return list(o)

    # fall back to string representation
    return str(o)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_npz(path: str, **arrays: Any) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)

def load_npz(path: str) -> Dict[str, Any]:
    return dict(np.load(path, allow_pickle=True))

def save_csv(path: str, df: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def pack_config(params: Any, cfg: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["params"] = asdict(params) if is_dataclass(params) else dict(params)
    out["train_cfg"] = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    if extra:
        out["extra"] = extra
    return out


def save_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
