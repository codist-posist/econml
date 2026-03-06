#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams, TrainConfig
from src.critique_adaptation import (
    CritiqueAugmentedTrainer,
    CritiqueCurriculumConfig,
    RobustModTaylorRuleConfig,
)
from src.deqn import PolicyNetwork
from src.io_utils import make_run_dir, pack_config, save_csv, save_json, save_run_metadata, save_torch
from src.steady_states import export_rbar_tensor, solve_flexprice_sss
from scripts.build_paper_figures_author_strict import _parse_bool, _parse_dtype


def _make_cfg(
    *,
    artifacts_root: str,
    run_dir: str,
    seed: int,
    n_episodes: int | None,
    author_step_cap: int | None,
    log_every: int,
) -> TrainConfig:
    kwargs: Dict[str, object] = {
        "seed": int(seed),
        "artifacts_root": str(artifacts_root),
        "run_dir": str(run_dir),
        "log_every": int(log_every),
    }
    if n_episodes is not None and int(n_episodes) > 0:
        kwargs["n_episodes"] = int(n_episodes)
    if author_step_cap is not None and int(author_step_cap) > 0:
        kwargs["author_step_cap"] = int(author_step_cap)
    return TrainConfig.author_like(policy="mod_taylor", **kwargs)


def _train_variant(
    *,
    artifacts_root: str,
    device: str,
    dtype: torch.dtype,
    seed: int,
    variant: str,
    enable_robust_rule: bool,
    n_episodes: int | None,
    author_step_cap: int | None,
    log_every: int,
    curriculum: CritiqueCurriculumConfig,
    robust_rule: RobustModTaylorRuleConfig,
) -> Dict[str, object]:
    params = ModelParams(device=device, dtype=dtype).to_torch()

    probe = TrainConfig.author_like(policy="mod_taylor", seed=int(seed))
    run_tag = f"{probe.mode}_{variant}"
    run_dir = make_run_dir(artifacts_root, "mod_taylor", tag=run_tag, seed=int(seed))
    cfg = _make_cfg(
        artifacts_root=artifacts_root,
        run_dir=run_dir,
        seed=seed,
        n_episodes=n_episodes,
        author_step_cap=author_step_cap,
        log_every=log_every,
    )

    flex = solve_flexprice_sss(params)
    rbar_by_regime = export_rbar_tensor(params, flex)

    save_run_metadata(
        run_dir,
        pack_config(
            params,
            cfg,
            extra={
                "policy": "mod_taylor",
                "critique_variant": str(variant),
                "curriculum_enabled": bool(curriculum.enabled),
                "robust_rule_enabled": bool(enable_robust_rule),
            },
        ),
    )

    net = PolicyNetwork(
        5,
        4,
        hidden=cfg.hidden_layers,
        activation=cfg.activation,
        init_mode=cfg.init_mode,
        init_scale=cfg.init_scale,
        seed=cfg.seed,
    )

    trainer = CritiqueAugmentedTrainer(
        params=params,
        cfg=cfg,
        policy="mod_taylor",
        net=net,
        rbar_by_regime=rbar_by_regime,
        curriculum=curriculum,
        robust_rule=(robust_rule if enable_robust_rule else replace(robust_rule, enabled=False)),
    )

    losses = trainer.train(
        commitment_sss=None,
        n_path=cfg.n_path,
        n_paths_per_step=cfg.n_paths_per_step,
    )

    save_torch(os.path.join(run_dir, "weights.pt"), trainer.net.state_dict())
    save_csv(
        os.path.join(run_dir, "train_log.csv"),
        pd.DataFrame({"iter": np.arange(len(losses), dtype=np.int64), "loss": np.asarray(losses, dtype=np.float64)}),
    )

    crit_meta = {
        "variant": str(variant),
        "run_dir": str(run_dir),
        "policy": "mod_taylor",
        "curriculum": {
            "enabled": bool(curriculum.enabled),
            "bad_state": int(curriculum.bad_state),
            "ramp_episodes": int(curriculum.ramp_episodes),
            "shock_times": [int(v) for v in curriculum.shock_times],
            "shock_size": float(curriculum.shock_size),
        },
        "robust_rule": {
            "enabled": bool(enable_robust_rule),
            "bad_state": int(robust_rule.bad_state),
            "kappa": float(robust_rule.kappa),
            "uncertainty_ref": float(robust_rule.uncertainty_ref),
            "max_premium": float(robust_rule.max_premium),
            "bad_only": bool(robust_rule.bad_only),
        },
        "n_losses": int(len(losses)),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "min_loss": float(np.min(losses)) if losses else float("nan"),
    }
    save_json(os.path.join(run_dir, "critique_variant_meta.json"), crit_meta)
    print(f"[ok] {variant}: {run_dir}")
    return crit_meta


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train mod_taylor critique-aware variants: curriculum baseline and robust-rule variant."
    )
    ap.add_argument("--artifacts-root", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=151082)

    ap.add_argument("--train-baseline", type=str, default="true")
    ap.add_argument("--train-robust", type=str, default="true")
    ap.add_argument("--n-episodes", type=int, default=0, help="0 => keep TrainConfig.author_like default")
    ap.add_argument("--author-step-cap", type=int, default=0, help="0 => no explicit cap override")
    ap.add_argument("--log-every", type=int, default=100)

    ap.add_argument("--curr-ramp-episodes", type=int, default=2500)
    ap.add_argument("--curr-shock-times", type=str, default="0,4,8,12")
    ap.add_argument("--curr-shock-size", type=float, default=1.25)
    ap.add_argument(
        "--curr-bad-state",
        type=int,
        default=1,
        help="Stress-state threshold for curriculum shocks (applies to regimes s >= this index).",
    )

    ap.add_argument("--robust-kappa", type=float, default=0.02)
    ap.add_argument("--robust-uncert-ref", type=float, default=0.0)
    ap.add_argument("--robust-max-premium", type=float, default=0.03)
    ap.add_argument("--robust-bad-only", type=str, default="true")
    ap.add_argument(
        "--robust-bad-state",
        type=int,
        default=1,
        help="Stress-state threshold for robust premium (applies to regimes s >= this index).",
    )

    ap.add_argument(
        "--save-runs-json",
        default=os.path.join(ROOT, "artifacts", "critique_mod_taylor_variants", "latest_runs.json"),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    os.makedirs(os.path.dirname(str(args.save_runs_json)), exist_ok=True)

    curriculum = CritiqueCurriculumConfig(
        enabled=True,
        bad_state=int(args.curr_bad_state),
        ramp_episodes=max(1, int(args.curr_ramp_episodes)),
        shock_times=tuple(int(v.strip()) for v in str(args.curr_shock_times).split(",") if v.strip()),
        shock_size=float(args.curr_shock_size),
    )
    robust_rule = RobustModTaylorRuleConfig(
        enabled=True,
        bad_state=int(args.robust_bad_state),
        kappa=float(args.robust_kappa),
        uncertainty_ref=float(args.robust_uncert_ref),
        max_premium=float(args.robust_max_premium),
        bad_only=bool(_parse_bool(args.robust_bad_only)),
    )

    out: Dict[str, Dict[str, object]] = {}
    n_episodes = None if int(args.n_episodes) <= 0 else int(args.n_episodes)
    author_step_cap = None if int(args.author_step_cap) <= 0 else int(args.author_step_cap)

    if _parse_bool(args.train_baseline):
        out["baseline_curriculum"] = _train_variant(
            artifacts_root=str(args.artifacts_root),
            device=str(args.device),
            dtype=_parse_dtype(str(args.dtype)),
            seed=int(args.seed),
            variant="baseline_curriculum",
            enable_robust_rule=False,
            n_episodes=n_episodes,
            author_step_cap=author_step_cap,
            log_every=int(args.log_every),
            curriculum=curriculum,
            robust_rule=robust_rule,
        )

    if _parse_bool(args.train_robust):
        out["robust_curriculum"] = _train_variant(
            artifacts_root=str(args.artifacts_root),
            device=str(args.device),
            dtype=_parse_dtype(str(args.dtype)),
            seed=int(args.seed),
            variant="robust_curriculum",
            enable_robust_rule=True,
            n_episodes=n_episodes,
            author_step_cap=author_step_cap,
            log_every=int(args.log_every),
            curriculum=curriculum,
            robust_rule=robust_rule,
        )

    save_json(str(args.save_runs_json), out)
    print("[saved]", str(args.save_runs_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
