from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch

from .config import ModelParams, PolicyName, TrainConfig
from .model_common import unpack_state, shock_laws_of_motion
from .transforms import decode_outputs
from .deqn import Trainer, residual_metrics, residual_metrics_by_regime


def _broadcast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.dim() <= 1:
        return x.view(-1)
    return x.view(-1, *([1] * (ref.dim() - 1)))


def _infer_policy_input_dim(net: torch.nn.Module) -> int | None:
    try:
        for p in net.parameters():
            if p.ndim == 2:
                return int(p.shape[1])
    except Exception:
        return None
    return None


def _params_with_net_dtype(params: ModelParams, net: torch.nn.Module) -> ModelParams:
    try:
        net_dt = next(net.parameters()).dtype
    except StopIteration:
        return params
    if net_dt == params.dtype:
        return params
    return replace(params, dtype=net_dt).to_torch()


@dataclass
class TrajectoryResidualCheckResult:
    n_states_evaluated: int
    tol: float
    metrics: Dict[str, float]


@dataclass
class FixedPointCheckResult:
    regime: int
    max_abs_state_diff: float


@dataclass
class ResidualCheckResult:
    regime: int
    max_abs_residual: float
    residuals: Dict[str, float]



def _state_from_policy_sss(
    params: ModelParams,
    policy: PolicyName,
    sss: Dict[str, float],
    regime: int,
    *,
    commitment_state_dim: int | None = None,
) -> torch.Tensor:
    dev, dt = params.device, params.dtype
    s = float(int(regime))

    if policy == "commitment":
        base = [
            float(sss.get("Delta_prev", sss["Delta"])),
            float(sss["logA"]),
            float(sss["loggtilde"]),
            float(sss["xi"]),
            s,
            float(sss.get("vartheta_prev", 0.0)),
            float(sss.get("varrho_prev", 0.0)),
        ]
        if (commitment_state_dim is not None and int(commitment_state_dim) >= 8) or ("c_prev" in sss):
            base.append(float(sss.get("c_prev", sss.get("c", 1.0))))
        return torch.tensor(base, device=dev, dtype=dt).view(1, -1)

    if policy == "commitment_zlb":
        return torch.tensor(
            [
                float(sss.get("Delta_prev", sss["Delta"])),
                float(sss["logA"]),
                float(sss["loggtilde"]),
                float(sss["xi"]),
                s,
                float(sss.get("vartheta_prev", 0.0)),
                float(sss.get("varrho_prev", 0.0)),
                float(sss.get("c_prev", sss.get("c", 1.0))),
                float(sss.get("i_nom_prev", sss.get("i_nom", 0.0))),
                float(sss.get("varphi_prev", sss.get("varphi", 0.0))),
            ],
            device=dev,
            dtype=dt,
        ).view(1, -1)

    if policy == "taylor_para":
        return torch.tensor(
            [
                float(sss.get("Delta_prev", sss["Delta"])),
                float(sss["logA"]),
                float(sss["loggtilde"]),
                float(sss["xi"]),
                s,
                float(sss.get("i_old", sss.get("i_nom", 0.0))),
                float(sss.get("p21", params.p21)),
            ],
            device=dev,
            dtype=dt,
        ).view(1, -1)

    return torch.tensor(
        [
            float(sss.get("Delta_prev", sss["Delta"])),
            float(sss["logA"]),
            float(sss["loggtilde"]),
            float(sss["xi"]),
            s,
        ],
        device=dev,
        dtype=dt,
    ).view(1, -1)



def _deterministic_next_state(
    params: ModelParams,
    policy: PolicyName,
    st,
    out: Dict[str, torch.Tensor],
    *,
    regime: int,
) -> torch.Tensor:
    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)

    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)

    if policy == "commitment":
        if st.c_prev is not None:
            return torch.stack(
                [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt), out["vartheta"].view(-1), out["varrho"].view(-1), out["c"].view(-1)],
                dim=-1,
            )
        return torch.stack(
            [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt), out["vartheta"].view(-1), out["varrho"].view(-1)],
            dim=-1,
        )

    if policy == "commitment_zlb":
        return torch.stack(
            [
                out["Delta"],
                logA_n.view(-1),
                logg_n.view(-1),
                xi_n.view(-1),
                s_n.to(dt),
                out["vartheta"].view(-1),
                out["varrho"].view(-1),
                out["c"].view(-1),
                out["i_nom"].view(-1),
                out["varphi"].view(-1),
            ],
            dim=-1,
        )

    if policy == "taylor_para":
        p21_prev = st.p21 if st.p21 is not None else torch.full_like(out["Delta"].view(-1), float(params.p21))
        return torch.stack(
            [
                out["Delta"],
                logA_n.view(-1),
                logg_n.view(-1),
                xi_n.view(-1),
                s_n.to(dt),
                out["i_nom"].view(-1),
                p21_prev.view(-1),
            ],
            dim=-1,
        )

    return torch.stack(
        [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt)],
        dim=-1,
    )



def fixed_point_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    rbar_by_regime: Optional[torch.Tensor] = None,
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, FixedPointCheckResult]:
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    params = _params_with_net_dtype(params, net)
    commit_dim = _infer_policy_input_dim(net) if policy == "commitment" else None

    out_by_regime: Dict[int, FixedPointCheckResult] = {}
    for r, sss in sss_by_regime.items():
        rr = int(r)
        x = _state_from_policy_sss(params, policy, sss, rr, commitment_state_dim=commit_dim)
        st = unpack_state(x, policy)
        out = decode_outputs(
            policy,
            net(x),
            floors=floors,
            params=params,
            st=st,
            rbar_by_regime=rbar_by_regime if policy in ("mod_taylor", "mod_taylor_zlb") else None,
        )
        x_next = _deterministic_next_state(params, policy, st, out, regime=rr)
        out_by_regime[rr] = FixedPointCheckResult(regime=rr, max_abs_state_diff=float((x_next - x).abs().max().item()))

    return out_by_regime



def residuals_check_switching_consistent(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    rbar_by_regime: Optional[torch.Tensor] = None,
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, ResidualCheckResult]:
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    params = _params_with_net_dtype(params, net)
    commit_dim = _infer_policy_input_dim(net) if policy == "commitment" else None

    cfg_sim = TrainConfig.author_like(policy=policy, seed=0)
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, rbar_by_regime=rbar_by_regime)

    results: Dict[int, ResidualCheckResult] = {}
    for r, sss in sss_by_regime.items():
        rr = int(r)
        x = _state_from_policy_sss(params, policy, sss, rr, commitment_state_dim=commit_dim)
        ctx = torch.enable_grad() if policy in ("discretion", "discretion_zlb") else torch.inference_mode()
        with ctx:
            resid_vec = trainer._residuals(x).view(-1)

        res_cpu = {k: float(v.detach().cpu().item()) for k, v in zip(trainer.res_keys, resid_vec)}
        max_abs = max(abs(v) for v in res_cpu.values()) if res_cpu else float("nan")
        results[rr] = ResidualCheckResult(regime=rr, max_abs_residual=float(max_abs), residuals=res_cpu)

    return results



def residuals_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    rbar_by_regime: Optional[torch.Tensor] = None,
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, ResidualCheckResult]:
    return residuals_check_switching_consistent(
        params,
        net,
        policy=policy,
        sss_by_regime=sss_by_regime,
        rbar_by_regime=rbar_by_regime,
        floors=floors,
    )



def trajectory_residuals_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sim_paths: Dict[str, "np.ndarray"],
    rbar_by_regime: Optional[torch.Tensor] = None,
    gh_n: int = 3,
    tol: float = 1e-3,
    max_states: int = 50_000,
    batch_size: int = 4096,
    seed: int = 0,
) -> TrajectoryResidualCheckResult:
    import numpy as np

    required = {"Delta", "logA", "loggtilde", "xi", "s"}
    missing = required.difference(sim_paths.keys())
    if missing:
        raise KeyError(
            "trajectory_residuals_check requires sim_paths with keys %s; missing %s. "
            "Run simulate_paths(..., store_states=True)." % (sorted(required), sorted(missing))
        )

    params = _params_with_net_dtype(params, net)
    dev, dt = params.device, params.dtype

    Delta = np.asarray(sim_paths["Delta"], dtype=np.float64)
    logA = np.asarray(sim_paths["logA"], dtype=np.float64)
    logg = np.asarray(sim_paths["loggtilde"], dtype=np.float64)
    xi = np.asarray(sim_paths["xi"], dtype=np.float64)
    s = np.asarray(sim_paths["s"], dtype=np.int64)

    if Delta.shape != logA.shape or Delta.shape != logg.shape or Delta.shape != xi.shape or Delta.shape != s.shape:
        raise ValueError("sim_paths arrays must have identical shapes for Delta/logA/loggtilde/xi/s")
    if Delta.shape[0] < 2:
        raise ValueError("Need at least 2 stored periods to build lagged states")

    Delta_prev = Delta[:-1]
    logA_c = logA[1:]
    logg_c = logg[1:]
    xi_c = xi[1:]
    s_c = s[1:]

    extra: Dict[str, np.ndarray] = {}
    if policy == "commitment":
        for kreq in ["vartheta_prev", "varrho_prev", "c"]:
            if kreq not in sim_paths:
                raise KeyError(
                    f"policy='commitment' requires sim_paths['{kreq}'] to reconstruct x; run simulate_paths(..., store_states=True)."
                )
        extra["vartheta_prev"] = np.asarray(sim_paths["vartheta_prev"], dtype=np.float64)[1:]
        extra["varrho_prev"] = np.asarray(sim_paths["varrho_prev"], dtype=np.float64)[1:]
        extra["c_prev"] = np.asarray(sim_paths["c"], dtype=np.float64)[:-1]
    elif policy == "commitment_zlb":
        for kreq in ["vartheta_prev", "varrho_prev", "c", "i_nom_prev", "varphi_prev"]:
            if kreq not in sim_paths:
                raise KeyError(
                    f"policy='commitment_zlb' requires sim_paths['{kreq}'] to reconstruct x; run simulate_paths(..., store_states=True)."
                )
        extra["vartheta_prev"] = np.asarray(sim_paths["vartheta_prev"], dtype=np.float64)[1:]
        extra["varrho_prev"] = np.asarray(sim_paths["varrho_prev"], dtype=np.float64)[1:]
        extra["c_prev"] = np.asarray(sim_paths["c"], dtype=np.float64)[:-1]
        extra["i_nom_prev"] = np.asarray(sim_paths["i_nom_prev"], dtype=np.float64)[1:]
        extra["varphi_prev"] = np.asarray(sim_paths["varphi_prev"], dtype=np.float64)[1:]

    K2, B = Delta_prev.shape
    rng = np.random.default_rng(int(seed))
    pool_n = K2 * B
    take = int(min(max_states, pool_n))
    flat_idx = rng.choice(pool_n, size=take, replace=False) if take < pool_n else np.arange(pool_n)
    t_idx = flat_idx // B
    b_idx = flat_idx % B

    if policy == "commitment":
        X = np.stack(
            [
                Delta_prev[t_idx, b_idx],
                logA_c[t_idx, b_idx],
                logg_c[t_idx, b_idx],
                xi_c[t_idx, b_idx],
                s_c[t_idx, b_idx].astype(np.float64),
                extra["vartheta_prev"][t_idx, b_idx],
                extra["varrho_prev"][t_idx, b_idx],
                extra["c_prev"][t_idx, b_idx],
            ],
            axis=1,
        )
    elif policy == "commitment_zlb":
        X = np.stack(
            [
                Delta_prev[t_idx, b_idx],
                logA_c[t_idx, b_idx],
                logg_c[t_idx, b_idx],
                xi_c[t_idx, b_idx],
                s_c[t_idx, b_idx].astype(np.float64),
                extra["vartheta_prev"][t_idx, b_idx],
                extra["varrho_prev"][t_idx, b_idx],
                extra["c_prev"][t_idx, b_idx],
                extra["i_nom_prev"][t_idx, b_idx],
                extra["varphi_prev"][t_idx, b_idx],
            ],
            axis=1,
        )
    else:
        X = np.stack(
            [
                Delta_prev[t_idx, b_idx],
                logA_c[t_idx, b_idx],
                logg_c[t_idx, b_idx],
                xi_c[t_idx, b_idx],
                s_c[t_idx, b_idx].astype(np.float64),
            ],
            axis=1,
        )

    x = torch.tensor(X, device=dev, dtype=dt)

    if params.device == "cpu":
        cfg_sim = TrainConfig.dev(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    else:
        cfg_sim = TrainConfig.full(seed=0, cpu_num_threads=None, cpu_num_interop_threads=None)
    trainer = Trainer(params=params, cfg=cfg_sim, policy=policy, net=net, gh_n=int(gh_n), rbar_by_regime=rbar_by_regime)

    resids = []
    ctx = torch.enable_grad() if policy in ("discretion", "discretion_zlb") else torch.inference_mode()
    with ctx:
        for i in range(0, x.shape[0], int(batch_size)):
            xb = x[i : i + int(batch_size)]
            rb = trainer._residuals(xb)
            resids.append(rb.detach())
    resid = torch.cat(resids, dim=0)

    keys = trainer.res_keys
    m = residual_metrics(resid, keys, tol=float(tol))
    m_reg = residual_metrics_by_regime(
        x,
        resid,
        keys,
        tol=float(tol),
        policy=policy,
        n_regimes=int(params.n_regimes),
    )
    m.update(m_reg)

    return TrajectoryResidualCheckResult(n_states_evaluated=int(x.shape[0]), tol=float(tol), metrics=m)
