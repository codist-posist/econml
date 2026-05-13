from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import BaselineParams, NATURAL_OUTPUT_NAMES, QMCConfig
from .economics import derive_natural, unpack_natural_state
from .qmc import QMCNodes, make_qmc_nodes
from .transitions import transition_natural_states
from .transforms import decode_natural_outputs, steady_decode_targets


TensorDict = Dict[str, torch.Tensor]


def _mean_over_nodes(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)


def _reshape_outputs(data: TensorDict, shape: torch.Size) -> TensorDict:
    return {key: value.reshape(shape) for key, value in data.items()}


def _atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, min=-1.0 + float(eps), max=1.0 - float(eps))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _encode_bounded_log_center(value: torch.Tensor, center: float, width: float) -> torch.Tensor:
    scaled = torch.log(torch.clamp(value, min=1e-30) / max(float(center), 1e-30)) / float(width)
    return _atanh_clamped(scaled)


def encode_natural_outputs(out: TensorDict, params: BaselineParams = BaselineParams()) -> torch.Tensor:
    """Encode natural oracle levels into raw outputs compatible with decode_natural_outputs."""

    targets = steady_decode_targets(params)
    raw = []
    for name in NATURAL_OUTPUT_NAMES:
        if name == "C_n":
            raw.append(_encode_bounded_log_center(out[name], targets["C"], torch.log(torch.as_tensor(3.0)).item()))
        elif name == "Y_n":
            raw.append(_encode_bounded_log_center(out[name], targets["Y"], 2.0))
        elif name == "R_n_real":
            raw.append(_encode_bounded_log_center(out[name], targets["R"], torch.log(torch.as_tensor(1.50)).item()))
        else:
            raise ValueError(f"Unsupported natural output: {name}")
    return torch.stack(raw, dim=-1)


class NaturalOracleNet(nn.Module):
    """Drop-in natural benchmark module backed by the numerical oracle.

    Forward returns raw outputs so legacy code that calls
    ``decode_natural_outputs(natural_net(z_n), ...)`` keeps working.  New code
    can call ``outputs(..., need_rate=False)`` to avoid computing the Euler
    natural rate when only ``C_n`` and ``Y_n`` are needed.
    """

    is_natural_oracle = True

    def __init__(
        self,
        *,
        params: BaselineParams = BaselineParams(),
        qmc_cfg: QMCConfig = QMCConfig(),
        n_nodes: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 8192,
        max_iter: int = 80,
        tol: float = 1e-10,
    ) -> None:
        super().__init__()
        self.params = params
        self.qmc_cfg = qmc_cfg
        self.chunk_size = int(chunk_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        nodes = make_qmc_nodes(int(n_nodes or qmc_cfg.n_train), cfg=qmc_cfg, device=device, dtype=dtype)
        self.register_buffer("_eps_z", nodes.eps_z)
        self.register_buffer("_eps_lam_D", nodes.eps_lam_D)
        self.register_buffer("_eps_lam_X", nodes.eps_lam_X)
        self.register_buffer("_u_N_D", nodes.u_N_D)
        self.register_buffer("_u_N_X", nodes.u_N_X)

    def _nodes(self) -> QMCNodes:
        return QMCNodes(
            eps_z=self._eps_z,
            eps_lam_D=self._eps_lam_D,
            eps_lam_X=self._eps_lam_X,
            u_N_D=self._u_N_D,
            u_N_X=self._u_N_X,
        )

    @torch.no_grad()
    def outputs(
        self,
        z_n: torch.Tensor,
        *,
        nodes: QMCNodes | None = None,
        qmc_cfg: QMCConfig | None = None,
        need_rate: bool = True,
    ) -> TensorDict:
        cfg = qmc_cfg or self.qmc_cfg
        if need_rate:
            out, _ = natural_oracle_outputs(
                z_n,
                nodes or self._nodes(),
                params=self.params,
                qmc_cfg=cfg,
                chunk_size=self.chunk_size,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            return out
        return _solve_static_chunks(
            z_n,
            params=self.params,
            chunk_size=self.chunk_size,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    @torch.no_grad()
    def forward(self, z_n: torch.Tensor) -> torch.Tensor:
        out = self.outputs(z_n, need_rate=True)
        return encode_natural_outputs(out, self.params)


def natural_benchmark_outputs(
    z_n: torch.Tensor,
    natural_net: nn.Module,
    *,
    params: BaselineParams = BaselineParams(),
    need_rate: bool = True,
    nodes: QMCNodes | None = None,
    qmc_cfg: QMCConfig | None = None,
) -> TensorDict:
    """Return natural benchmark outputs from either the oracle or legacy network."""

    if getattr(natural_net, "is_natural_oracle", False):
        out = natural_net.outputs(z_n, nodes=nodes, qmc_cfg=qmc_cfg, need_rate=need_rate)
        if "R_n_real" not in out:
            out = dict(out)
            out["R_n_real"] = torch.full_like(out["Y_n"], float(params.bar_R))
        return out
    return decode_natural_outputs(natural_net(z_n), NATURAL_OUTPUT_NAMES, params=params)


def _natural_static_residuals_from_logs(
    z_n: torch.Tensor,
    log_C: torch.Tensor,
    log_Y: torch.Tensor,
    params: BaselineParams,
) -> Tuple[torch.Tensor, TensorDict]:
    """Static flexible-price residuals for a candidate log(C_n), log(Y_n)."""

    C = torch.exp(log_C)
    Y = torch.exp(log_Y)
    out = {"C_n": C, "Y_n": Y, "R_n_real": torch.full_like(C, float(params.bar_R))}
    drv = derive_natural(unpack_natural_state(z_n), out, params)
    mc_flex = torch.full_like(C, (float(params.epsilon) - 1.0) / float(params.epsilon))
    mc_res = torch.log(torch.clamp(drv["mc"] / mc_flex, min=1e-30))
    resource_res = (Y - C - drv["pm"] * drv["M"] - float(params.p_d) * drv["S"]) / torch.clamp(Y, min=1e-30)
    return torch.stack([mc_res, resource_res], dim=-1), drv


@torch.no_grad()
def solve_natural_static(
    z_n: torch.Tensor,
    *,
    params: BaselineParams = BaselineParams(),
    max_iter: int = 80,
    tol: float = 1e-10,
    fd_step: float = 1e-4,
    max_newton_step: float = 0.75,
) -> Tuple[TensorDict, dict[str, float]]:
    """Solve the pointwise flexible-price static block without a neural net.

    The two unknowns are current natural consumption and output.  They are
    pinned down by marginal cost equal to the flexible-price markup target and
    by the natural resource constraint.  The solver uses batched damped Newton
    steps in log(C), log(Y), with finite-difference Jacobians so that the
    imported-input cap kink remains harmless.
    """

    if z_n.shape[-1] != 6:
        raise ValueError("z_n must have last dimension 6: D, X, ell_D, ell_X, log_Z, A.")
    orig_shape = z_n.shape[:-1]
    z_flat = z_n.reshape(-1, 6)
    targets = steady_decode_targets(params)
    log_C = torch.full((z_flat.shape[0],), float(targets["C"]), device=z_flat.device, dtype=z_flat.dtype).log()
    log_Y = torch.full((z_flat.shape[0],), float(targets["Y"]), device=z_flat.device, dtype=z_flat.dtype).log()
    log_floor = torch.as_tensor(-30.0, device=z_flat.device, dtype=z_flat.dtype)
    log_ceiling = torch.as_tensor(5.0, device=z_flat.device, dtype=z_flat.dtype)
    h = torch.as_tensor(float(fd_step), device=z_flat.device, dtype=z_flat.dtype)

    residual, drv = _natural_static_residuals_from_logs(z_flat, log_C, log_Y, params)
    norm = torch.linalg.vector_norm(residual, dim=-1)

    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        if bool((norm.max() < float(tol)).detach().cpu()):
            break

        res_C, _ = _natural_static_residuals_from_logs(z_flat, log_C + h, log_Y, params)
        res_Y, _ = _natural_static_residuals_from_logs(z_flat, log_C, log_Y + h, params)
        dC = (res_C - residual) / h
        dY = (res_Y - residual) / h
        j00 = dC[..., 0]
        j10 = dC[..., 1]
        j01 = dY[..., 0]
        j11 = dY[..., 1]
        r0 = residual[..., 0]
        r1 = residual[..., 1]
        det = j00 * j11 - j01 * j10
        safe_sign = torch.where(det >= 0.0, torch.ones_like(det), -torch.ones_like(det))
        det = torch.where(det.abs() > 1e-14, det, safe_sign * 1e-14)
        step_C = (r0 * j11 - j01 * r1) / det
        step_Y = (j00 * r1 - r0 * j10) / det
        step_C = torch.clamp(step_C, -float(max_newton_step), float(max_newton_step))
        step_Y = torch.clamp(step_Y, -float(max_newton_step), float(max_newton_step))

        best_log_C = log_C
        best_log_Y = log_Y
        best_residual = residual
        best_drv = drv
        best_norm = norm
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            cand_log_C = torch.clamp(log_C - float(scale) * step_C, min=log_floor, max=log_ceiling)
            cand_log_Y = torch.clamp(log_Y - float(scale) * step_Y, min=log_floor, max=log_ceiling)
            cand_residual, cand_drv = _natural_static_residuals_from_logs(z_flat, cand_log_C, cand_log_Y, params)
            cand_norm = torch.linalg.vector_norm(cand_residual, dim=-1)
            accept = torch.isfinite(cand_norm) & (cand_norm < best_norm)
            best_log_C = torch.where(accept, cand_log_C, best_log_C)
            best_log_Y = torch.where(accept, cand_log_Y, best_log_Y)
            best_residual = torch.where(accept[:, None], cand_residual, best_residual)
            best_norm = torch.where(accept, cand_norm, best_norm)
            best_drv = {
                key: torch.where(accept, cand_value, best_drv[key])
                for key, cand_value in cand_drv.items()
                if key in best_drv
            }

        improved = best_norm < norm
        log_C = best_log_C
        log_Y = best_log_Y
        residual = best_residual
        drv = best_drv
        norm = best_norm
        if not bool(improved.detach().any().cpu()):
            break

    C = torch.exp(log_C)
    Y = torch.exp(log_Y)
    out = _reshape_outputs(
        {
            "C_n": C,
            "Y_n": Y,
            "R_n_real": torch.full_like(C, float(params.bar_R)),
        },
        orig_shape,
    )
    info = {
        "static_rms": float(torch.sqrt(residual.pow(2).mean()).detach().cpu()),
        "static_max_abs": float(residual.abs().max().detach().cpu()),
        "static_iters": float(n_iter),
    }
    return out, info


def _solve_static_chunks(
    z_n: torch.Tensor,
    *,
    params: BaselineParams,
    chunk_size: int,
    max_iter: int,
    tol: float,
) -> TensorDict:
    z_flat = z_n.reshape(-1, 6)
    pieces: dict[str, list[torch.Tensor]] = {"C_n": [], "Y_n": [], "R_n_real": []}
    for start in range(0, z_flat.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), z_flat.shape[0])
        out, _ = solve_natural_static(z_flat[start:stop], params=params, max_iter=max_iter, tol=tol)
        for key in pieces:
            pieces[key].append(out[key].reshape(-1))
    return {key: torch.cat(value, dim=0).reshape(z_n.shape[:-1]) for key, value in pieces.items()}


@torch.no_grad()
def natural_oracle_outputs(
    z_n: torch.Tensor,
    nodes: QMCNodes,
    *,
    params: BaselineParams = BaselineParams(),
    qmc_cfg: QMCConfig = QMCConfig(),
    chunk_size: int = 8192,
    max_iter: int = 80,
    tol: float = 1e-10,
) -> Tuple[TensorDict, dict[str, float]]:
    """Numerical natural benchmark: static solve plus Euler-implied real rate."""

    current, info = solve_natural_static(z_n, params=params, max_iter=max_iter, tol=tol)
    z_flat = z_n.reshape(-1, 6)
    st = unpack_natural_state(z_flat)
    z_next = transition_natural_states(st, nodes, params, qmc_cfg)
    next_out = _solve_static_chunks(
        z_next.reshape(-1, 6),
        params=params,
        chunk_size=chunk_size,
        max_iter=max_iter,
        tol=tol,
    )
    B = z_flat.shape[0]
    S = nodes.n
    C = current["C_n"].reshape(B)
    C_next = next_out["C_n"].reshape(B, S)
    lambda_ratio = C_next.pow(-float(params.sigma)) / C[:, None].pow(-float(params.sigma))
    sdf = _mean_over_nodes(lambda_ratio)
    R_n = 1.0 / torch.clamp(float(params.beta) * sdf, min=1e-30)
    out = {
        "C_n": current["C_n"],
        "Y_n": current["Y_n"],
        "R_n_real": R_n.reshape(z_n.shape[:-1]),
    }
    info["euler_sdf_min"] = float(sdf.min().detach().cpu())
    info["euler_sdf_max"] = float(sdf.max().detach().cpu())
    return out, info


@torch.no_grad()
def natural_oracle_residuals(
    z_n: torch.Tensor,
    nodes: QMCNodes,
    *,
    params: BaselineParams = BaselineParams(),
    qmc_cfg: QMCConfig = QMCConfig(),
    chunk_size: int = 8192,
    max_iter: int = 80,
    tol: float = 1e-10,
) -> Tuple[TensorDict, TensorDict]:
    """Residual diagnostics for the numerical natural benchmark."""

    out, info = natural_oracle_outputs(
        z_n,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        chunk_size=chunk_size,
        max_iter=max_iter,
        tol=tol,
    )
    st = unpack_natural_state(z_n)
    drv = derive_natural(st, out, params)
    z_next = transition_natural_states(unpack_natural_state(z_n.reshape(-1, 6)), nodes, params, qmc_cfg)
    next_out = _solve_static_chunks(
        z_next.reshape(-1, 6),
        params=params,
        chunk_size=chunk_size,
        max_iter=max_iter,
        tol=tol,
    )
    B = z_n.reshape(-1, 6).shape[0]
    S = nodes.n
    C = out["C_n"].reshape(B)
    C_next = next_out["C_n"].reshape(B, S)
    lambda_ratio = C_next.pow(-float(params.sigma)) / C[:, None].pow(-float(params.sigma))
    mc_flex = torch.full_like(out["C_n"], (float(params.epsilon) - 1.0) / float(params.epsilon))
    res: TensorDict = {}
    res["n_mc"] = drv["mc"] / mc_flex - 1.0
    res["n_resource"] = (
        out["Y_n"] - out["C_n"] - drv["pm"] * drv["M"] - float(params.p_d) * drv["S"]
    ) / torch.clamp(out["Y_n"], min=1e-30)
    res["n_euler"] = torch.log(
        torch.clamp(float(params.beta) * out["R_n_real"] * _mean_over_nodes(lambda_ratio), min=1e-30)
    )
    return res, {**out, **drv, **{f"oracle_{key}": torch.as_tensor(value, device=out["C_n"].device, dtype=out["C_n"].dtype) for key, value in info.items()}}
