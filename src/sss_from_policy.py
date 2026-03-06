from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Dict, Optional

import torch

from .config import ModelParams, PolicyName
from .model_common import unpack_state, transition_probs_to_next_regimes
from .transforms import decode_outputs


@dataclass
class PolicySSS:
    """Stochastic steady state computed as a fixed point of a (trained) policy."""

    by_regime: Dict[int, Dict[str, float]]


def _uncond_mean_log_ar1(rho: float, sigma: float) -> float:
    """Mean of log state under the drift-corrected AR(1) used in shock_laws_of_motion."""
    # Appendix A.1 / author code: stationary mean is -sigma^2 / 2.
    # Keep rho in the signature for call-site compatibility.
    _ = rho
    return float(-(sigma**2) / 2.0)





# NOTE: Frozen-regime ("regime held fixed") SSS is *not* used in the paper's Table-2 objects,
# but we provide a diagnostic routine that computes it with the regime transition matrix set to identity.
# This is printed for debugging/comparison only and is never used in any downstream calculations.

def frozen_policy_sss_by_regime_from_policy(
    params: "ModelParams",
    net: "torch.nn.Module",
    *,
    policy: "PolicyName",
    **kwargs,
) -> "PolicySSS":
    """Diagnostic SSS with regimes held fixed (P = I).

    Returns an object with by_regime[s] computed under frozen regimes.
    """
    P_I = torch.eye(int(params.n_regimes), device=params.device, dtype=params.dtype)
    return switching_policy_sss_by_regime_from_policy(params, net, policy=policy, P_override=P_I, **kwargs)


def _stationary_dist(P: torch.Tensor) -> torch.Tensor:
    """Stationary distribution for row-stochastic Markov matrix P[current, next]."""
    R = int(P.shape[0])
    dev, dt = P.device, P.dtype
    pi = torch.full((R,), 1.0 / float(R), device=dev, dtype=dt)
    for _ in range(2048):
        nxt = pi @ P
        if torch.max(torch.abs(nxt - pi)) < 1e-12:
            pi = nxt
            break
        pi = nxt
    pi = torch.clamp(pi, min=0.0)
    s = torch.sum(pi)
    if float(s) <= 0.0:
        return torch.full((R,), 1.0 / float(R), device=dev, dtype=dt)
    return pi / s


def _backward_weights(P: torch.Tensor, pi_stat: torch.Tensor, curr_s: int) -> torch.Tensor:
    """w_prev[j] = Pr(s_{t-1}=j | s_t=curr_s)."""
    R = int(P.shape[0])
    idx = torch.arange(R, device=P.device, dtype=torch.long)
    w = P[idx, int(curr_s)] * pi_stat[idx]
    denom = w.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError(
            "Invalid Markov transition matrix: cannot form backward weights. "
            f"curr_s={curr_s}, denom={float(denom.item())}, P={P.detach().cpu().numpy().tolist()}, pi_stat={pi_stat.detach().cpu().numpy().tolist()}"
        )
    return w / denom


def _infer_policy_input_dim(net: torch.nn.Module) -> int | None:
    try:
        for p in net.parameters():
            if p.ndim == 2:
                return int(p.shape[1])
    except Exception:
        return None
    return None


@torch.no_grad()
def switching_policy_sss_by_regime_from_policy(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    rbar_by_regime: Optional[torch.Tensor] = None,
    author_commitment_zlb_p12: Optional[float] = None,
    P_override: Optional[torch.Tensor] = None,
    max_iter: int = 50_000,
    tol: float = 1e-12,
    damping: float = 0.5,
    floors: Optional[Dict[str, float]] = None,
    show_progress: bool = False,
    progress_every: int = 200,
) -> PolicySSS:
    """Switching-consistent SSS-by-regime as a fixed point of the trained policy.

    This matches the paper's Table-2 'SSS by regime' definition:
      - innovations set to zero
      - continuous shocks fixed at their drift-corrected stationary means
      - regime switching enters through expectations/backward weights via transition probabilities

    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    # Optional policy-specific override for commitment_zlb SSS analysis.
    if policy == "commitment_zlb" and author_commitment_zlb_p12 is not None:
        p12_target = float(author_commitment_zlb_p12)
        if abs(float(params.p12) - p12_target) > 1e-12:
            params = replace(params, p12=p12_target).to_torch()

    # Keep state tensors dtype-compatible with the policy network. After phase-2
    # training, the net can be float64 while caller-side params are still float32.
    try:
        net_dt = next(net.parameters()).dtype
    except StopIteration:
        net_dt = params.dtype

    dev, dt = params.device, net_dt
    P_fixed = P_override.to(device=dev, dtype=dt) if P_override is not None else None
    if (policy in ("mod_taylor", "mod_taylor_zlb")) and (rbar_by_regime is not None):
        rbar_by_regime = rbar_by_regime.to(device=dev, dtype=dt)
    R = int(params.n_regimes)
    if (policy in ("mod_taylor", "mod_taylor_zlb")) and (rbar_by_regime is not None) and (int(rbar_by_regime.numel()) != R):
        raise ValueError(f"rbar_by_regime must have {R} entries, got {int(rbar_by_regime.numel())}.")

    # Exogenous states fixed at unconditional means (drift-corrected AR(1))
    logA0 = _uncond_mean_log_ar1(float(params.rho_A), float(params.sigma_A))
    logg0 = _uncond_mean_log_ar1(float(params.rho_g), float(params.sigma_g))
    xi0 = _uncond_mean_log_ar1(float(params.rho_tau), float(params.sigma_tau))

    def _init_state(regime: int) -> torch.Tensor:
        s_val = float(regime)
        xi_ss = float(xi0)
        if policy == "commitment_zlb":
            vec = [1.0, logA0, logg0, xi_ss, s_val, 0.0, 0.0, 1.0, 0.002461, -0.000012]
            return torch.tensor(vec, device=dev, dtype=dt).view(1, -1)
        if policy == "taylor_para":
            i_nom_ss = (1.0 + float(params.pi_bar)) / float(params.beta) - 1.0
            p21_mid = 0.5 * (float(getattr(params, "p21_l", params.p21)) + float(getattr(params, "p21_u", params.p21)))
            vec = [1.0, logA0, logg0, xi_ss, s_val, i_nom_ss, p21_mid]
            return torch.tensor(vec, device=dev, dtype=dt).view(1, -1)
        if policy == "commitment":
            d_in = _infer_policy_input_dim(net) or 7
            if d_in >= 8:
                vec = [1.0, logA0, logg0, xi_ss, s_val, 0.0, 0.0, 1.0]
            else:
                vec = [1.0, logA0, logg0, xi_ss, s_val, 0.0, 0.0]
            return torch.tensor(vec, device=dev, dtype=dt).view(1, -1)
        vec = [1.0, logA0, logg0, xi_ss, s_val]
        return torch.tensor(vec, device=dev, dtype=dt).view(1, -1)

    x_by_regime: Dict[int, torch.Tensor] = {r: _init_state(r) for r in range(R)}

    def _decode(xr: torch.Tensor) -> Dict[str, torch.Tensor]:
        return decode_outputs(
            policy,
            net(xr),
            floors=floors,
            params=params,
            st=unpack_state(xr, policy),
            rbar_by_regime=rbar_by_regime if policy in ("mod_taylor", "mod_taylor_zlb") else None,
        )

    def _weighted_scalar(weights: torch.Tensor, vals: Dict[int, torch.Tensor]) -> torch.Tensor:
        acc = torch.zeros((), device=dev, dtype=dt)
        for j in range(R):
            acc = acc + weights[j] * vals[j]
        return acc

    def _effective_P_matrix(x_cur: Dict[int, torch.Tensor]) -> torch.Tensor:
        if P_fixed is not None:
            return P_fixed
        rows = []
        for r in range(R):
            st_r = unpack_state(x_cur[r], policy)
            probs_r = transition_probs_to_next_regimes(
                params,
                st_r.s,
                xi=st_r.xi,
                p21_state=getattr(st_r, "p21", None),
            ).view(-1, R)[0]
            rows.append(probs_r)
        return torch.stack(rows, dim=0)

    t0 = time.time()
    best_diff = float("inf")
    converged = False
    progress_step = max(1, int(progress_every))

    it = -1
    for it in range(int(max_iter)):
        P_eff = _effective_P_matrix(x_by_regime)
        pi_stat = _stationary_dist(P_eff)
        out_by_regime: Dict[int, Dict[str, torch.Tensor]] = {r: _decode(x_by_regime[r]) for r in range(R)}
        Delta_by_regime: Dict[int, torch.Tensor] = {r: out_by_regime[r]["Delta"].view(()) for r in range(R)}

        x_next_by_regime: Dict[int, torch.Tensor] = {}
        max_diff = 0.0
        for curr_s in range(R):
            wprev = _backward_weights(P_eff, pi_stat, curr_s)
            Delta_prev = _weighted_scalar(wprev, Delta_by_regime)
            s_val = float(curr_s)
            xi_ss = float(xi_by_reg[int(curr_s)] if int(curr_s) < len(xi_by_reg) else xi_by_reg[-1])

            if policy == "commitment_zlb":
                vartheta_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["vartheta"].view(()) for j in range(R)})
                varrho_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["varrho"].view(()) for j in range(R)})
                c_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["c"].view(()) for j in range(R)})
                i_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["i_nom"].view(()) for j in range(R)})
                varphi_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["varphi"].view(()) for j in range(R)})
                vec = [
                    float(Delta_prev.item()),
                    logA0,
                    logg0,
                    xi_ss,
                    s_val,
                    float(vartheta_prev.item()),
                    float(varrho_prev.item()),
                    float(c_prev.item()),
                    float(i_prev.item()),
                    float(varphi_prev.item()),
                ]
                x_next = torch.tensor(vec, device=dev, dtype=dt).view(1, -1)
            elif policy == "commitment":
                vartheta_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["vartheta"].view(()) for j in range(R)})
                varrho_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["varrho"].view(()) for j in range(R)})
                if int(x_by_regime[curr_s].shape[-1]) >= 8:
                    c_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["c"].view(()) for j in range(R)})
                    vec = [
                        float(Delta_prev.item()),
                        logA0,
                        logg0,
                        xi_ss,
                        s_val,
                        float(vartheta_prev.item()),
                        float(varrho_prev.item()),
                        float(c_prev.item()),
                    ]
                else:
                    vec = [
                        float(Delta_prev.item()),
                        logA0,
                        logg0,
                        xi_ss,
                        s_val,
                        float(vartheta_prev.item()),
                        float(varrho_prev.item()),
                    ]
                x_next = torch.tensor(vec, device=dev, dtype=dt).view(1, -1)
            else:
                vec = [float(Delta_prev.item()), logA0, logg0, xi_ss, s_val]
                if policy == "taylor_para":
                    i_prev = _weighted_scalar(wprev, {j: out_by_regime[j]["i_nom"].view(()) for j in range(R)})
                    p21_prev = (
                        float(x_by_regime[curr_s][0, 6].item())
                        if int(x_by_regime[curr_s].shape[-1]) >= 7
                        else float(params.p21)
                    )
                    vec.extend([float(i_prev.item()), p21_prev])
                x_next = torch.tensor(vec, device=dev, dtype=dt).view(1, -1)

            max_diff = max(max_diff, float((x_next - x_by_regime[curr_s]).abs().max().item()))
            x_next_by_regime[curr_s] = x_next

        for r in range(R):
            x_by_regime[r] = (1.0 - damping) * x_by_regime[r] + damping * x_next_by_regime[r]
        best_diff = min(best_diff, float(max_diff))
        if bool(show_progress) and ((it == 0) or ((it + 1) % progress_step == 0) or (max_diff < tol)):
            elapsed = time.time() - t0
            print(
                f"[sss:{policy}] iter={it+1}/{int(max_iter)} "
                f"max_diff={float(max_diff):.3e} best={float(best_diff):.3e} elapsed={elapsed:.1f}s"
            )
        if max_diff < tol:
            converged = True
            break

    # report
    out_by_regime: Dict[int, Dict[str, torch.Tensor]] = {r: _decode(x_by_regime[r]) for r in range(R)}

    def _pack_out(x: torch.Tensor, out: Dict[str, torch.Tensor]) -> Dict[str, float]:
        logA = float(x[0, 1].item())
        logg = float(x[0, 2].item())
        xi = float(x[0, 3].item())

        base = {
            "c": float(out["c"].item()),
            "pi": float(out["pi"].item()),
            "pstar": float(out["pstar"].item()),
            "lam": float(out["lam"].item()),
            "w": float(out["w"].item()),
            "XiN": float(out["XiN"].item()),
            "XiD": float(out["XiD"].item()),
            "Delta_prev": float(x[0, 0].item()),
            "Delta": float(out["Delta"].item()),
            "mu": float(out.get("mu", torch.tensor(0.0, device=dev, dtype=dt)).item()),
            "nu": float(out.get("nu", torch.tensor(0.0, device=dev, dtype=dt)).item()),
            "zeta": float(out.get("zeta", torch.tensor(0.0, device=dev, dtype=dt)).item()),
            "logA": logA,
            "loggtilde": logg,
            "xi": xi,
        }
        if policy in ("commitment", "commitment_zlb"):
            base.update({
                "vartheta": float(out["vartheta"].item()),
                "varrho": float(out["varrho"].item()),
                "vartheta_prev": float(x[0, 5].item()),
                "varrho_prev": float(x[0, 6].item()),
            })
            if int(x.shape[-1]) >= 8:
                base["c_prev"] = float(x[0, 7].item())
        if policy == "commitment_zlb":
            base["i_nom"] = float(out["i_nom"].item())
            base["varphi"] = float(out["varphi"].item())
            base["i_nom_prev"] = float(x[0, 8].item())
            base["varphi_prev"] = float(x[0, 9].item())
        if policy in ("taylor_para", "mod_taylor", "mod_taylor_zlb") and ("i_nom" in out):
            base["i_nom"] = float(out["i_nom"].item())
        if policy == "taylor_para" and int(x.shape[-1]) >= 7:
            base["i_old"] = float(x[0, 5].item())
            base["p21"] = float(x[0, 6].item())
        return base

    if bool(show_progress):
        elapsed = time.time() - t0
        status = "converged" if converged else "max_iter reached"
        print(
            f"[sss:{policy}] {status}; iterations={it+1}, "
            f"best_max_diff={float(best_diff):.3e}, elapsed={elapsed:.1f}s"
        )

    return PolicySSS(by_regime={r: _pack_out(x_by_regime[r], out_by_regime[r]) for r in range(R)})
