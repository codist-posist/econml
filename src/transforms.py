from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import ModelParams
from .model_common import State
from .policy_rules import i_taylor, i_taylor_zlb, i_modified_taylor, i_modified_taylor_zlb


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, min=-40.0, max=40.0))


def positive(x: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    return _safe_exp(x) + float(floor)


def bounded(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    lo = float(low)
    hi = float(high)
    if not (hi > lo):
        raise ValueError(f"Invalid bounds: low={lo}, high={hi}")
    return lo + (hi - lo) * torch.sigmoid(x)


def _pi_aux_from_pstar_aux(params: ModelParams, p_star_aux: torch.Tensor) -> torch.Tensor:
    eps = float(params.eps)
    theta = float(params.theta)
    # Author Definitions.py:
    # pi_aux = (-((-1 + p_star_aux^(1-1/eps) - p_star_aux^(1-1/eps)*theta)/theta))^(eps/(eps-1))
    a = torch.clamp(p_star_aux, min=1e-12).pow((eps - 1.0) / eps)
    inside = (1.0 - (1.0 - theta) * a) / theta
    inside = torch.clamp(inside, min=1e-12)
    return inside.pow(eps / (eps - 1.0))


def _pi_aux_from_pstar(params: ModelParams, p_star: torch.Tensor) -> torch.Tensor:
    eps = float(params.eps)
    theta = float(params.theta)
    a = torch.clamp(p_star, min=1e-12).pow(-(eps - 1.0))
    inside = (1.0 - (1.0 - theta) * a) / theta
    inside = torch.clamp(inside, min=1e-12)
    return inside.pow(eps / (eps - 1.0))


def _common_derived(
    params: ModelParams,
    st: State,
    *,
    c: torch.Tensor,
    XiN: torch.Tensor,
    XiD: torch.Tensor,
    p_star_aux: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    eps = float(params.eps)
    theta = float(params.theta)

    pi_aux = _pi_aux_from_pstar_aux(params, p_star_aux)
    one_plus_pi = torch.clamp(pi_aux, min=1e-12).pow(1.0 / eps)
    pi = one_plus_pi - 1.0

    Delta = theta * pi_aux * st.Delta_prev + (1.0 - theta) * p_star_aux
    Delta = torch.clamp(Delta, min=1e-12)

    c = torch.clamp(c, min=1e-12)
    XiN = torch.clamp(XiN, min=1e-12)
    XiD = torch.clamp(XiD, min=1e-12)

    pstar = torch.clamp(params.M * XiN / XiD, min=1e-12)
    lam = c.pow(-params.gamma)

    A = torch.exp(st.logA)
    g = params.g_bar * torch.exp(st.loggtilde)
    y = c + g
    h = y * Delta / torch.clamp(A, min=1e-12)
    w = h.pow(params.omega) / torch.clamp(lam, min=1e-12)

    out: Dict[str, torch.Tensor] = {
        "c": c,
        "XiN": XiN,
        "XiD": XiD,
        "p_star_aux": p_star_aux,
        "pi_aux": pi_aux,
        "one_plus_pi": one_plus_pi,
        "pi": pi,
        "Delta": Delta,
        "pstar": pstar,
        "lam": lam,
        "w": w,
        "A": A,
        "g": g,
        "y": y,
        "h": h,
    }
    return out


def decode_outputs(
    policy: str,
    raw: torch.Tensor,
    floors: Dict[str, float],
    *,
    params: Optional[ModelParams] = None,
    st: Optional[State] = None,
    rbar_by_regime: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # Legacy fallback: keep backward compatibility for utility call-sites that do
    # not provide state/params.
    if params is None or st is None:
        if policy in ("taylor", "taylor_zlb", "mod_taylor", "mod_taylor_zlb"):
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta"]
        elif policy == "discretion":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "rho", "zeta"]
        elif policy == "discretion_zlb":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "rho", "zeta", "varphi"]
        elif policy == "commitment":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "nu", "zeta", "vartheta", "varrho"]
        elif policy == "commitment_zlb":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "nu", "zeta", "vartheta", "varrho", "i_nom", "varphi"]
        else:
            raise ValueError(policy)

        if raw.shape[-1] != len(names):
            raise ValueError(f"decode_outputs legacy: raw last dim {raw.shape[-1]} != expected {len(names)} for policy={policy}")

        out = {n: raw[..., i] for i, n in enumerate(names)}
        _raw_pi = out["pi"]
        pi_low = floors.get("pi_low", None)
        pi_high = floors.get("pi_high", None)
        if (pi_low is not None) and (pi_high is not None):
            out["one_plus_pi"] = bounded(_raw_pi, 1.0 + float(pi_low), 1.0 + float(pi_high))
        else:
            pi_floor = floors.get("one_plus_pi", floors.get("pi", 1e-6))
            out["one_plus_pi"] = positive(_raw_pi, pi_floor)
        out["pi"] = out["one_plus_pi"] - 1.0
        out["c"] = positive(out["c"], floors.get("c", 1e-10))
        out["Delta"] = 1.0 + positive(out["Delta"], floors.get("Delta", 1e-10))
        out["pstar"] = positive(out["pstar"], floors.get("pstar", 1e-10))
        out["lam"] = positive(out["lam"], floors.get("lam", 1e-12))
        out["w"] = positive(out["w"], floors.get("w", 1e-12))
        out["XiN"] = positive(out["XiN"], floors.get("XiN", 1e-14))
        out["XiD"] = positive(out["XiD"], floors.get("XiD", 1e-14))
        return out

    # Strict author-style decoding: policy net outputs auxiliary objects and
    # economic variables are built in Definitions-style formulas.
    if policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
        if raw.shape[-1] != 4:
            raise ValueError(f"policy={policy} expects d_out=4, got {raw.shape[-1]}")
        num_raw, den_raw, pstar_aux_raw, cons_raw = [raw[..., i] for i in range(4)]
        XiN = num_raw
        XiD = den_raw
        c_ss = (1.0 / params.M) ** (1.0 / (params.omega + params.gamma))
        c = c_ss + cons_raw
        p_star_aux = 1.0 + pstar_aux_raw
        out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)

        if policy == "taylor":
            out["i_nom"] = i_taylor(params, out["pi"])
        elif policy == "taylor_zlb":
            out["i_nom"] = i_taylor_zlb(params, out["pi"], zlb_floor=0.0)
        elif policy == "mod_taylor":
            if rbar_by_regime is None:
                raise ValueError("policy=mod_taylor requires rbar_by_regime from flex-price SSS")
            out["i_nom"] = i_modified_taylor(params, out["pi"], rbar_by_regime, st.s)
        else:
            if rbar_by_regime is None:
                raise ValueError("policy=mod_taylor_zlb requires rbar_by_regime from flex-price SSS")
            out["i_nom"] = i_modified_taylor_zlb(params, out["pi"], rbar_by_regime, st.s, zlb_floor=0.0)

        out["i_rule_target"] = out["i_nom"]
        return out

    if policy == "discretion":
        if raw.shape[-1] != 5:
            raise ValueError(f"policy=discretion expects d_out=5 (author), got {raw.shape[-1]}")
        num_raw, den_raw, pstar_aux_raw, cons_raw, eta_raw = [raw[..., i] for i in range(5)]
        XiN = F.softplus(num_raw)
        XiD = F.softplus(den_raw)
        c = 1.0 + cons_raw
        p_star_aux = 1.0 + pstar_aux_raw
        out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)
        out["eta"] = eta_raw
        out["zeta"] = -(
            out["eta"] * (params.eps - 1.0)
        ) / (params.eps * torch.clamp(out["one_plus_pi"], min=1e-12) * torch.clamp(st.Delta_prev, min=1e-12))
        out["mu"] = torch.zeros_like(out["c"])
        return out

    if policy == "discretion_zlb":
        if raw.shape[-1] != 6:
            raise ValueError(f"policy=discretion_zlb expects d_out=6, got {raw.shape[-1]}")
        num_raw, den_raw, pstar_aux_raw, cons_raw, eta_raw, varphi_raw = [raw[..., i] for i in range(6)]
        XiN = F.softplus(num_raw)
        XiD = F.softplus(den_raw)
        c = 1.0 + cons_raw
        p_star_aux = 1.0 + pstar_aux_raw
        out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)
        out["eta"] = eta_raw
        out["varphi"] = -F.softplus(varphi_raw)
        out["zeta"] = -(
            out["eta"] * (params.eps - 1.0)
        ) / (params.eps * torch.clamp(out["one_plus_pi"], min=1e-12) * torch.clamp(st.Delta_prev, min=1e-12))
        out["mu"] = torch.zeros_like(out["c"])
        return out

    if policy == "commitment":
        if raw.shape[-1] != 5:
            raise ValueError(f"policy=commitment expects d_out=5 (author), got {raw.shape[-1]}")
        cons_raw, num_raw, den_raw, pstar_aux_raw, eta_raw = [raw[..., i] for i in range(5)]
        XiN = F.softplus(num_raw)
        XiD = F.softplus(den_raw)
        c = 1.0 + cons_raw
        p_star_aux = 1.0 + pstar_aux_raw
        out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)
        out["eta"] = eta_raw

        # commitment Definitions.py objects
        c_prev = st.c_prev if st.c_prev is not None else out["c"]
        vartheta_old = st.vartheta_prev if st.vartheta_prev is not None else torch.zeros_like(out["c"])
        varrho_old = st.varrho_prev if st.varrho_prev is not None else torch.zeros_like(out["c"])
        Delta_prev = torch.clamp(st.Delta_prev, min=1e-12)

        zeta_num = out["c"].pow(-params.gamma) * (
            vartheta_old * params.eps * out["pi_aux"] * out["XiN"] * c_prev.pow(params.gamma)
            - out["eta"] * out["c"].pow(params.gamma) * out["pi_aux"].pow((params.eps - 1.0) / params.eps)
            + params.eps * out["eta"] * out["c"].pow(params.gamma) * out["pi_aux"].pow((params.eps - 1.0) / params.eps)
            - varrho_old * out["XiD"] * c_prev.pow(params.gamma) * out["pi_aux"].pow((params.eps - 1.0) / params.eps)
            + params.eps * varrho_old * out["XiD"] * c_prev.pow(params.gamma) * out["pi_aux"].pow((params.eps - 1.0) / params.eps)
        )
        out["zeta"] = -(zeta_num) / (params.eps * out["pi_aux"] * Delta_prev)

        out["mu"] = (
            out["eta"] * (1.0 - params.theta) * (1.0 - params.eps) * out["p_star_aux"]
            + out["zeta"] * (1.0 - params.theta) * (-params.eps) * out["p_star_aux"].pow((params.eps + 1.0) / params.eps)
        ) / torch.clamp(out["XiD"], min=1e-12)

        out["vartheta"] = (
            params.M * out["mu"]
            + vartheta_old * params.theta * c_prev.pow(params.gamma) * out["c"].pow(-params.gamma) * out["pi_aux"]
        )
        out["varrho"] = (
            -out["mu"] * out["pstar"]
            + varrho_old * params.theta * c_prev.pow(params.gamma) * out["c"].pow(-params.gamma) * out["pi_aux"].pow((params.eps - 1.0) / params.eps)
        )
        out["nu"] = torch.zeros_like(out["c"])
        return out

    if policy == "commitment_zlb":
        if raw.shape[-1] != 7:
            raise ValueError(f"policy=commitment_zlb expects d_out=7, got {raw.shape[-1]}")
        cons_raw, num_raw, den_raw, i_nom_raw, pstar_aux_raw, eta_raw, varphi_raw = [raw[..., i] for i in range(7)]
        XiN = F.softplus(num_raw)
        XiD = F.softplus(den_raw)
        c = 1.0 + cons_raw
        p_star_aux = 1.0 + pstar_aux_raw
        out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)
        out["eta"] = eta_raw
        out["i_nom"] = F.softplus(i_nom_raw)
        out["varphi"] = -F.softplus(varphi_raw)

        c_prev = st.c_prev if st.c_prev is not None else out["c"]
        vartheta_old = st.vartheta_prev if st.vartheta_prev is not None else torch.zeros_like(out["c"])
        varrho_old = st.varrho_prev if st.varrho_prev is not None else torch.zeros_like(out["c"])
        i_nom_old = st.i_nom_prev if st.i_nom_prev is not None else torch.zeros_like(out["c"])
        varphi_old = st.varphi_prev if st.varphi_prev is not None else torch.zeros_like(out["c"])
        Delta_prev = torch.clamp(st.Delta_prev, min=1e-12)

        zeta_num = out["c"].pow(-params.gamma) * (
            vartheta_old * params.eps * out["pi_aux"].pow((params.eps + 1.0) / params.eps) * out["XiN"] * c_prev.pow(params.gamma)
            + (params.eps - 1.0) * out["eta"] * out["c"].pow(params.gamma) * out["pi_aux"]
            + (params.eps - 1.0) * varrho_old * out["XiD"] * c_prev.pow(params.gamma) * out["pi_aux"]
            - varphi_old * (1.0 + i_nom_old) / params.beta
        )
        out["zeta"] = -zeta_num / (
            params.eps * torch.clamp(out["pi_aux"].pow((params.eps + 1.0) / params.eps), min=1e-12) * Delta_prev
        )

        out["mu"] = (
            out["eta"] * (1.0 - params.theta) * (1.0 - params.eps) * out["p_star_aux"]
            + out["zeta"] * (1.0 - params.theta) * (-params.eps) * out["p_star_aux"].pow((params.eps + 1.0) / params.eps)
        ) / torch.clamp(out["XiD"], min=1e-12)

        out["vartheta"] = (
            params.M * out["mu"]
            + vartheta_old * params.theta * c_prev.pow(params.gamma) * out["c"].pow(-params.gamma) * out["pi_aux"]
        )
        out["varrho"] = (
            -out["mu"] * out["pstar"]
            + varrho_old
            * params.theta
            * c_prev.pow(params.gamma)
            * out["c"].pow(-params.gamma)
            * out["pi_aux"].pow((params.eps - 1.0) / params.eps)
        )
        out["nu"] = torch.zeros_like(out["c"])
        return out

    raise ValueError(policy)
