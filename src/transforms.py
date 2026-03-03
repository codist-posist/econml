from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import ModelParams
from .model_common import State
from .policy_rules import i_modified_taylor, i_modified_taylor_zlb


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
    mod_taylor_variant: Optional[str] = None,
    rbar_by_regime: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # Legacy fallback: keep backward compatibility for utility call-sites that do
    # not provide state/params.
    if params is None or st is None:
        if policy == "taylor":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta"]
        elif policy == "mod_taylor":
            if raw.shape[-1] == 8:
                names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta"]
            elif raw.shape[-1] == 9:
                names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "i_nom"]
            else:
                raise ValueError(f"decode_outputs legacy mod_taylor raw dim {raw.shape[-1]} unsupported")
        elif policy == "discretion":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "rho", "zeta"]
        elif policy == "commitment":
            names = ["c", "pi", "pstar", "lam", "w", "XiN", "XiD", "Delta", "mu", "nu", "zeta", "vartheta", "varrho"]
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
    if policy == "taylor":
        if raw.shape[-1] != 4:
            raise ValueError(f"policy=taylor expects d_out=4 (author), got {raw.shape[-1]}")
        num_raw, den_raw, pstar_aux_raw, cons_raw = [raw[..., i] for i in range(4)]
        XiN = num_raw
        XiD = den_raw
        c_ss = (1.0 / params.M) ** (1.0 / (params.omega + params.gamma))
        c = c_ss + cons_raw
        p_star_aux = 1.0 + pstar_aux_raw
        out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)
        out["i_nom"] = (1.0 + params.pi_bar) / params.beta - 1.0 + params.psi * (out["pi"] - params.pi_bar)
        out["i_rule_target"] = out["i_nom"]
        return out

    if policy == "mod_taylor":
        variant = (mod_taylor_variant or "").strip().lower()
        if not variant:
            # shape-based fallback for old call-sites/artifacts
            variant = "author_repo_param_i" if raw.shape[-1] in (5, 6) else "rule_rbar"

        # ----- Variant A: author repo parametric i_t output -----
        if variant == "author_repo_param_i":
            # Author taylor_para has 6 outputs with an unused disp entry.
            if raw.shape[-1] == 6:
                num_raw, den_raw, _disp_unused, pstar_aux_raw, cons_raw, i_nom_raw = [raw[..., i] for i in range(6)]
            elif raw.shape[-1] == 5:
                num_raw, den_raw, pstar_aux_raw, cons_raw, i_nom_raw = [raw[..., i] for i in range(5)]
            else:
                raise ValueError(
                    f"policy=mod_taylor variant=author_repo_param_i expects d_out=6 (or 5 compatibility), got {raw.shape[-1]}"
                )
            XiN = torch.clamp(num_raw, min=1e-12)
            XiD = torch.clamp(den_raw, min=1e-12)
            c_ss = (1.0 / params.M) ** (1.0 / (params.omega + params.gamma))
            c = torch.clamp(c_ss + cons_raw, min=1e-12)
            p_star_aux = 1.0 + pstar_aux_raw

            pstar = torch.clamp(params.M * XiN / XiD, min=1e-12)
            pi_aux = _pi_aux_from_pstar(params, pstar)
            one_plus_pi = torch.clamp(pi_aux, min=1e-12).pow(1.0 / float(params.eps))
            pi = one_plus_pi - 1.0
            Delta = float(params.theta) * pi_aux * st.Delta_prev + (1.0 - float(params.theta)) * p_star_aux
            Delta = torch.clamp(Delta, min=1e-12)
            lam = c.pow(-params.gamma)
            A = torch.exp(st.logA)
            g = params.g_bar * torch.exp(st.loggtilde)
            y = c + g
            h = y * Delta / torch.clamp(A, min=1e-12)
            w = h.pow(params.omega) / torch.clamp(lam, min=1e-12)

            out = {
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
            i_base = (1.0 + params.pi_bar) / params.beta - 1.0
            out["i_nom"] = i_base + i_nom_raw
            out["i_nom_shift"] = i_nom_raw
            i_rule = (1.0 + params.pi_bar) / params.beta - 1.0 + params.psi * (out["pi"] - params.pi_bar)
            i_old = st.i_prev if st.i_prev is not None else torch.zeros_like(out["i_nom"])
            out["i_rule_target"] = params.rho_i * i_old + (1.0 - params.rho_i) * i_rule
            return out

        # ----- Variant B/C: modified Taylor rule with regime-specific rbar -----
        # legacy_rule_rbar is an alias kept for backward compatibility.
        if variant in ("rule_rbar", "legacy_rule_rbar", "rule_rbar_zlb"):
            if raw.shape[-1] != 4:
                raise ValueError(
                    f"policy=mod_taylor variant={variant} expects d_out=4 (same economic block as Taylor), got {raw.shape[-1]}"
                )
            if rbar_by_regime is None:
                raise ValueError(f"policy=mod_taylor variant={variant} requires rbar_by_regime from flex-price SSS")

            num_raw, den_raw, pstar_aux_raw, cons_raw = [raw[..., i] for i in range(4)]
            XiN = num_raw
            XiD = den_raw
            c_ss = (1.0 / params.M) ** (1.0 / (params.omega + params.gamma))
            c = c_ss + cons_raw
            p_star_aux = 1.0 + pstar_aux_raw
            out = _common_derived(params, st, c=c, XiN=XiN, XiD=XiD, p_star_aux=p_star_aux)

            if variant == "rule_rbar_zlb":
                out["i_nom"] = i_modified_taylor_zlb(params, out["pi"], rbar_by_regime, st.s, zlb_floor=0.0)
            else:
                out["i_nom"] = i_modified_taylor(params, out["pi"], rbar_by_regime, st.s)
            out["i_rule_target"] = out["i_nom"]
            return out

        raise ValueError(f"Unknown mod_taylor variant: {mod_taylor_variant!r}")

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

    raise ValueError(policy)
