from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .deqn import Trainer, _transition_probs_to_next
from .model_common import shock_laws_of_motion, unpack_state


@dataclass(frozen=True)
class CritiqueCurriculumConfig:
    """Training-time curriculum/domain-randomization for critique stressors."""

    enabled: bool = False
    bad_state: int = 1
    ramp_episodes: int = 2_000
    shock_times: Tuple[int, ...] = (0, 4, 8, 12)
    shock_size: float = 1.25
    bad_sigma_mult: float = 2.0
    apply_to_policies: Tuple[str, ...] = (
        "taylor",
        "taylor_para",
        "mod_taylor",
        "taylor_zlb",
        "mod_taylor_zlb",
        "discretion",
        "discretion_zlb",
        "commitment",
        "commitment_zlb",
    )


@dataclass(frozen=True)
class RobustModTaylorRuleConfig:
    """Uncertainty-aware add-on for modified Taylor rules in bad regime."""

    enabled: bool = False
    bad_state: int = 1
    kappa: float = 0.02
    uncertainty_ref: float = 0.0
    max_premium: float = 0.03
    bad_only: bool = True


class CritiqueAugmentedTrainer(Trainer):
    """Trainer with optional curriculum shocks and robust mod_taylor premium."""

    def __init__(
        self,
        *args,
        curriculum: CritiqueCurriculumConfig | None = None,
        robust_rule: RobustModTaylorRuleConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.curriculum = curriculum or CritiqueCurriculumConfig()
        self.robust_rule = robust_rule or RobustModTaylorRuleConfig()
        self._critique_step_calls = 0
        self._shock_times = {int(v) for v in self.curriculum.shock_times}

    def _curriculum_position(self) -> tuple[int, int, float]:
        n_path = max(1, int(getattr(self.cfg, "n_path", 1) or 1))
        call_idx = int(self._critique_step_calls)
        ep_idx = call_idx // n_path
        t_in_ep = call_idx % n_path
        ramp = max(1, int(self.curriculum.ramp_episodes))
        strength = min(1.0, float(ep_idx) / float(ramp))
        return ep_idx, t_in_ep, float(strength)

    def _sample_next_regime(self, st, *, B: int, dev: str, dt: torch.dtype) -> torch.Tensor:
        u = torch.rand(B, device=dev, dtype=dt)
        probs = _transition_probs_to_next(self.params, st)
        if isinstance(probs, tuple):
            p0, _ = probs
            return torch.where(u < p0, torch.zeros_like(st.s), torch.ones_like(st.s))
        cdf = torch.cumsum(probs, dim=-1)
        cdf[..., -1] = 1.0
        return torch.sum(u.view(-1, 1) > cdf, dim=-1).to(torch.long)

    def _policy_outputs(self, x: torch.Tensor, *, apply_hard_bounds: bool = True):
        out = super()._policy_outputs(x, apply_hard_bounds=apply_hard_bounds)
        rr = self.robust_rule
        if not bool(rr.enabled):
            return out
        if self.policy not in ("mod_taylor", "mod_taylor_zlb"):
            return out

        st = unpack_state(x, self.policy)
        ref = float(rr.uncertainty_ref)
        if ref <= 0.0:
            ref = max(1e-8, float(self.params.sigma_tau))

        uncert = torch.abs(st.xi) / float(ref)
        premium = float(rr.kappa) * uncert
        if float(rr.max_premium) > 0.0:
            premium = torch.clamp(premium, min=0.0, max=float(rr.max_premium))

        if bool(rr.bad_only):
            premium = torch.where(
                st.s == int(rr.bad_state),
                premium,
                torch.zeros_like(premium),
            )

        out["i_nom"] = out["i_nom"] + premium
        if self.policy == "mod_taylor_zlb":
            out["i_nom"] = torch.clamp(out["i_nom"], min=0.0)
        out["i_rule_target"] = out["i_nom"]
        return out

    @torch.no_grad()
    def _step_state(self, x: torch.Tensor) -> torch.Tensor:
        """One-step law of motion with optional critique curriculum."""
        dev, dt = self.params.device, self.params.dtype
        B = x.shape[0]
        st = unpack_state(x, self.policy)
        out = self._policy_outputs(x)

        epsA = torch.randn(B, device=dev, dtype=dt)
        epsg = torch.randn(B, device=dev, dtype=dt)
        epst = torch.randn(B, device=dev, dtype=dt)

        if self.policy in ("taylor", "mod_taylor", "taylor_zlb", "mod_taylor_zlb"):
            nss = int(getattr(self.cfg, "author_n_steady_state_batches", 0) or 0)
            min_b = int(getattr(self.cfg, "author_n_steady_state_min_batch", 500) or 500)
            if nss > 0 and B > min_b:
                nss_eff = min(int(B), int(nss))
                if nss_eff > 0:
                    epsA[-nss_eff:] = 0.0
                    epsg[-nss_eff:] = 0.0
                    epst[-nss_eff:] = 0.0

        if bool(self.curriculum.enabled) and (self.policy in self.curriculum.apply_to_policies):
            _, t_in_ep, strength = self._curriculum_position()
            if strength > 0.0:
                bad_mult = 1.0 + strength * (float(self.curriculum.bad_sigma_mult) - 1.0)
                mult = torch.where(
                    st.s == int(self.curriculum.bad_state),
                    torch.full((B,), float(bad_mult), device=dev, dtype=dt),
                    torch.ones((B,), device=dev, dtype=dt),
                )
                epst = epst * mult
                if int(t_in_ep) in self._shock_times:
                    epst = epst + float(self.curriculum.shock_size) * float(strength)

        s_next = self._sample_next_regime(st, B=B, dev=dev, dt=dt)

        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(self.params, st, epsA, epsg, epst, s_next)

        if self.policy == "commitment":
            if st.c_prev is not None or self._commitment_has_c_prev:
                x_next = torch.stack(
                    [out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"], out["c"]],
                    dim=-1,
                )
            else:
                x_next = torch.stack(
                    [out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["vartheta"], out["varrho"]],
                    dim=-1,
                )
        elif self.policy == "commitment_zlb":
            x_next = torch.stack(
                [
                    out["Delta"],
                    logA_n,
                    logg_n,
                    xi_n,
                    s_n.to(dt),
                    out["vartheta"],
                    out["varrho"],
                    out["c"],
                    out["i_nom"],
                    out["varphi"],
                ],
                dim=-1,
            )
        elif self.policy == "taylor_para":
            if not bool(getattr(self, "_taylor_para_has_extended_state", False)):
                x_next = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)
            else:
                if st.p21 is not None:
                    p21_prev = st.p21
                else:
                    p21_prev = torch.full_like(out["Delta"], float(self.params.p21))
                x_next = torch.stack(
                    [out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt), out["i_nom"], p21_prev],
                    dim=-1,
                )
        else:
            x_next = torch.stack([out["Delta"], logA_n, logg_n, xi_n, s_n.to(dt)], dim=-1)

        self._critique_step_calls += 1
        return x_next
