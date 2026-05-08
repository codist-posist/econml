from __future__ import annotations

import torch

from .config import BaselineParams, NetworkConfig, QMCConfig, RULE_OUTPUT_NAMES, NATURAL_OUTPUT_NAMES
from .networks import MLP
from .qmc import make_qmc_nodes
from .residuals import natural_residuals, rule_residuals, stack_residuals
from .sampling import natural_from_rule_states, sample_rule_states


def run_smoke(device: str = "cpu", dtype: torch.dtype = torch.float64) -> None:
    params = BaselineParams()
    qmc_cfg = QMCConfig(n_train=16, seed=7)
    net_cfg = NetworkConfig(hidden_width=32, hidden_depth=2)

    natural_net = MLP(6, len(NATURAL_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)
    rule_net = MLP(7, len(RULE_OUTPUT_NAMES), net_cfg).to(device=device, dtype=dtype)
    z = sample_rule_states(8, params=params, device=device, dtype=dtype, seed=1)
    z_n = natural_from_rule_states(z)
    nodes = make_qmc_nodes(16, cfg=qmc_cfg, device=device, dtype=dtype)

    raw_n = natural_net(z_n)
    res_n, _ = natural_residuals(
        z_n,
        raw_n,
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=1e-4,
    )
    rn = stack_residuals(res_n)
    if not torch.isfinite(rn).all():
        raise RuntimeError("Natural residuals contain non-finite values.")

    raw = rule_net(z)
    res, _ = rule_residuals(
        z,
        raw,
        rule_net,
        natural_net,
        nodes,
        params=params,
        qmc_cfg=qmc_cfg,
        fb_epsilon=1e-4,
        policy="fixed",
    )
    rr = stack_residuals(res)
    if not torch.isfinite(rr).all():
        raise RuntimeError("Rule residuals contain non-finite values.")
    print("critical_input_deqn smoke OK", {"natural_shape": tuple(rn.shape), "rule_shape": tuple(rr.shape)})


if __name__ == "__main__":
    run_smoke()

