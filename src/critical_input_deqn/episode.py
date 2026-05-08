from __future__ import annotations

import torch

from .config import BaselineParams, RULE_OUTPUT_NAMES, NATURAL_OUTPUT_NAMES
from .economics import derive_rule, unpack_rule_state
from .transforms import decode_natural_outputs, decode_rule_outputs


def random_rule_step(
    z: torch.Tensor,
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
) -> torch.Tensor:
    """One simulated state transition under the current policy network.

    This mirrors the author-code DEQN logic: the network is used to generate
    an endogenous state path, and residual minimization is then performed on
    states from that path.  The step is used for sampling training states, not
    for differentiating through the whole simulated path.
    """

    st = unpack_rule_state(z)
    out = decode_rule_outputs(rule_net(z), RULE_OUTPUT_NAMES)
    out_n = decode_natural_outputs(natural_net(z[..., :6]), NATURAL_OUTPUT_NAMES)
    drv = derive_rule(st, out, params, Y_n=out_n["Y_n"], R_n=out_n["R_n_real"], policy=policy)

    lam_D = torch.exp(st.ell_D)
    lam_X = torch.exp(st.ell_X)
    n_D = torch.poisson(torch.clamp(lam_D, min=1e-12))
    n_X = torch.poisson(torch.clamp(lam_X, min=1e-12))

    D_next = (1.0 - float(params.delta_D)) * st.D + n_D * float(params.mark_D)
    X_next = (1.0 - float(params.delta_X)) * st.X + n_X * float(params.mark_X)
    ell_D_next = (
        (1.0 - float(params.rho_lambda_D)) * float(params.log_bar_lambda_D)
        + float(params.rho_lambda_D) * st.ell_D
        + float(params.kappa_D_lambda) * st.D
        + float(params.sigma_lambda_D) * torch.randn_like(st.ell_D)
    )
    ell_X_next = (
        (1.0 - float(params.rho_lambda_X)) * float(params.log_bar_lambda_X)
        + float(params.rho_lambda_X) * st.ell_X
        + float(params.beta_X) * st.D
        + float(params.sigma_lambda_X) * torch.randn_like(st.ell_X)
    )
    log_Z_next = float(params.rho_z) * st.log_Z + float(params.sigma_z) * torch.randn_like(st.log_Z)
    log_Delta_next = torch.log(torch.clamp(drv["Delta"], min=1e-12))

    return torch.stack(
        [
            D_next,
            X_next,
            ell_D_next,
            ell_X_next,
            log_Z_next,
            drv["A_next"],
            log_Delta_next,
        ],
        dim=-1,
    )


def simulate_rule_episode(
    initial_z: torch.Tensor,
    rule_net,
    natural_net,
    *,
    policy: str,
    params: BaselineParams,
    length: int,
) -> torch.Tensor:
    """Simulate a batch of parallel DEQN episodes.

    Returns a tensor of shape (length, batch, state_dim).  The first slice is
    the supplied initial state, matching the episode buffer in the Keras code.
    """

    states = [initial_z]
    z = initial_z
    with torch.no_grad():
        for _ in range(1, int(length)):
            z = random_rule_step(z, rule_net, natural_net, policy=policy, params=params)
            states.append(z)
    return torch.stack(states, dim=0)
