from __future__ import annotations
import torch


def transition_probs(P: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Return transition probabilities for each element of a batch.

    Inputs:
      P: (R,R) with orientation P[s_current, s_next] (row-stochastic)
      s: (B,) long in {0,...,R-1}

    Output:
      probs_next: (R,B) where probs_next[k,b] = P[s[b], k]
    """
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square (R,R)"
    assert s.dtype == torch.long
    assert s.ndim == 1
    R = int(P.shape[0])
    assert torch.all((s >= 0) & (s < R)), f"s must be in {{0,...,{R-1}}}"
    return P[s, :].T.contiguous()  # (R,B)


def simulate_markov(P: torch.Tensor, s0: torch.Tensor, T: int) -> torch.Tensor:
    """
    Simulate an R-state Markov chain.

    Convention: P[s_current, s_next] (row-stochastic), with states {0,...,R-1}.
    Returns:
      s: (T,B) long
    """
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square (R,R)"
    assert T >= 1
    assert s0.dtype == torch.long
    assert s0.ndim == 1
    R = int(P.shape[0])
    assert torch.all((s0 >= 0) & (s0 < R)), f"s0 must be in {{0,...,{R-1}}}"

    # Optional but recommended: validate P rows sum to 1 (within tolerance)
    rowsum = P.sum(dim=1)
    if not torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-7, rtol=0.0):
        raise ValueError("P must satisfy sum_{s_next} P[s_current, s_next] = 1 for each row")
    if torch.any(P < 0) or torch.any(P > 1):
        raise ValueError("P entries must be in [0,1]")

    B = s0.shape[0]
    s = torch.empty((T, B), dtype=torch.long, device=s0.device)
    s[0] = s0

    for t in range(1, T):
        cur = s[t - 1]                # (B,)
        u = torch.rand(B, device=s0.device)
        probs = P[cur, :]             # (B,R)
        cdf = torch.cumsum(probs, dim=-1)
        cdf[:, -1] = 1.0
        s[t] = torch.sum(u.view(-1, 1) > cdf, dim=-1).to(torch.long)

    return s
