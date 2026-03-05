from __future__ import annotations
import torch


def transition_probs(P: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Return transition probabilities for each element of a batch.

    Inputs:
      P: (2,2) with orientation P[s_current, s_next] (row-stochastic)
      s: (B,) long in {0,1}

    Output:
      probs_next: (2,B) where probs_next[k,b] = P[s[b], k]
    """
    assert P.shape == (2, 2)
    assert s.dtype == torch.long
    assert s.ndim == 1
    assert torch.all((s == 0) | (s == 1)), "s must be in {0,1}"
    return P[s, :].T.contiguous()  # (2,B)


def simulate_markov(P: torch.Tensor, s0: torch.Tensor, T: int) -> torch.Tensor:
    """
    Simulate a 2-state Markov chain.

    Convention: P[s_current, s_next] (row-stochastic), with states {0,1}.
    Returns:
      s: (T,B) long
    """
    assert P.shape == (2, 2)
    assert T >= 1
    assert s0.dtype == torch.long
    assert s0.ndim == 1
    assert torch.all((s0 == 0) | (s0 == 1)), "s0 must be in {0,1}"

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
        p0 = P[cur, 0]                # P(s_next=0 | s_current=cur)
        s[t] = torch.where(u < p0, torch.zeros_like(cur), torch.ones_like(cur))

    return s
