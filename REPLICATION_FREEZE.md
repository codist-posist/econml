# Replication Freeze: Author-Alignment Policy

This file fixes the agreed project policy for replication quality and code changes.

## Goal

Primary target: reproduce the paper economics and results as faithfully as possible.

Two modes are allowed:

- `strict_author`: maximize runtime/implementation similarity to the public author code.
- `robust`: keep numerically safer engineering choices that do not change model equations.

## What Must Stay Aligned (Do Not Drift)

1. Economic equations and variable meaning (Appendix A.1 / A.2 / A.3 blocks).
2. Shock laws of motion and Markov switching semantics.
3. Moment/figure definitions used for reported outputs.
4. Commitment timeless workflow (SSS-aware initialization before timeless simulation).

## Safe Engineering Improvements (Allowed to Keep)

These may remain enabled because they are computational/diagnostic and do not change equations:

1. Residual chunking / memory-safe evaluation.
2. Better artifact validation and run selection checks.
3. Extra diagnostics (`eq_rms`, `eq_max`, sanity logs, progress bars).
4. GPU/CPU performance knobs that do not modify objective definitions.

## Author-Like Behavior To Preserve in `strict_author`

1. `TrainConfig.author_like(...)` preset for compute profile and training mode.
2. Commitment initialization switch:
   - `sss` (paper-faithful baseline for timeless workflow),
   - `author` (author-style warm start for sensitivity checks).
3. No hidden equation changes under preset switches.

## Change-Control Rule

Any change touching training objective, residual definitions, state transitions, or reported moments must include:

1. explicit note in commit message,
2. before/after metrics (train quality + sanity),
3. statement whether paper-faithful behavior changed.

