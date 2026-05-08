# Critical Input DEQN

This package implements the new baseline model from
`critical_input_new_baseline_model.tex`.

It is intentionally separate from the older `critical_input_rule_*` files,
which used the previous services-sector/Rotemberg formulation.  The numerical
method here remains DEQN: neural networks approximate recursive policy
functions and are trained by minimizing equilibrium residuals.

## Implemented first

- auxiliary flexible-price benchmark network;
- fixed Taylor and bottleneck-adjusted Taylor residual systems;
- fixed Sobol/QMC expectation nodes;
- marked-Poisson transition block with deterministic marks;
- smoothed Fischer--Burmeister complementarity residuals;
- mixture state sampler for normal, crisis, relief, and boundary regions.

## Not a replacement method

This is not perturbation and not a deterministic grid projection.  It keeps the
DEQN logic used in the replicated project, but the Markov-switching regime index
is replaced by the continuous/event-based external bottleneck state
`(D, X, lambda_D, lambda_X)`.

## Smoke test

```bash
python -m src.critical_input_deqn.smoke
```

## Tiny training test

```bash
python -m src.critical_input_deqn.run_small_train
```

The tiny training run only checks infrastructure.  It is not a quantitative
solution.

## Full training entry point

```bash
python -m src.critical_input_deqn.run_train ^
  --output-dir baseline_artifacts/critical_input_deqn ^
  --policies fixed,ba
```

The launcher first trains the auxiliary flexible-price benchmark network,
freezes it, and then trains the requested rule-based policy networks.  By
default, rule networks are trained in an author-style DEQN episode loop:
simulate state trajectories under the current policy network, batch the
simulated states, and minimize equilibrium residuals.  It saves model
checkpoints and equation-level residual diagnostics as JSON files.

The default network/training choices are intentionally close to the local
Keras DEQN framework: SELU hidden layers with small variance-scaled
initialization, Adam updates, and Huber residual loss.  The implementation is
PyTorch, but the numerical object remains the same: a neural approximation to
recursive policy functions trained on equilibrium residuals.

For debugging only, the rule networks can also be trained on independently
sampled mixture states:

```bash
python -m src.critical_input_deqn.run_train --rule-trainer iid
```
