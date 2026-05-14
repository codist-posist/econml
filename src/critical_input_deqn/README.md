# Critical Input DEQN

This package implements the new baseline model from
`critical_input_new_baseline_model.tex`.

It is intentionally separate from the older `critical_input_rule_*` files,
which used the previous services-sector/Rotemberg formulation.  The numerical
method here remains DEQN: neural networks approximate recursive policy
functions and are trained by minimizing equilibrium residuals.

## Implemented first

- auxiliary flexible-price benchmark network;
- fixed Taylor, natural-rate-adjusted Taylor, cap-pressure bottleneck Taylor, and
  repair-aware Taylor residual systems;
- fixed Sobol/QMC expectation nodes;
- marked-Poisson transition block with deterministic marks;
- exact one-dimensional imported-input MCP for scarcity rents;
- bounded repair-capacity KKT map for adaptation investment;
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

## Preflight before long training

```bash
python -m src.critical_input_deqn.preflight
```

The preflight is not a convergence test.  It checks that the natural
benchmark, fixed Taylor, natural-rate-adjusted Taylor, cap-pressure bottleneck
Taylor, repair-aware Taylor, discretion, and commitment residual systems all
build finite residual matrices with the current state and output architecture.

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
  --policies fixed,ba,bottleneck,repair_aware
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

Training uses a maximum step/episode budget and a validation-residual stopping
profile by default.  The default profile is stricter for the natural and
rule-based networks and looser for discretion and commitment.  Stopping is
triggered only after the validation RMS and maximum absolute residual criteria
hold for `--early-stop-patience` consecutive logged checks after the minimum
step count.  Use `--no-auto-stop` to disable these defaults, or override them
with `--target-rms`, `--target-max-abs`, `--early-stop-patience`, and
`--min-steps-before-stop`.

Progress is printed during training.  With `tqdm` installed, the progress bar
shows `train_rms`, `val_rms`, `val_max`, and the current stopping-hit counter.
Without `tqdm`, the launcher prints the same diagnostics at logged checks.

For commitment, the inherited promise block is not initialized at zero.  The
initialization follows the nonzero commitment-state values used in the local
Galo--Nuno code for the pricing-promise analogues
(`vartheta_old=-0.019182`, `rho_old=0.016500`).  Because this implementation
uses scaled promises rather than carrying `c_old` as a separate state, the
pricing-promise means and standard deviations multiply the author raw values by
`0.921336**2`.  The repair-value promise is initialized as a centered cloud
because it has no exact analogue in the original model; there is no Euler
promise in the Euler-substituted implementation.  The launcher option
`--promise-init-scale` scales the standard deviations around these means; it no
longer sets arbitrary promise means.

For debugging only, the rule networks can also be trained on independently
sampled mixture states:

```bash
python -m src.critical_input_deqn.run_train --rule-trainer iid
```

## Notebook workflow

The notebooks are experiment wrappers around the package code:

1. `notebooks/critical_input_deqn_00_natural_benchmark.ipynb` trains and saves
   the auxiliary natural benchmark.
2. `notebooks/critical_input_deqn_01_fixed_taylor.ipynb` trains the fixed
   Taylor rule using the frozen benchmark.
3. `notebooks/critical_input_deqn_02_modified_taylor.ipynb` trains the
   natural-rate-adjusted Taylor rule using the frozen benchmark.
4. `notebooks/critical_input_deqn_10_bottleneck_taylor.ipynb` trains the
   cap-pressure bottleneck Taylor rule using the frozen benchmark.
5. `notebooks/critical_input_deqn_11_repair_aware_taylor.ipynb` trains the
   repair-aware Taylor rule using the frozen benchmark.
6. `notebooks/critical_input_deqn_04_discretion.ipynb` trains the discretionary
   optimal-policy network.
7. `notebooks/critical_input_deqn_05_commitment.ipynb` trains the commitment
   optimal-policy network with promise states.
8. `notebooks/critical_input_deqn_06_postprocess_artifacts.ipynb` loads all
   trained checkpoints and saves simulated states, deterministic scenario
   responses, definitions, and summary statistics for downstream figures.
9. `notebooks/critical_input_deqn_03_compare_all_policies.ipynb` loads saved
   diagnostics from the rule and optimal-policy environments and writes a
   comparison table.
10. `notebooks/critical_input_deqn_07_tables_and_figures.ipynb` builds the
   benchmark moments, counterfactual/sensitivity tables, and main figure files
   from the postprocessed arrays that already exist.

All saved results go under `baseline_artifacts/critical_input_deqn/`.

## Experiment variants

The calculation plan is encoded in `src.critical_input_deqn.experiments`.
Each non-baseline variant changes model primitives and therefore should be
solved as its own DEQN economy, not evaluated only with baseline weights.  The
registered variants include:

- core mechanism variants: `price_only` removes the quantity-cap channel and
  keeps the cap slack while retaining procurement-price shocks, and
  `quantity_only` removes the direct procurement-price channel while retaining
  the physical cap channel;
- counterfactuals: `no_cap` keeps the external price and quantity processes in
  the state but makes the cap slack, plus `no_adaptation`, `no_financing`,
  `no_relief`;
- sensitivity variants: `deep_crisis`, `persistent_crisis`, `fast_relief`,
  `fragile_relief`, granular crisis-depth/persistence variants, granular
  relief-arrival/relief-durability variants, bottleneck-tightness variants,
  technology variants, residual-import-dependence variants, repair-cost
  variants, financing-sensitivity variants, and policy-rule variants such as
  `hawkish_policy`, `dovish_policy`, `output_gap_policy`,
  `weak_bottleneck_policy`, `strong_bottleneck_policy`,
  `weak_repair_aware_policy`, and `strong_repair_aware_policy`.

The `price_only` and `no_cap` allocations can be numerically close when the cap
is far from binding.  Their roles differ diagnostically: `price_only` removes
the quantity-shock channel, while `no_cap` asks what remains of the full
external process once the physical constraint cannot generate scarcity rents.

One structured experiment can be run with:

```bash
python -m src.critical_input_deqn.run_experiment --experiment baseline
```

For a dry run that only prints the commands:

```bash
python -m src.critical_input_deqn.run_experiment --experiment quantity_only --dry-run
```

An experiment suite can be launched with:

```bash
python -m src.critical_input_deqn.run_suite --suite all_registered
```

Numerical robustness checks change approximation settings rather than model
primitives:

```bash
python -m src.critical_input_deqn.run_numerical_robustness --dry-run
```

Training dimensions can be forwarded to each experiment, for example:

```bash
python -m src.critical_input_deqn.run_suite --suite counterfactual -- --natural-steps 50000 --rule-steps 50000 --optimal-steps 50000
```

Baseline artifacts are written directly under
`baseline_artifacts/critical_input_deqn/`.  Non-baseline variants are written
under `baseline_artifacts/critical_input_deqn/experiments/<experiment>/`.

## Saved artifacts

Training writes the objects needed to restart analysis without retraining:

- `run_config.json`: calibration, network architecture, QMC settings, stopping
  rules, and launcher arguments.
- `*.pt`: PyTorch model checkpoints with network weights and metadata.
- `checkpoints/*_step_*.pt`: periodic training-state checkpoints with network
  weights, optimizer state, RNG state, current simulated state when applicable,
  and the current step/episode.  These mirror the role of the author-code
  checkpoint folders and protect long runs from losing all progress.
- `*_train_log.json`: training and validation residual paths.
- `*_eval.json`: equation-level out-of-sample residual diagnostics.

Post-processing writes the objects needed for figures and quantitative tables:

- `postprocess/*_states.npz`: simulated state paths.
- `postprocess/*_definitions.npz`: decoded controls and derived economic
  variables, including output gaps, scarcity rents, imported-input use,
  adaptation investment, next-period adaptation, marginal costs, price
  dispersion, and natural-benchmark objects.
- `postprocess/IR_*_states.npz` and `postprocess/IR_*_definitions.npz`:
  author-style conditional scenario paths.  Since the new model has no binary
  Markov regime, these files replace regime-switch IRs with controlled
  external-access scenarios: no event, one disruption event, severe disruption,
  one relief event, and delayed-relief paths after disruption.
- `postprocess/*_summary.json`: mean, standard deviation, quantiles, minimum,
  and maximum for each saved variable.
- `postprocess/postprocess_manifest.json`: run metadata for the generated
  artifacts.

The table builder writes:

- `tables/table_1_calibration.csv`;
- `tables/table_2_ergodic_moments_by_policy.csv`;
- `tables/table_3_counterfactual_decomposition_<policy>.csv`;
- `tables/table_4_sensitivity_summary_<policy>.csv`;
- `tables/table_5_irf_peak_responses_<policy>.csv`;
- `tables/table_6_numerical_diagnostics.csv`;
- `tables/table_7_numerical_robustness_<policy>.csv`;
- `tables/experiment_registry.json` and `tables/experiment_overrides.csv`.

The figure builder writes the main mechanism, policy-comparison,
counterfactual, sensitivity, distribution, convergence, residual, and
complementarity-diagnostic figures when the required postprocess files are
present.  Missing variants are skipped rather than failing the whole build.
