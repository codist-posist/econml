# Google Colab Quick Start (CUDA)

## 1) Clone repo in Colab

```bash
!git clone https://github.com/codist-posist/econml.git
%cd /content/econml
```

## 2) Install and verify dependencies

```bash
!pip -q install numpy pandas matplotlib tqdm nbformat nbclient jupyter
```

PyTorch with CUDA is usually preinstalled in Colab. Verify:

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

## 3) Run training notebooks one by one

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/00_main_pipeline.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/10_train_taylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/11_train_mod_taylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/12_train_discretion.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/13_train_commitment.ipynb
```

Training notebooks auto-select device:
`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.

### Optional: run critique scenarios A/B

The training notebooks now support:
- `UNCERTAINTY_VARIANT=baseline` (default)
- `UNCERTAINTY_VARIANT=A` (regime-dependent `sigma_tau`)
- `UNCERTAINTY_VARIANT=B` (regime-dependent `sigma_A,sigma_tau,sigma_g`)

Example for scenario A with `bad_multiplier=2.0`:

```bash
%env UNCERTAINTY_VARIANT=A
%env BAD_MULTIPLIER=2.0
%env NORMAL_MULTIPLIER=1.0
%env ARTIFACTS_ROOT=/content/econml/artifacts

!jupyter nbconvert --to notebook --execute --inplace notebooks/00_main_pipeline.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/10_train_taylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/11_train_mod_taylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/12_train_discretion.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/13_train_commitment.ipynb
```

Scenario outputs are saved under:
- `artifacts/critique/A_bm_2/...`
- `artifacts/critique/B_bm_2/...`

## 4) Run analysis and figure notebooks

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/90_results_analysis.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/91_fig1_ergodic_distributions.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/92_fig2_transition_commitment_vs_discretion.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/93_fig3_persistent_vs_temporary.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/94_fig4_persistence_sensitivity.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/96_fig6_asymmetry.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/99_fig9_taylor_vs_modtaylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/100_fig10_sensitivity_p21.ipynb
```

All notebooks include a `PROJECT_ROOT` bootstrap and work both locally and in Colab (`/content/econml`).
Figure notebooks now honor `ARTIFACTS_ROOT` from environment, so you can switch between baseline and critique runs without editing code.

## 5) Compare Critique Variants A vs B

If you train scenario A and B into separate artifacts directories, compare them with:

```bash
!python scripts/compare_uncertainty_variants.py \
  --artifacts_a /content/econml/artifacts/critique/A_bm_2 \
  --artifacts_b /content/econml/artifacts/critique/B_bm_2 \
  --device cuda --dtype float64 \
  --bad_multiplier 2.0 --normal_multiplier 1.0
```

Outputs:
- `artifacts/comparisons/table2_variants_combined.csv`
- `artifacts/comparisons/table2_variantB_minus_variantA.csv`

## 6) Fast Discretion Retraining on 4-Point Grid

For critique sensitivity with full retraining of discretion (A/B variants) on a fixed
4-point grid of `bad_multiplier`:

```bash
!python scripts/train_discretion_uncertainty_sweep.py \
  --artifacts_root /content/econml/artifacts/critique_discretion \
  --device cuda \
  --bad_grid 1.0,1.25,1.5,2.0
```

Notes:
- Uses a faster discretion preset (`TrainConfig.mid_discretion_fast`) to reduce OOM risk.
- Saves one run per scenario under `artifacts/critique_discretion/variant_*_bm_*/runs/discretion/...`
- Writes summary: `artifacts/critique_discretion/discretion_sweep_summary.csv`

## 7) New Critique Figures

After generating comparison CSVs and sweep summary:

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/101_fig11_critique_variants.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/102_fig12_critique_sensitivity.ipynb
```

Saved plots:
- `artifacts/figures/fig11_critique_variants_table2.png`
- `artifacts/figures/fig11_critique_variants_delta.png`
- `artifacts/figures/fig12_discretion_bad_multiplier_sensitivity.png`

## 8) Zero-Manual Figure Run (baseline + A + B)

If you do not want to switch anything by hand, run one command:

```bash
!python scripts/run_figures_all_scenarios.py \
  --base_artifacts_root /content/econml/artifacts \
  --bad_multiplier 2.0
```

This command will:
- execute `90/91/92/93/94/96/99/100` for `baseline`, `A`, `B` automatically,
- build `A vs B` comparison CSVs,
- execute `101_fig11_critique_variants.ipynb`,
- execute `102_fig12_critique_sensitivity.ipynb` if sweep summary exists.
