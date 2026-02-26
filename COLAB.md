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
