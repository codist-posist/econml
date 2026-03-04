# Google Colab Quick Start (CUDA)

Replication policy and change-control rules are fixed in
`REPLICATION_FREEZE.md`.

## 1) Clone repo in Colab

```bash
!git clone https://github.com/codist-posist/econml.git
%cd /content/econml
```

## 2) Install dependencies

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

## 3) Train models

Core models (paper baseline workflows):

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/10_train_taylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/11_train_mod_taylor.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/12_train_discretion.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/13_train_commitment.ipynb
```

Optional models (ZLB + parameterized Taylor):

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/14_train_taylor_zlb.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/15_train_mod_taylor_zlb.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/16_train_discretion_zlb.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/17_train_commitment_zlb.ipynb
!jupyter nbconvert --to notebook --execute --inplace notebooks/18_train_taylor_para.ipynb
```

Training notebooks auto-select device:
`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.

## 4) Build tables and figures

Paper tables (Table 1-4):

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/101_paper_tables_1_4.ipynb
```

Article-mode figures (paper methodology, SSS starts where required):

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/115_paper_figures_2_14.ipynb
```

Note: full Figure 2-14 coverage needs the optional ZLB trainings (`14..17`).

Author-code artifacts (author-style postprocess + IR files):

```bash
!jupyter nbconvert --to notebook --execute --inplace notebooks/116_paper_figures_author_strict_colab.ipynb
```

## 5) Where outputs are saved

- Training runs: `artifacts/runs/<policy>/<run_id>/`
- Selected run pointers: `artifacts/selected_runs.json`
- Paper tables CSV: `artifacts/table1_calibration.csv`, `artifacts/table2_*.csv`, `artifacts/table3_*.csv`, `artifacts/table4_*.csv`
- Article-mode figures (notebook 115): `artifacts/paper_figures_nb115_article_sss/`
- Author-code exports (notebook 116, per run):
  - `author_postprocess/simulated_definitions*.npz`
  - `IRS/IR_definitions.npz`
  - `IRS/IR_states.npz`

All notebooks include a `PROJECT_ROOT` bootstrap and work both locally and in Colab (`/content/econml`).
