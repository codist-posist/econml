# Google Colab Quick Start (CUDA)

## 1) Clone repo in Colab

```bash
!git clone https://github.com/codist-posist/econml.git
%cd /content/econml
```

## 2) Install/verify dependencies

```bash
!pip -q install numpy pandas matplotlib tqdm nbformat nbclient
```

PyTorch with CUDA is usually preinstalled in Colab. Verify:

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

## 3) Train with auto device selection

```bash
!python scripts/train_all_mid.py --device auto --show_progress
```

`--device auto` picks CUDA when available, otherwise CPU.

## 4) Run notebooks

Training notebooks and figure notebooks now include robust project-root bootstrap logic, so imports from `src/` work in both local runs and Colab clones (e.g., `/content/econml`).
