# Retinal image pipeline — reproducibility package

Reproducible code and trained weights for the super-resolution + U-Net
segmentation + masking pipeline of Villarreal et al., *Scientific Reports*
**14**:30332 (2024), [doi:10.1038/s41598-024-81680-9](https://doi.org/10.1038/s41598-024-81680-9).

> **Scope.** This is a post-publication reimplementation that integrates the three
> pipeline stages (ESPCN super-resolution, U-Net vessel segmentation, image-processing
> masking) behind one CLI, so the article's quantitative claims can be re-run and
> checked. It is **not** the original 2022 code (which was not fully preserved);
> the ESPCN and masking modules are faithful reconstructions, and the segmentation
> U-Net is the original preserved architecture + checkpoint. See each module's
> status in the response letter and `THIRD_PARTY_NOTICES.md`.

## What runs which editor item

| Item | Script | Output |
|---|---|---|
| **4** — PSNR/MSE/SSIM | `reproducir/compute_sr_metrics.py` (via `reproducir_todo.sh`) | per-image + summary CSVs (Set5/Set14) |
| **5** — segmentation vs ground truth | `reproducir/compute_segmentation_metrics.py` | Dice/IoU/precision/recall/AUC (DRIVE 21–40) |
| **8** — complexity / runtime | `reproducir/reproduce_timing.py` | per-stage benchmark CSV + JSON |
| tables/figures of the paper | `reproduce_paper.py` | SR + segmentation tables, panel figures 4/5 |
| end-to-end demo | `run_demo.sh` | seg (±masking), SR, masking on one image |
| seeded retraining | `entrenar_base_reproducible.sh` | `unet.weights.h5` + history + metadata |

One command runs Items 4, 5 and 8: **`bash reproducir_todo.sh`**.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.9
pip install -r requirements.txt
bash descargar_datos.sh      # fetches Set5 + Set14 (DRIVE/HRF: see data/README.md)
bash reproducir_todo.sh      # Items 4, 5, 8  ->  resultados_reproduccion/
```

## Data — not bundled, download by link
No image data is committed. Every `data/<name>/README.md` lists the official
download link and the **exact filenames + SHA-256** the scripts expect, so you can
verify byte-for-byte that you have the same images. See **[`data/README.md`](data/README.md)**.
Datasets are third-party (DRIVE requires registration and restricts redistribution;
Set5/Set14/HRF are public benchmarks) — hence links, not copies.

## Weights — tracked with Git LFS
Model weights (`*.h5`, `*.hdf5`) are stored via **[Git LFS](https://git-lfs.com)**.
Install LFS (`git lfs install`) before cloning, or the weight files arrive as pointers.

| Weights | What it is | SHA-256 (first 16) |
|---|---|---|
| `weights/baseline_unet_...best.hdf5` | **original published** U-Net checkpoint (reproduces the paper figures) | `4700217fc02ab367` |
| `weights/espcn_best.weights.h5` | ESPCN, fully trained (100 ep, BSD100), seed 20240501 | `833a5db13ca2cdd8` |
| `weights/seed_20240501/` | one full seeded retraining run (U-Net + ESPCN + logs) | U-Net `0257a0c406ffac3a` |

## Layout
```
retina/              the package (SR + segmentation + masking + metrics + CLI)
reproducir/          per-item compute scripts (called by reproducir_todo.sh)
reproducir_todo.sh   one-shot: Items 4, 5, 8
reproduce_paper.py   reproduces the paper's tables/figures
run_demo.sh          end-to-end demonstration on one image
entrenar_base_reproducible.sh   seeded retraining
descargar_datos.sh   fetch Set5/Set14
weights/             model weights (Git LFS)
data/                download targets (only README/checksums committed)
tests/               unit smoke tests (pytest)  — `python -m pytest tests/`
requirements.txt     pinned Python 3.9 environment
LICENSE              MIT
THIRD_PARTY_NOTICES.md   upstream attributions (MIT / Apache-2.0)
```

## Environment
Python 3.9, TensorFlow 2.16.1, Keras 3.3.3, scikit-image 0.24.0 (full set in
`requirements.txt`). Seeded runs are reproducible within a fixed software/hardware
environment; bit-exact identity across different GPUs is not guaranteed.

## Licence
Code: **MIT** (see `LICENSE`). Reused upstream components retain their licences
(MIT / Apache-2.0) — see `THIRD_PARTY_NOTICES.md`. Datasets keep their own terms.
