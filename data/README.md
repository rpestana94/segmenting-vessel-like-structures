# Datasets — download instructions (no image data is bundled in this repository)

This repository ships **code and trained weights only**. The image datasets are
public, third-party resources and are **not redistributed here**; download each
from its official source and place it in the matching `data/<name>/` folder. Each
subfolder has a `README.md` listing the **exact filenames and SHA-256 checksums**
the scripts expect, so you can verify you have the identical files.

| Dataset | Folder | Needed for | Bundled? | Source |
|---|---|---|---|---|
| Set5  | `data/Set5/`           | Item 4 (SR metrics)          | no | HuggingFace `eugenesiow/Set5` |
| Set14 | `data/Set14/`          | Item 4 (SR metrics)          | no | HuggingFace `eugenesiow/Set14` |
| DRIVE | `data/DRIVE_training/` | Item 5 (segmentation), Item 8 | no | drive.grand-challenge.org (registration) |
| HRF   | `data/HRF/`            | Item 5 (optional cross-dataset) | no | www5.cs.fau.de fundus-images |

## Why links instead of the files
The images are not ours. DRIVE in particular requires registration and its terms
restrict redistribution; Set5/Set14/HRF are public but are third-party benchmarks.
Distributing download instructions + per-file SHA-256 (rather than copies) is the
standard, licence-safe way to make the analysis reproducible.

## Quick start
```bash
bash descargar_datos.sh      # fetches Set5 + Set14 automatically (HuggingFace)
# DRIVE and HRF require manual download — see data/DRIVE_training/README.md
#                                           and data/HRF/README.md
bash reproducir_todo.sh      # runs Items 4, 5, 8
```
Set5 + Set14 are enough for Item 4. DRIVE is required for Items 5 and 8.
HRF is optional (cross-dataset check only).
