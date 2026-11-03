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

### Other datasets used in the paper (not required by the reproduction scripts)

These are the non-retinal / additional domains the article applied the pipeline to.
They are **not** consumed by `reproducir_todo.sh` or `reproduce_paper.py` (which
cover the retinal segmentation and the SR benchmarks); they are catalogued here for
completeness of Item 3, with per-image filenames + SHA-256 in the response package
(`02_evidencia_soporte/imagenes_por_dataset/`). The article's use of these domains
is qualitative — there is no pixel-level ground truth for them.

| Dataset | Count | Format | Source |
|---|---:|---|---|
| STARE (retina) | 16 | `.ppm`, 700×605 RGB | STARE project, Hoover et al. — https://cecas.clemson.edu/~ahoover/stare/ |
| PCB defects | 16 | `.jpg`, high-res | PKU-Market-PCB / HRIPCB, Open Lab, Peking University — https://robotics.pkusz.edu.cn/resources/dataset/ (6 defect types: missing hole, mouse bite, open circuit, short, spur, spurious copper) |
| Citrus leaves | 16 | `.png`/`.jpg`, 256×256 RGB | Rauf et al., Mendeley Data — https://data.mendeley.com/datasets/3f83gxmv57 (CC BY 4.0; healthy subset used) |

## Why links instead of the files
The images are not ours. DRIVE requires accepting the site's terms before download
and carries no clearly specified open licence; Set5/Set14/HRF and the paper's
non-retinal datasets are public third-party resources. Distributing download
instructions + per-file SHA-256 (rather than copies) is the standard, licence-safe
way to make the analysis reproducible.

## Quick start
```bash
bash descargar_datos.sh      # fetches Set5 + Set14 automatically (HuggingFace)
# DRIVE and HRF require manual download — see data/DRIVE_training/README.md
#                                           and data/HRF/README.md
bash reproducir_todo.sh      # runs Items 4, 5, 8
```
Set5 + Set14 are enough for Item 4. DRIVE is required for Items 5 and 8.
HRF is optional (cross-dataset check only).
