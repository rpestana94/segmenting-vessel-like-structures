# Set5 — super-resolution benchmark (5 images)

Canonical 5-image SR benchmark (baby, bird, butterfly, head, woman). Fixed, standard since ~2012. Used by Item 4 (PSNR/MSE/SSIM). Files are the HR/LR (3×) pairs.

## Download
- **Source:** https://huggingface.co/datasets/eugenesiow/Set5  (HR tarball: https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz)

> **Licence / redistribution:** Public, openly redistributed. The article's originally-cited link is no longer operational; this HuggingFace mirror serves the identical canonical set.

**Used by:** reproducir_todo.sh (Item 4), reproduce_paper.py, run_demo.sh

## Files (10 files, 0.9 MB) — place them in `data/Set5/`

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `img_001_SRF_3_HR.png` | 369361 | `676c9c2ed5c02a52c463799ceeb28fbf0508490758cccdb90cf9d698f87b821c` |
| `img_001_SRF_3_LR.png` | 52148 | `0ff620bb6b068fc4cc5ec0b44561ef9a03da7d96ab7be4194f95f5352c9944ca` |
| `img_002_SRF_3_HR.png` | 120453 | `7ed486a42ce3fca8d9ed2ff328419a92a50fc64e479765552ee544eb45f187a2` |
| `img_002_SRF_3_LR.png` | 18839 | `ee9b76206d4b0ca3f4cd4f88027f5f16238134bbb8cdc6b8d25b94f212aa7ae0` |
| `img_003_SRF_3_HR.png` | 126566 | `6c1af99062dc37d69e950e493cc9cf01460f96863ea6779f20d6efb72fd991cb` |
| `img_003_SRF_3_LR.png` | 17346 | `c5366bc368d0d1ee4b1e1b0b4f95af4cc6e8a01ed9bd83ddb3a49dc63b797b33` |
| `img_004_SRF_3_HR.png` | 112937 | `ce231e0d6d38b4742fda8fb4cb0da1938537d7d3cebbcdaf7ae2bdc9bf26fbf9` |
| `img_004_SRF_3_LR.png` | 15310 | `612c398bb1140f78fb2c4a204e9d461b709b1b12dcf5f830158f59f21894d724` |
| `img_005_SRF_3_HR.png` | 118230 | `4ddb1ad08b0fa118463c53fb345402a7542c26c7320d16236e0e1504c8daaaf2` |
| `img_005_SRF_3_LR.png` | 17892 | `9c79151de3b4e51c48f08ab45f5bb897787689fc40daf7943481aafb87d54613` |

After downloading, verify integrity against the SHA-256 column above (e.g. `sha256sum data/Set5/<file>`). These are the exact files the scripts expect.