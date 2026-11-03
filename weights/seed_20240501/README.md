# Seeded retraining — SEED = 20240501

Fresh, seeded training of both neural modules (not the original, non-seeded 2022 run).
Each folder holds the best checkpoint (lowest training loss), the per-epoch log
(`history.csv`) and `metadata.json`.

| | U-Net (segmentation) | ESPCN (3× super-resolution) |
|---|---|---|
| Weights | `unet/unet.weights.h5` | `espcn/espcn.weights.h5` |
| Training data | DRIVE, 20 training images (21–40) | BSD100, 100 HR images, Y channel |
| Epochs × steps | 100 × 100 | 100 × 200 |
| Batch | 16 (4 images × 4 random 64×64 crops) | 16 random 96×96 HR / 32×32 LR (bicubic) crops |
| Architecture | U-Net 32→512 filters, dropout 0.1, sigmoid output | 64-64-32-9 conv (5/3/3/3), pixel shuffle |
| Loss | binary cross-entropy | MSE |
| Augmentation | flips (p=0.5); rotation/shift/zoom/photometric (p=0.05 each) | none |
| Optimizer | Adam, lr 1e-3 | Adam, lr 1e-3 |
| Final training loss | 0.0998 | 0.0030 |

No validation split was used; "best" means lowest training loss.

**Training environment:** NVIDIA RTX 5080, Python 3.11, TensorFlow 2.21.0, Keras 3.15.1,
XLA enabled. (TF 2.16.1 has no kernels for this GPU's compute capability 12.0.)
The weights were verified to load with the pinned `requirements.txt` stack
(TF 2.16.1 / Keras 3.3.3). Runs are reproducible for a fixed seed within the same
software/hardware environment, not bit-identical across versions.

Load:
```python
from retina.models import build_unet, build_espcn
unet = build_unet(32, 0.1); unet.load_weights("weights/seed_20240501/unet/unet.weights.h5")
espcn = build_espcn(3);     espcn.load_weights("weights/seed_20240501/espcn/espcn.weights.h5")
```
