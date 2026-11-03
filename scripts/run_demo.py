from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(var, str(args.threads))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import keras
    from retina import config, losses
    from retina.data import DriveDataset, list_pairs, read_mask, read_rgb
    from retina.metrics import segmentation_metrics
    from retina.models import build_unet, compile_model
    from retina.predict import segment

    root = Path(args.data_root)
    keras.utils.set_random_seed(config.SEED)
    pairs = list_pairs(str(root / "training/images/*.tif"),
                       str(root / "training/1st_manual/*.gif"))
    train_pairs, eval_pairs = pairs[:-2], pairs[-2:]
    ds = DriveDataset(train_pairs, batch_size=config.BATCH_SIZE,
                      crops_per_image=config.CROPS_PER_IMAGE, crop_size=config.CROP_SIZE,
                      steps_per_epoch=30, augment=True, seed=config.SEED,
                      workers=1, use_multiprocessing=False, max_queue_size=4)
    model = compile_model(build_unet(config.BASE_FILTERS, config.DROPOUT),
                          losses.resolve("bce"), config.LEARNING_RATE)
    print("U-Net parameters:", model.count_params(), flush=True)
    model.fit(ds, epochs=args.epochs, verbose=2)

    for img_path, gt_path in eval_pairs:
        gt = read_mask(gt_path)[..., 0] > 0.5
        prob = segment(model, read_rgb(img_path))
        m = segmentation_metrics(gt, prob, threshold=0.5)
        print(f"{Path(img_path).stem}: dice={m['dice']:.3f} auc={m['roc_auc']:.3f}", flush=True)
    print("demo OK", flush=True)


if __name__ == "__main__":
    main()
