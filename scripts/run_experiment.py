from __future__ import annotations
import argparse
import json
import os
import sys
import time
from glob import glob
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="DRIVE root (training/ and test/)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--unet-epochs", type=int, default=40)
    ap.add_argument("--sr-epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100, help="batches per epoch")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads cap (protect the machine)")
    ap.add_argument("--seed", type=int, default=20240501)
    args = ap.parse_args()


    t = str(args.threads)
    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(var, t)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import keras
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    from retina import config, losses
    from retina.data import DriveDataset, EspcnDataset, list_pairs, read_mask, read_rgb
    from retina.metrics import bootstrap_mean, segmentation_metrics
    from retina.models import build_espcn, build_unet, compile_model
    from retina.predict import evaluate_super_resolution, segment

    root = Path(args.data_root)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env = dict(tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__)
    print("ENV", env, flush=True)
    keras.utils.set_random_seed(args.seed)
    ds_kw = dict(workers=1, use_multiprocessing=False, max_queue_size=4)

    pairs = list_pairs(str(root / "training/images/*.tif"),
                       str(root / "training/1st_manual/*.gif"))
    n = len(pairs)
    n_val = max(1, n // 7)
    train_pairs = pairs[: n - 2 * n_val]
    val_pairs = pairs[n - 2 * n_val: n - n_val]
    test_pairs = pairs[n - n_val:]
    print(f"split: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test", flush=True)

    # --- U-Net training ---
    dataset = DriveDataset(train_pairs, batch_size=config.BATCH_SIZE,
                           crops_per_image=config.CROPS_PER_IMAGE, crop_size=config.CROP_SIZE,
                           steps_per_epoch=args.steps, augment=True, seed=args.seed, **ds_kw)
    unet = compile_model(build_unet(config.BASE_FILTERS, config.DROPOUT),
                         losses.resolve("bce"), config.LEARNING_RATE)
    t0 = time.perf_counter()
    hist = unet.fit(dataset, epochs=args.unet_epochs, verbose=2)
    train_s = round(time.perf_counter() - t0, 1)
    unet.save_weights(str(out / "unet.weights.h5"))

    def predict_set(subset):
        return [(Path(p).stem, read_mask(g)[..., 0] > 0.5, segment(unet, read_rgb(p)))
                for p, g in subset]


    val_pred = predict_set(val_pairs)
    candidates = [round(x, 2) for x in np.arange(0.05, 0.61, 0.05)]
    sweep = [(thr, float(np.mean([segmentation_metrics(gt, pr, threshold=thr)["dice"]
                                  for _, gt, pr in val_pred]))) for thr in candidates]
    best_thr = max(sweep, key=lambda r: (r[1], r[0]))[0]
    print("threshold sweep (val):", sweep, "-> best", best_thr, flush=True)


    test_rows = [dict(image=nm, **segmentation_metrics(gt, pr, threshold=best_thr))
                 for nm, gt, pr in predict_set(test_pairs)]
    seg_summary = {m: bootstrap_mean([r[m] for r in test_rows])["mean"]
                   for m in ("dice", "iou", "precision", "recall", "specificity", "roc_auc")}


    sr_images = sorted(glob(str(root / "training/images/*.tif")))
    sr_ds = EspcnDataset(sr_images, scale=config.SR_SCALE, hr_size=config.SR_SCALE * 32,
                         batch_size=config.BATCH_SIZE, steps_per_epoch=args.steps,
                         seed=args.seed, **ds_kw)
    espcn = compile_model(build_espcn(config.SR_SCALE), keras.losses.MeanSquaredError(),
                          config.SR_LEARNING_RATE)
    espcn.fit(sr_ds, epochs=args.sr_epochs, verbose=2)
    espcn.save_weights(str(out / "espcn.weights.h5"))
    sr_test = sorted(glob(str(root / "test/images/*.tif")))[:5] or sr_images[:5]
    sr_summary = evaluate_super_resolution(espcn, sr_test, out / "sr_eval",
                                           scale=config.SR_SCALE, crop=3, channel="y")

    report = dict(env=env, seed=args.seed, unet_epochs=args.unet_epochs, sr_epochs=args.sr_epochs,
                  seg_train_seconds=train_s, seg_loss_curve=[float(x) for x in hist.history["loss"]],
                  calibrated_threshold=best_thr, threshold_sweep=sweep,
                  test_per_image=test_rows, seg_summary_test=seg_summary,
                  sr_summary=[s for s in sr_summary if s["metric"] in ("psnr_db", "ssim")])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT", json.dumps(report["seg_summary_test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
