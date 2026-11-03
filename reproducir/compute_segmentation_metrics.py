#!/usr/bin/env python

import sys, os, glob, csv
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RETINA_REPO",
    os.path.join(HERE, ".."))   
sys.path.insert(0, os.path.abspath(REPO))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from retina.models import build_unet
from retina.predict import segment
from retina.metrics import segmentation_metrics, bootstrap_mean

def main():
    img_dir, gt_dir, fov_dir, weights = sys.argv[1:5]
    out = sys.argv[5] if len(sys.argv) > 5 else HERE
    os.makedirs(out, exist_ok=True)

    unet = build_unet(base_filters=32, dropout=0.1, channels=3)
    unet.load_weights(weights)

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    gts  = sorted(glob.glob(os.path.join(gt_dir, "*.gif")))
    fovs = sorted(glob.glob(os.path.join(fov_dir, "*.gif")))
    assert len(imgs) == len(gts) == len(fovs) and imgs, "image/GT/FOV counts must match"

    rows = []
    for im, gt, fv in zip(imgs, gts, fovs):
        name = os.path.splitext(os.path.basename(im))[0]
        x = np.asarray(Image.open(im).convert("RGB"), np.float32) / 255.0
        prob = segment(unet, x)                                  
        g = np.asarray(Image.open(gt)) > 127
        m = np.asarray(Image.open(fv).convert("L")) > 127
        r = segmentation_metrics(g, prob, fov=m, threshold=0.5)
        r["image"] = name
        rows.append(r)
        print(f"seg {name}: dice={r['dice']:.4f} iou={r['iou']:.4f} auc={r['roc_auc']:.4f}", flush=True)

    fields = ["image", "dice", "iou", "precision", "recall", "specificity",
              "roc_auc", "average_precision", "tp", "tn", "fp", "fn", "n_pixels"]
    with open(os.path.join(out, "per_image_segmentation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})


    with open(os.path.join(out, "summary_segmentation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "mean_per_image", "ci_low", "ci_high", "micro_pooled"])
        TP = sum(r["tp"] for r in rows); TN = sum(r["tn"] for r in rows)
        FP = sum(r["fp"] for r in rows); FN = sum(r["fn"] for r in rows)
        micro = {
            "dice": 2*TP/(2*TP+FP+FN), "iou": TP/(TP+FP+FN),
            "precision": TP/(TP+FP), "recall": TP/(TP+FN), "specificity": TN/(TN+FP),
        }
        for metric in ["dice","iou","precision","recall","specificity","roc_auc","average_precision"]:
            bm = bootstrap_mean([r[metric] for r in rows])
            w.writerow([metric, bm["mean"], bm["ci_low"], bm["ci_high"], micro.get(metric, "")])
    n = len(rows)
    print(f"\nWROTE per_image_segmentation.csv ({n} images) + summary_segmentation.csv")

if __name__ == "__main__":
    main()
