#!/usr/bin/env python

import sys, os, glob, csv
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RETINA_REPO",
    os.path.join(HERE, ".."))   
sys.path.insert(0, os.path.abspath(REPO))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from retina.models import build_espcn, enhance          
from retina.metrics import image_metrics, bootstrap_mean 

def process_dataset(hr_glob, espcn, scale, channel, crop):
    rows = []
    for path in sorted(glob.glob(hr_glob)):
        name = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")
        w, h = image.size
        hr = image.crop((0, 0, w - w % scale, h - h % scale))        
        lr = hr.resize((hr.width // scale, hr.height // scale), Image.BICUBIC)
        ref = np.asarray(hr, np.float32) / 255.0
        low = np.asarray(lr, np.float32) / 255.0
        for variant in ("bicubic", "espcn_y"):
            if variant == "bicubic":
                pred = np.asarray(lr.resize(hr.size, Image.BICUBIC), np.float32) / 255.0
            else:
                pred = enhance(low, espcn, scale, "y")
            m = image_metrics(ref, pred, crop=crop, channel=channel)
            rows.append(dict(image=name, variant=variant,
                             mse=m["mse"], psnr_db=m["psnr_db"], ssim=m["ssim"],
                             height=hr.height, width=hr.width))
    return rows

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for r in rows: wr.writerow({k: r.get(k) for k in fields})

def summarize(rows):
    out = []
    for variant in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == variant]
        for metric in ("mse", "psnr_db", "ssim"):
            out.append(dict(variant=variant, metric=metric,
                            **bootstrap_mean([r[metric] for r in sub])))
    return out

def main():
    set5_dir, set14_dir, weights = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else HERE
    scale, channel, crop = 3, "y", 3
    os.makedirs(out, exist_ok=True)

    espcn = build_espcn(scale=scale); espcn.load_weights(weights)

    for tag, d in (("set5", set5_dir), ("set14", set14_dir)):
        rows = process_dataset(os.path.join(d, "*_HR.png"), espcn, scale, channel, crop)
        write_csv(os.path.join(out, f"per_image_{tag}.csv"), rows,
                  ["image", "variant", "mse", "psnr_db", "ssim", "height", "width"])
        write_csv(os.path.join(out, f"summary_{tag}.csv"), summarize(rows),
                  ["variant", "metric", "n", "mean", "ci_low", "ci_high"])
        n_img = len({r["image"] for r in rows})
        print(f"{tag}: {n_img} images, {len(rows)} rows -> per_image_{tag}.csv + summary_{tag}.csv")

if __name__ == "__main__":
    main()
