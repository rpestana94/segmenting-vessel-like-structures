import argparse
import glob
import os

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

from retina.models import build_espcn, enhance

SCALE = 3
BORDER = 3  


def modcrop(im, s):
    w, h = im.size
    return im.crop((0, 0, w - w % s, h - h % s))


def psnr_protocols(hr_rgb, sr_rgb):
    hr = Image.fromarray(np.uint8(np.clip(hr_rgb, 0, 255)))
    sr = Image.fromarray(np.uint8(np.clip(sr_rgb, 0, 255)))
    a = np.asarray(hr.convert("YCbCr"), np.float64)
    b = np.asarray(sr.convert("YCbCr"), np.float64)
    r = BORDER
    y_std = psnr(a[r:-r, r:-r, 0], b[r:-r, r:-r, 0], data_range=255)
    ch = []
    for i in range(3):
        v = psnr(a[..., i], b[..., i], data_range=255)
        ch.append(min(v, 100.0) if np.isinf(v) else v)
    ycbcr_avg = float(np.mean(ch))
    return y_std, ycbcr_avg


def eval_set(name, hr_dir, model):
    files = sorted(glob.glob(os.path.join(hr_dir, "*_HR.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    ys, yc = [], []
    for f in files:
        hr = modcrop(Image.open(f).convert("RGB"), SCALE)
        w, h = hr.size
        lr = hr.resize((w // SCALE, h // SCALE), Image.Resampling.BICUBIC)
        lr_up = np.asarray(lr, np.float32) / 255.0
        sr = enhance(lr_up, model, SCALE, mode="y", iterations=1) * 255.0
        sr_img = Image.fromarray(np.uint8(np.clip(sr, 0, 255))).resize((w, h), Image.Resampling.BICUBIC)
        a, b = psnr_protocols(np.asarray(hr, np.float32), np.asarray(sr_img, np.float32))
        ys.append(a); yc.append(b)
    return np.mean(ys), np.mean(yc), len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set5", required=True)
    ap.add_argument("--set14", required=True)
    ap.add_argument("--weights", default="weights/espcn_best.weights.h5")
    args = ap.parse_args()

    model = build_espcn(SCALE)
    model.load_weights(args.weights)

