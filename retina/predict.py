from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import resize
from . import config
from .data import read_mask, read_rgb
from .masking import mask_image
from .metrics import bootstrap_mean, image_metrics, segmentation_metrics
from .models import enhance


def segment(model, rgb: np.ndarray, infer_side: int = config.INFER_SIDE) -> np.ndarray:
    native_hw = rgb.shape[:2]
    resized = resize(rgb, (infer_side, infer_side), order=1,
                     anti_aliasing=True, preserve_range=True).astype(np.float32)
    prob = model(resized[None], training=False).numpy()[0, ..., 0]
    prob = resize(prob, native_hw, order=1, anti_aliasing=False, preserve_range=True)
    return np.clip(prob, 0.0, 1.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_segmentation(model, pairs, output, *, threshold=0.5,
                          apply_masking=False, mask_cfg: config.MaskConfig | None = None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    mask_cfg = mask_cfg or config.MaskConfig()
    rows = []
    for item in pairs:
        image_path, gt_path, fov_path = (list(item) + [None])[:3]
        rgb = read_rgb(image_path)
        gt = read_mask(gt_path)[..., 0] > 0.5
        fov = (read_mask(fov_path)[..., 0] > 0.5) if fov_path else None
        prob = segment(model, rgb)

        binary = None
        if apply_masking:
            result = mask_image(prob, threshold=mask_cfg.threshold,
                                roi_coverage=mask_cfg.roi_coverage,
                                erosion_iter=mask_cfg.erosion_iter,
                                dilation_iter=mask_cfg.dilation_iter, fov=fov)
            binary = result["morphed"] > 0

        metrics = segmentation_metrics(gt, prob, binary=binary, fov=fov, threshold=threshold)
        name = Path(image_path).stem
        Image.fromarray(np.uint8(prob * 255)).save(out / f"{name}_prob.png")
        if binary is not None:
            Image.fromarray(np.uint8(binary) * 255).save(out / f"{name}_mask.png")
        rows.append(dict(image=name, **metrics))
        print(f"evaluated {name}", flush=True)

    _write_csv(out / "per_image_segmentation.csv", rows)
    summary = {m: bootstrap_mean([r[m] for r in rows])
               for m in ("dice", "iou", "precision", "recall", "specificity",
                         "roc_auc", "average_precision")}
    _write_csv(out / "summary.csv",
               [dict(metric=m, **stats) for m, stats in summary.items()])
    return summary


def evaluate_super_resolution(model, image_paths, output, *, scale, crop=3, channel="y"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        hr = image.crop((0, 0, w - w % scale, h - h % scale))  # modcrop
        lr = hr.resize((hr.width // scale, hr.height // scale), Image.Resampling.BICUBIC)
        ref = np.asarray(hr, np.float32) / 255.0
        low = np.asarray(lr, np.float32) / 255.0
        name = Path(image_path).stem
        for variant in ("bicubic", "espcn_y", "espcn_all"):
            if variant == "bicubic":
                pred = np.asarray(lr.resize(hr.size, Image.Resampling.BICUBIC), np.float32) / 255.0
            else:
                pred = enhance(low, model, scale, variant.split("_")[-1])
            Image.fromarray(np.uint8(pred * 255)).save(out / f"{name}_{variant}.png")
            rows.append(dict(image=name, variant=variant, **image_metrics(ref, pred, crop, channel)))
        print(f"SR evaluated {name}", flush=True)

    _write_csv(out / "per_image_sr.csv", rows)
    summary = []
    for variant in sorted({r["variant"] for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        for metric in ("mse", "psnr_db", "ssim"):
            stats = bootstrap_mean([r[metric] for r in subset])
            if metric == "psnr_db" and any(r["psnr_infinite"] for r in subset):
                stats = dict(n=len(subset), mean=None, ci_low=None, ci_high=None)
            summary.append(dict(variant=variant, metric=metric, **stats))
    _write_csv(out / "summary.csv", summary)
    return summary
