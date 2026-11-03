from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_auc_score


def segmentation_metrics(gt, probability, binary=None, fov=None, threshold=0.5) -> dict:
    gt, probability = np.asarray(gt, bool), np.asarray(probability, float)
    if gt.shape != probability.shape or gt.ndim != 2:
        raise ValueError("GT and probability must be aligned 2D arrays")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probabilities must be finite and in [0,1]")
    roi = np.ones_like(gt) if fov is None else np.asarray(fov, bool)
    if roi.shape != gt.shape or not roi.any():
        raise ValueError("FOV must match shape and contain pixels")
    prediction = probability >= threshold if binary is None else np.asarray(binary, bool)
    if prediction.shape != gt.shape:
        raise ValueError("Binary prediction shape mismatch")

    y, pred, score = gt[roi], prediction[roi], probability[roi]
    tp = int(np.sum(y & pred)); tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred))

    def ratio(n, d):
        return float(n / d) if d else None

    both_empty = tp + fp + fn == 0
    two_classes = bool(y.any() and (~y).any())
    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn, n_pixels=int(y.size),
        dice=1.0 if both_empty else ratio(2 * tp, 2 * tp + fp + fn),
        iou=1.0 if both_empty else ratio(tp, tp + fp + fn),
        precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
        specificity=ratio(tn, tn + fp),
        roc_auc=float(roc_auc_score(y, score)) if two_classes else None,
        average_precision=float(average_precision_score(y, score)) if two_classes else None,
    )


def image_metrics(reference, prediction, crop=3, channel="rgb") -> dict:
    """Float RGB in [0,1]; optional BT.601 luminance; 11x11 Gaussian SSIM."""
    ref, pred = np.asarray(reference, float), np.asarray(prediction, float)
    if ref.shape != pred.shape or ref.ndim != 3 or ref.shape[-1] != 3:
        raise ValueError("SR metrics require aligned RGB arrays")
    if crop < 0:
        raise ValueError("Crop must be nonnegative")
    for arr in (ref, pred):
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError("SR images must have finite values in [0,1]")
    if crop:
        ref, pred = ref[crop:-crop, crop:-crop], pred[crop:-crop, crop:-crop]
    if min(ref.shape[:2]) < 11:
        raise ValueError("At least 11x11 pixels required after border crop")
    if channel == "y":
        from skimage.color import rgb2ycbcr
        ref, pred = rgb2ycbcr(ref)[..., 0] / 255, rgb2ycbcr(pred)[..., 0] / 255
    elif channel != "rgb":
        raise ValueError("channel must be rgb or y")

    mse = float(np.mean((ref - pred) ** 2))
    ssim = float(structural_similarity(
        ref, pred, data_range=1.0,
        channel_axis=-1 if channel == "rgb" else None,
        gaussian_weights=True, sigma=1.5, win_size=11,
        use_sample_covariance=False, K1=0.01, K2=0.03))
    return dict(mse=mse, psnr_db=None if mse == 0 else float(10 * np.log10(1 / mse)),
                psnr_infinite=bool(mse == 0), ssim=ssim)


def bootstrap_mean(values, seed=20240501, repetitions=2000) -> dict:
    values = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if not values.size:
        return dict(n=0, mean=None, ci_low=None, ci_high=None)
    if values.size == 1:
        return dict(n=1, mean=float(values[0]), ci_low=None, ci_high=None)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (repetitions, len(values)), replace=True), axis=1)
    return dict(n=int(values.size), mean=float(values.mean()),
                ci_low=float(np.quantile(means, 0.025)), ci_high=float(np.quantile(means, 0.975)))
