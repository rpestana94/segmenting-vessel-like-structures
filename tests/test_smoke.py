import numpy as np
from retina import masking, metrics


def test_masking_pipeline_runs_and_is_binary():
    demo = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    out = masking.mask_image(demo)
    for stage in ("binary", "morphed"):
        assert set(np.unique(out[stage]).tolist()) <= {0, 1}
    assert out["params"]["threshold"] == 100
    assert out["pixel_counts"]["roi"] > 0


def test_masking_threshold_monotonic():
    img = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    low = masking.mask_image(img, threshold=50)["binary"].sum()
    high = masking.mask_image(img, threshold=200)["binary"].sum()
    assert low > high  


def test_segmentation_metrics_perfect():
    gt = np.zeros((50, 50), bool)
    gt[10:40, 20] = True
    prob = np.where(gt, 0.9, 0.01)
    m = metrics.segmentation_metrics(gt, prob, threshold=0.5)
    assert m["dice"] == 1.0 and m["iou"] == 1.0
    assert m["roc_auc"] == 1.0


def test_segmentation_metrics_rejects_out_of_range():
    gt = np.zeros((11, 11), bool)
    bad = np.full((11, 11), 2.0)
    try:
        metrics.segmentation_metrics(gt, bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for probabilities outside [0,1]")


def test_image_metrics_identity_is_infinite_psnr():
    rng = np.random.default_rng(0)
    ref = rng.random((40, 40, 3))
    m = metrics.image_metrics(ref, ref.copy(), crop=3, channel="rgb")
    assert m["psnr_infinite"] and m["ssim"] > 0.999


def test_bootstrap_mean_handles_none():
    stats = metrics.bootstrap_mean([0.5, 0.6, None, 0.7, float("nan")])
    assert stats["n"] == 3 and 0.5 <= stats["mean"] <= 0.7
