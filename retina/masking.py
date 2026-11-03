from __future__ import annotations
import numpy as np
from scipy import ndimage

GRAYSCALE_THRESHOLD = 100  


def to_grayscale_uint8(image) -> np.ndarray:
    arr = np.asarray(image, np.float64)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    elif arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() <= 1.0 + 1e-9:  
        arr = arr * 255.0
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def circular_roi(shape, coverage: float = 0.95) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    radius = min(h, w) / 2.0 * coverage
    return ((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) <= radius ** 2


def binarize(gray, threshold: int = GRAYSCALE_THRESHOLD) -> np.ndarray:
    g = np.asarray(gray)
    if g.dtype != np.uint8:
        g = to_grayscale_uint8(g)
    return (g > int(threshold)).astype(np.uint8)


def erode_then_dilate(binary, erosion_iter: int = 1, dilation_iter: int = 1,
                      structure=None) -> np.ndarray:
    b = np.asarray(binary, bool)
    if structure is None:
        structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(b, structure=structure, iterations=int(erosion_iter))
    dilated = ndimage.binary_dilation(eroded, structure=structure, iterations=int(dilation_iter))
    return dilated.astype(np.uint8)


def mask_image(image, *, threshold=GRAYSCALE_THRESHOLD, roi_coverage=0.95,
               erosion_iter=1, dilation_iter=1, fov=None) -> dict:
    gray = to_grayscale_uint8(image)
    roi = circular_roi(gray.shape, roi_coverage) if fov is None else np.asarray(fov, bool)
    if roi.shape != gray.shape:
        raise ValueError("FOV/ROI shape does not match image")

    masked_gray = np.where(roi, gray, 0).astype(np.uint8)              # step 1
    binary = binarize(masked_gray, threshold) & roi.astype(np.uint8)  # step 2
    morphed = erode_then_dilate(binary, erosion_iter, dilation_iter) & roi.astype(np.uint8)  # step 3

    return dict(
        gray=gray, roi=roi.astype(np.uint8), masked_gray=masked_gray,
        binary=binary, morphed=morphed,
        params=dict(threshold=int(threshold), roi_coverage=float(roi_coverage),
                    erosion_iter=int(erosion_iter), dilation_iter=int(dilation_iter)),
        pixel_counts=dict(roi=int(roi.sum()), binary=int(binary.sum()),
                          morphed=int(morphed.sum())),
    )


if __name__ == "__main__":
    demo = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    out = mask_image(demo)
    print("masking smoke test OK:", out["pixel_counts"], out["params"])
