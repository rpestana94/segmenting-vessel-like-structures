from __future__ import annotations
from glob import glob
from pathlib import Path
import keras
import numpy as np
from PIL import Image
from scipy import ndimage


def read_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def read_mask(path) -> np.ndarray:
    arr = np.asarray(Image.open(path), np.float32)
    arr = arr / (arr.max() if arr.max() > 0 else 1.0)
    return arr[..., None]


def list_pairs(images_glob: str, masks_glob: str) -> list[tuple[str, str]]:
    images, masks = sorted(glob(images_glob)), sorted(glob(masks_glob))
    if not images or len(images) != len(masks):
        raise ValueError(f"Image/mask count mismatch: {len(images)} vs {len(masks)}")
    return list(zip(images, masks))


class RetinaAugmentor:

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def _geometric(self, img, mask):
        r = self.rng

        if r.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if r.random() < 0.5:
            img, mask = img[::-1], mask[::-1]

        if r.random() < 0.05:
            theta = r.uniform(-10, 10)
            img = ndimage.rotate(img, theta, axes=(0, 1), reshape=False, order=1, mode="nearest")
            mask = ndimage.rotate(mask, theta, axes=(0, 1), reshape=False, order=0, mode="nearest")

        if r.random() < 0.05:
            h, w = img.shape[:2]
            sh = (r.uniform(-0.1, 0.1) * h, r.uniform(-0.1, 0.1) * w)
            img = ndimage.shift(img, (sh[0], sh[1], 0), order=1, mode="nearest")
            mask = ndimage.shift(mask, (sh[0], sh[1], 0), order=0, mode="nearest")

        if r.random() < 0.05:
            z = r.uniform(0.9, 1.1)
            img = self._zoom(img, z, order=1)
            mask = self._zoom(mask, z, order=0)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    @staticmethod
    def _zoom(x, factor, order):
        h, w = x.shape[:2]
        centre = np.array([h / 2.0, w / 2.0])
        offset = centre - centre / factor
        matrix = np.diag([1.0 / factor, 1.0 / factor])  
        out = np.empty_like(x)
        for c in range(x.shape[2]):
            out[..., c] = ndimage.affine_transform(
                x[..., c], matrix, offset=offset, order=order, mode="nearest")
        return out

    def _photometric(self, img):
        r = self.rng
        luma = np.array([0.114, 0.587, 0.299], np.float32)
        if r.random() < 0.05: 
            img = np.clip(img * (1.0 + r.uniform(-0.1, 0.1)), 0.0, 1.0)
        if r.random() < 0.05:  
            alpha = 1.0 + r.uniform(-0.1, 0.1)
            gray = (img * luma).sum(axis=2, keepdims=True)
            img = np.clip(alpha * img + (1.0 - alpha) * gray, 0.0, 1.0)
        if r.random() < 0.05:  
            alpha = 1.0 + r.uniform(-0.1, 0.1)
            gray = (img * luma).sum() * 3.0 * (1.0 - alpha) / img.size
            img = np.clip(alpha * img + gray, 0.0, 1.0)
        return img

    def __call__(self, img, mask):
        img, mask = self._geometric(img, mask)
        return self._photometric(img), mask


class DriveDataset(keras.utils.PyDataset):


    def __init__(self, pairs, *, batch_size, crops_per_image, crop_size,
                 steps_per_epoch, augment, seed, **kwargs):
        super().__init__(**kwargs)
        if batch_size % crops_per_image:
            raise ValueError("batch_size must be a multiple of crops_per_image")
        self.pairs = pairs
        self.batch_size = batch_size
        self.crops_per_image = crops_per_image
        self.crop_size = crop_size
        self.steps_per_epoch = steps_per_epoch
        self.augment = RetinaAugmentor(seed) if augment else None
        self.rng = np.random.default_rng(seed + 1)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return self.steps_per_epoch

    def _load(self, index):
        if index not in self._cache:
            img_path, mask_path = self.pairs[index]
            self._cache[index] = (read_rgb(img_path), read_mask(mask_path))
        return self._cache[index]

    def _random_crop(self, img, mask):
        c = self.crop_size
        i = self.rng.integers(0, img.shape[0] - c + 1)
        j = self.rng.integers(0, img.shape[1] - c + 1)
        return img[i:i + c, j:j + c], mask[i:i + c, j:j + c]

    def __getitem__(self, _):
        xs, ys = [], []
        n_images = self.batch_size // self.crops_per_image
        for index in self.rng.integers(0, len(self.pairs), n_images):
            img, mask = self._load(int(index))
            if self.augment is not None:
                img, mask = self.augment(img.copy(), mask.copy())
            for _ in range(self.crops_per_image):
                patch, label = self._random_crop(img, mask)
                xs.append(patch)
                ys.append(label)
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


class EspcnDataset(keras.utils.PyDataset):

    def __init__(self, image_paths, *, scale, hr_size, batch_size,
                 steps_per_epoch, seed, **kwargs):
        super().__init__(**kwargs)
        if hr_size % scale:
            raise ValueError("hr_size must be divisible by scale")
        self.image_paths = image_paths
        self.scale = scale
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return self.steps_per_epoch

    def _luma(self, index):
        if index not in self._cache:
            img = Image.open(self.image_paths[index]).convert("YCbCr").getchannel("Y")
            self._cache[index] = np.asarray(img, np.float32)
        return self._cache[index]

    def __getitem__(self, _):
        xs, ys = [], []
        c = self.hr_size
        for index in self.rng.integers(0, len(self.image_paths), self.batch_size):
            y = self._luma(int(index))
            i = self.rng.integers(0, y.shape[0] - c + 1)
            j = self.rng.integers(0, y.shape[1] - c + 1)
            hr = Image.fromarray(np.uint8(y[i:i + c, j:j + c]))
            lr = hr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
            xs.append(np.asarray(lr, np.float32)[..., None] / 255.0)
            ys.append(np.asarray(hr, np.float32)[..., None] / 255.0)
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32)
