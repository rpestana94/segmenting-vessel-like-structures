from __future__ import annotations
import keras
from keras import ops

_EPS = 1e-7


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):

    def loss(y_true, y_pred):
        y_true = ops.cast(y_true, y_pred.dtype)
        p = ops.clip(y_pred, _EPS, 1.0 - _EPS)
        pos = -alpha * ops.power(1.0 - p, gamma) * ops.log(p)
        neg = -(1.0 - alpha) * ops.power(p, gamma) * ops.log(1.0 - p)
        return ops.sum(ops.where(ops.equal(y_true, 1.0), pos, neg))

    loss.__name__ = "focal_loss_fixed"
    return loss


def resolve(name: str):
    if name == "bce":
        return keras.losses.BinaryCrossentropy()
    if name == "focal":
        return focal_loss()
    raise ValueError(f"Unknown loss: {name!r} (expected 'bce' or 'focal')")
