from __future__ import annotations
import json
from pathlib import Path
import keras
from . import config, losses
from .data import DriveDataset, EspcnDataset, list_pairs
from .models import build_espcn, build_unet, compile_model


def _record(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def train_unet(images_glob, masks_glob, output, cfg: config.TrainConfig):
    keras.utils.set_random_seed(cfg.seed)
    out = Path(output)
    pairs = list_pairs(images_glob, masks_glob)
    dataset = DriveDataset(
        pairs, batch_size=cfg.batch_size, crops_per_image=cfg.crops_per_image,
        crop_size=cfg.crop_size, steps_per_epoch=cfg.steps_per_epoch,
        augment=cfg.augment, seed=cfg.seed)

    model = compile_model(build_unet(cfg.base_filters, cfg.dropout),
                          losses.resolve(cfg.loss), cfg.learning_rate)
    weights = out / "unet.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(weights), monitor="loss",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    out.mkdir(parents=True, exist_ok=True)
    history = model.fit(dataset, epochs=cfg.epochs, callbacks=callbacks, verbose=2)
    _record(out, dict(kind="unet", seed=cfg.seed, loss=cfg.loss,
                       base_filters=cfg.base_filters, dropout=cfg.dropout,
                       learning_rate=cfg.learning_rate, epochs=cfg.epochs,
                       steps_per_epoch=cfg.steps_per_epoch, n_images=len(pairs),
                       final_loss=float(history.history["loss"][-1]),
                       note="fresh seeded training"))
    return weights


def train_espcn(images_glob, output, cfg: config.TrainConfig):
    keras.utils.set_random_seed(cfg.seed)
    out = Path(output)
    image_paths = sorted(__import__("glob").glob(images_glob))
    if not image_paths:
        raise ValueError(f"No images matched {images_glob!r}")
    hr_size = cfg.scale * 32
    dataset = EspcnDataset(image_paths, scale=cfg.scale, hr_size=hr_size,
                           batch_size=cfg.batch_size, steps_per_epoch=cfg.steps_per_epoch,
                           seed=cfg.seed)
    model = compile_model(build_espcn(cfg.scale), keras.losses.MeanSquaredError(),
                          cfg.learning_rate)
    weights = out / "espcn.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(weights), monitor="loss",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    out.mkdir(parents=True, exist_ok=True)
    history = model.fit(dataset, epochs=cfg.epochs, callbacks=callbacks, verbose=2)
    _record(out, dict(kind="espcn", seed=cfg.seed, scale=cfg.scale,
                       learning_rate=cfg.learning_rate, epochs=cfg.epochs,
                       steps_per_epoch=cfg.steps_per_epoch, n_images=len(image_paths),
                       final_loss=float(history.history["loss"][-1]),
                       note="fresh seeded training"))
    return weights
