from __future__ import annotations
import keras
import numpy as np
import tensorflow as tf
from keras import layers


def build_unet(base_filters: int = 32, dropout: float = 0.1,
               activation: str = "relu", channels: int = 3) -> keras.Model:

    def conv_block(x, filters):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(dropout)(x)
        return x

    inputs = keras.Input((None, None, channels))
    skips, x = [], inputs
    widths = [base_filters * m for m in (1, 2, 4, 8)]
    for filters in widths:
        x = conv_block(x, filters)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    x = conv_block(x, base_filters * 16)

    for filters, skip in zip(reversed(widths), reversed(skips)):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = layers.Concatenate(axis=3)([x, skip])
        x = conv_block(x, filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="retina_unet")


def build_espcn(scale: int = 3) -> keras.Model:
    if scale < 2:
        raise ValueError("SR scale must be at least two")
    inputs = keras.Input((None, None, 1))
    x = inputs
    for i, (filters, kernel) in enumerate([(64, 5), (64, 3), (32, 3), (scale * scale, 3)]):
        x = layers.Conv2D(filters, kernel, activation="relu", padding="same",
                          kernel_initializer="orthogonal", name=f"sr_conv_{i}")(x)
    x = layers.Lambda(
        lambda a: tf.nn.depth_to_space(a, scale),
        output_shape=lambda s: (s[0], None, None, s[-1] // (scale * scale)),
        name="pixel_shuffle",
    )(x)
    return keras.Model(inputs, x, name="retina_espcn")


def compile_model(model: keras.Model, loss, learning_rate: float) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss, metrics=["accuracy"])
    return model


def enhance(rgb: np.ndarray, model: keras.Model, scale: int,
            mode: str = "y", iterations: int = 1) -> np.ndarray:
    from PIL import Image

    if mode not in {"y", "all"} or not 1 <= iterations <= 5:
        raise ValueError("Expected mode in {y, all} and 1..5 iterations")
    result = np.asarray(rgb, np.float32)
    for _ in range(iterations):
        h, w = result.shape[:2]
        ycc = Image.fromarray(np.uint8(np.clip(result * 255, 0, 255))).convert("YCbCr")
        channels, restored = list(ycc.split()), []
        for i, channel in enumerate(channels):
            if mode == "all" or i == 0:
                tensor = np.asarray(channel, np.float32)[None, ..., None] / 255.0
                values = model(tensor, training=False).numpy()[0, ..., 0]
                restored.append(Image.fromarray(np.uint8(np.clip(values * 255, 0, 255))))
            else:
                restored.append(channel.resize((w * scale, h * scale), Image.Resampling.BICUBIC))
        result = np.asarray(Image.merge("YCbCr", restored).convert("RGB"), np.float32) / 255.0
    return result
