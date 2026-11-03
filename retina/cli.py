from __future__ import annotations
import argparse
from glob import glob
from . import config


def _load_unet(weights, base_filters, dropout):
    from .models import build_unet
    model = build_unet(base_filters, dropout)
    model.load_weights(weights)
    return model


def _load_espcn(weights, scale):
    from .models import build_espcn
    model = build_espcn(scale)
    model.load_weights(weights)
    return model


def _zip_globs(*globs):
    lists = [sorted(glob(g)) if g else None for g in globs]
    present = [lst for lst in lists if lst is not None]
    if not present or any(len(lst) != len(present[0]) for lst in present):
        raise ValueError("Glob result counts differ")
    n = len(present[0])
    return [tuple(lst[i] if lst is not None else None for lst in lists) for i in range(n)]


def cmd_train(args):
    from .train import train_unet
    cfg = config.TrainConfig(kind="unet", epochs=args.epochs, steps_per_epoch=args.steps,
                             loss=args.loss, augment=not args.no_augment, seed=args.seed)
    weights = train_unet(args.images, args.masks, args.output, cfg)
    print(f"U-Net weights written to {weights}")


def cmd_train_sr(args):
    from .train import train_espcn
    cfg = config.TrainConfig(kind="espcn", epochs=args.epochs, steps_per_epoch=args.steps,
                             learning_rate=config.SR_LEARNING_RATE, scale=args.scale, seed=args.seed)
    weights = train_espcn(args.images, args.output, cfg)
    print(f"ESPCN weights written to {weights}")


def cmd_seg_eval(args):
    from .predict import evaluate_segmentation
    model = _load_unet(args.weights, args.base_filters, args.dropout)
    pairs = _zip_globs(args.images, args.masks, args.fov)
    summary = evaluate_segmentation(model, pairs, args.output, threshold=args.threshold,
                                    apply_masking=args.mask)
    print("Segmentation summary:", summary)


def cmd_sr_eval(args):
    from .predict import evaluate_super_resolution
    model = _load_espcn(args.weights, args.scale)
    image_paths = sorted(glob(args.images))
    summary = evaluate_super_resolution(model, image_paths, args.output,
                                        scale=args.scale, crop=args.crop, channel=args.channel)
    print("SR summary rows:", len(summary))


def cmd_mask(args):
    import numpy as np
    from PIL import Image
    from .masking import mask_image
    from pathlib import Path
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    result = mask_image(image, threshold=args.threshold)
    for stage in ("masked_gray", "binary", "morphed"):
        arr = result[stage]
        Image.fromarray(np.uint8(arr if arr.max() > 1 else arr * 255)).save(out / f"{stage}.png")
    print("Masking pixel counts:", result["pixel_counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train the U-Net segmenter")
    p.add_argument("--images", required=True); p.add_argument("--masks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--steps", type=int, default=100)
    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-sr", help="train the ESPCN super-resolver")
    p.add_argument("--images", required=True); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scale", type=int, default=config.SR_SCALE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.set_defaults(func=cmd_train_sr)

    p = sub.add_parser("seg-eval", help="evaluate segmentation on a labelled set")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True); p.add_argument("--masks", required=True)
    p.add_argument("--fov", default=None); p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mask", action="store_true", help="apply masking post-processing")
    p.add_argument("--base-filters", type=int, default=config.BASE_FILTERS)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.set_defaults(func=cmd_seg_eval)

    p = sub.add_parser("sr-eval", help="evaluate super-resolution")
    p.add_argument("--weights", required=True); p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=int, default=config.SR_SCALE)
    p.add_argument("--crop", type=int, default=3)
    p.add_argument("--channel", choices=["rgb", "y"], default="y")
    p.set_defaults(func=cmd_sr_eval)

    p = sub.add_parser("mask", help="run masking post-processing on one image")
    p.add_argument("--image", required=True); p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=int, default=config.GRAYSCALE_THRESHOLD)
    p.set_defaults(func=cmd_mask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
