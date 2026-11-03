#!/usr/bin/env python3

from __future__ import annotations
import csv, json, os, platform, sys, time, hashlib, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RETINA_REPO", os.path.join(HERE, ".."))   
sys.path.insert(0, os.path.abspath(REPO))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DATA   = os.path.join(os.path.abspath(REPO), "data")
WEIGHTS= os.path.join(os.path.abspath(REPO), "weights")
DRIVE  = os.path.join(DATA, "DRIVE_training", "images")
CKPT   = os.path.join(WEIGHTS, "baseline_unet_aug_do_0.1_activation_ReLU_weights.best.hdf5")
ESPCN_W= os.path.join(WEIGHTS, "espcn_best.weights.h5")
OUT    = os.environ.get("TIMING_OUT", os.path.join(HERE, "resultados_timing"))
os.makedirs(OUT, exist_ok=True)

SIDE   = 576
WARMUP = int(os.environ.get("WARMUP", "3"))
REPS   = int(os.environ.get("REPS", "20"))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def timeit(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    a = np.array(ts)
    return dict(mean_ms=round(float(a.mean()), 3), std_ms=round(float(a.std(ddof=1)), 3),
                min_ms=round(float(a.min()), 3), max_ms=round(float(a.max()), 3),
                reps=reps, warmup=warmup)

def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow.keras.utils import img_to_array
    from PIL import Image
    import PIL.Image
    from retina.models import build_unet, build_espcn
    from retina.masking import mask_image


    ck = sha256(CKPT)
    assert ck.startswith("4700217f"), f"unexpected U-Net checkpoint sha {ck[:16]}"
    unet = build_unet(base_filters=32, dropout=0.1, channels=3); unet.load_weights(CKPT)
    espcn = build_espcn(scale=3); espcn.load_weights(ESPCN_W)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"checkpoint {ck[:16]} | device={'GPU' if gpus else 'CPU'}", flush=True)


    ip = sorted(glob.glob(f"{DRIVE}/*.tif"))[0]
    img = Image.open(ip).convert("RGB"); W, H = img.size
    w3, h3 = W - W % 3, H - H % 3
    lr = img.crop((0, 0, w3, h3)).resize((w3 // 3, h3 // 3), Image.BICUBIC)


    def stage_sr():
        y, cb, cr = lr.convert("YCbCr").split()
        yy = img_to_array(y).astype("float32") / 255.
        o = espcn(np.expand_dims(yy, 0), training=False).numpy()[0] * 255.
        o = o.clip(0, 255).reshape(o.shape[0], o.shape[1])
        oy = PIL.Image.fromarray(np.uint8(o), "L")
        return PIL.Image.merge("YCbCr", (oy,
                                         cb.resize(oy.size, PIL.Image.BICUBIC),
                                         cr.resize(oy.size, PIL.Image.BICUBIC))).convert("RGB")
    sr_img = stage_sr()
    def stage_resize(): return np.asarray(sr_img.resize((SIDE, SIDE), Image.BILINEAR), np.float32) / 255.
    x576 = stage_resize()
    def stage_unet(): return unet(x576[None], training=False).numpy()[0, ..., 0]
    prob = stage_unet()
    seg_full = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.uint8)
    def stage_post(): return mask_image(seg_full)

    stages = [("1_super_resolucion", stage_sr), ("2_input_resize", stage_resize),
              ("3_segmentacion_unet", stage_unet), ("4_post_procesamiento", stage_post)]
    rows = []
    for name, fn in stages:
        r = timeit(fn); r["stage"] = name; rows.append(r)
        print(f"  {name:24s} {r['mean_ms']:9.3f} +/- {r['std_ms']:7.3f} ms (n={r['reps']}, warmup={r['warmup']})", flush=True)
    total = sum(r["mean_ms"] for r in rows)
    print(f"  {'TOTAL (pipeline with SR)':24s} {total:9.3f} ms", flush=True)

    cols = ["stage", "mean_ms", "std_ms", "min_ms", "max_ms", "reps", "warmup"]
    with open(f"{OUT}/benchmark_tiempos_por_etapa.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in cols})
    env = dict(units="milliseconds (ms); 1 ms = 1e-3 s",
               note="Per-stage wall time, warm-up + repetitions, one representative DRIVE image. "
                    "Self-contained: retina package + internal weights/data.",
               checkpoint=ck, device=("GPU" if gpus else "CPU"),
               python=platform.python_version(), tensorflow=tf.__version__,
               processor=platform.processor() or platform.machine(),
               cpu_count=os.cpu_count(), platform=platform.platform(),
               warmup=WARMUP, reps=REPS, total_pipeline_ms=round(total, 3),
               per_stage={r["stage"]: r["mean_ms"] for r in rows})
    json.dump(env, open(f"{OUT}/RESUMEN_timing.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {OUT}/ (benchmark_tiempos_por_etapa.csv, RESUMEN_timing.json)")

if __name__ == "__main__":
    main()
