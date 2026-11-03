#!/usr/bin/env bash

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RETINA_REPO="$HERE"                 # the retina/ package is here
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3

PY="${PYTHON:-python}"
DATA="$HERE/data"
W="$HERE/weights"
UNET="$W/baseline_unet_aug_do_0.1_activation_ReLU_weights.best.hdf5"
ESPCN="$W/espcn_best.weights.h5"
OUT="$HERE/resultados_reproduccion"
mkdir -p "$OUT"

echo "========================================================================"
echo " Self-contained reproduction — everything resolved inside: $HERE"
echo " Python: $($PY --version 2>&1)"
echo "========================================================================"

echo ""
echo ">>> ITEM 4 — PSNR/MSE/SSIM per image (Set5 + Set14)"
$PY "$HERE/reproducir/compute_sr_metrics.py" \
    "$DATA/Set5" "$DATA/Set14" "$ESPCN" "$OUT"
echo "    -> $OUT/per_image_set5.csv, per_image_set14.csv"

echo ""
echo ">>> ITEM 5 — segmentation validation vs ground truth (DRIVE 21-40)"
$PY "$HERE/reproducir/compute_segmentation_metrics.py" \
    "$DATA/DRIVE_training/images" \
    "$DATA/DRIVE_training/1st_manual" \
    "$DATA/DRIVE_training/mask" \
    "$UNET" "$OUT"
echo "    -> $OUT/per_image_segmentation.csv, summary_segmentation.csv"

echo ""
echo ">>> ITEM 8 — per-stage runtime benchmark (self-contained)"
TIMING_OUT="$OUT" $PY "$HERE/reproducir/reproduce_timing.py"
echo "    -> $OUT/benchmark_tiempos_por_etapa.csv, RESUMEN_timing.json"

echo ""
echo "========================================================================"
echo " DONE. Results written to: $OUT"
echo "========================================================================"
