#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"


PY="${PYTHON:-python}"
$PY -c "import keras,sys; v=int(keras.__version__.split('.')[0]); sys.exit(0 if v>=3 else 1)" 2>/dev/null || {
  echo "ERROR: este pipeline requiere Keras >= 3"
  echo "       Instala con:  pip install -r requirements.txt"
  exit 1
}


HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$HERE/data/DRIVE_training}"
SR_DIR="${SR_DIR:-$HERE/data/Set5}"
OUT="${2:-$HERE/demo_out}"
UNET="$HERE/weights/baseline_unet_aug_do_0.1_activation_ReLU_weights.best.hdf5"
ESPCN="$HERE/weights/espcn_best.weights.h5"
PY="${PYTHON:-python}"

echo "DATA_DIR=$DATA_DIR | SR_DIR=$SR_DIR | OUT=$OUT"
mkdir -p "$OUT"

echo ""; echo "[1/4] Segmentación U-Net"
$PY -m retina.cli seg-eval --weights "$UNET" \
    --images "$DATA_DIR/images/*.tif" --masks "$DATA_DIR/1st_manual/*.gif" --fov "$DATA_DIR/mask/*.gif" \
    --output "$OUT/seg_no_mask" --threshold 0.5

echo ""; echo "[2/4] Segmentación U-Net"
$PY -m retina.cli seg-eval --weights "$UNET" \
    --images "$DATA_DIR/images/*.tif" --masks "$DATA_DIR/1st_manual/*.gif" --fov "$DATA_DIR/mask/*.gif" \
    --output "$OUT/seg_with_mask" --threshold 0.5 --mask

echo ""; echo "[3/4] Super-resolución ESPCN"
$PY -m retina.cli sr-eval --weights "$ESPCN" \
    --images "$SR_DIR/*_HR.png" --output "$OUT/sr_eval" --scale 3 --channel y

echo ""; echo "[4/4] Masking"
FIRST=$(ls $DATA_DIR/images/*.tif | head -1)
$PY -m retina.cli mask --image "$FIRST" --output "$OUT/mask_demo" --threshold 100

echo ""; echo "=== LISTO. Resultados en $OUT/ ==="
