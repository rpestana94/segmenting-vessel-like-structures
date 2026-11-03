#!/usr/bin/env bash

set -euo pipefail

# ---- EDIT HERE -------------------------------------------------------------
SEED="${SEED:-20240501}"   
EPOCHS="${EPOCHS:-100}"   
STEPS="${STEPS:-100}"     
LOSS="${LOSS:-bce}"       
# ----------------------------------------------------------------------------


HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DRIVE_DIR="$HERE/data/DRIVE_training"
IMAGES="${IMAGES:-$DRIVE_DIR/images/*.tif}"
MASKS="${MASKS:-$DRIVE_DIR/1st_manual/*.gif}"

OUT="$HERE/runs/unet_base_seed${SEED}"

echo "=== Reproducible U-Net base training ==="
echo "  seed    = $SEED"
echo "  epochs  = $EPOCHS  steps/epoch = $STEPS  loss = $LOSS"
echo "  images  = $IMAGES"
echo "  output  = $OUT"
echo ""

python -m retina.cli train \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --output "$OUT" \
  --epochs "$EPOCHS" --steps "$STEPS" --loss "$LOSS" --seed "$SEED"

echo ""
echo "=== Done. Produced in $OUT: ==="
echo "  unet.weights.h5   (seeded, best-loss checkpoint)"
echo "  history.csv       (per-epoch loss)"
echo "  metadata.json     (seed, loss, architecture, lr, epochs, final loss)"
echo ""

