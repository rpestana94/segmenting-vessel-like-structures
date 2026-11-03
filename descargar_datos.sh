#!/usr/bin/env bash
# Fetch the openly-redistributable SR benchmarks (Set5, Set14) into data/.
# DRIVE and HRF require manual download (registration / official site) — see their READMEs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/data"

fetch() {  # $1 = dataset name, $2 = HR tarball URL
  local name="$1" url="$2"
  echo ">>> $name"
  if [ -d "$name" ] && [ -n "$(ls -A "$name" 2>/dev/null | grep -v README || true)" ]; then
    echo "    already present, skipping (delete data/$name to re-fetch)"; return
  fi
  mkdir -p "$name"; local tgz="$name/_hr.tar.gz"
  curl -fSL "$url" -o "$tgz"
  tar -xzf "$tgz" -C "$name" --strip-components=1 2>/dev/null || tar -xzf "$tgz" -C "$name"
  rm -f "$tgz"
  echo "    done ($(ls "$name" | grep -v README | wc -l) files)"
}

fetch Set5  "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz"
fetch Set14 "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz"

cat <<'EOF'

Set5 + Set14 ready (enough for Item 4).
DRIVE (Items 5, 8): register at https://drive.grand-challenge.org/ and place files
    per data/DRIVE_training/README.md
HRF (optional): https://www5.cs.fau.de/research/data/fundus-images/
    per data/HRF/README.md
Verify any download against the SHA-256 tables in each folder's README.md.
EOF
