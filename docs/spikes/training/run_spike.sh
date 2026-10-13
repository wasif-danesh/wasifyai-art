#!/usr/bin/env bash
# LoRA training spike on the dev Mac (mflux, MLX). NOT run by anything automatically: the first run
# downloads black-forest-labs/FLUX.2-klein-base-4B (about 15 GB, Apache-2.0, not gated) into the Hugging Face
# cache. Prerequisite: data/spikes/pipbot (make_dataset.py).
#
#   docs/spikes/training/run_spike.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VENV="${MFLUX_VENV:-$HOME/.cache/wasifyai-art/mflux-env}"
OUT="${TRAIN_OUT:-$REPO/data/spikes/train-out}"
[ -d "$REPO/data/spikes/pipbot" ] || { echo "dataset missing: run make_dataset.py first" >&2; exit 1; }
if [ ! -x "$VENV/bin/mflux-train" ]; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" mflux
fi
mkdir -p "$OUT"
sed -e "s|__DATA__|$REPO/data/spikes/pipbot|" -e "s|__OUT__|$OUT|" "$HERE/train_klein_base_4b.json" > "$OUT/train.json"
echo "config: $OUT/train.json"
# The Hugging Face "xet" transfer backend stalled to ~0 KB/s on the dev connection (measured); the plain
# HTTP path ran at ~9 MB/s. Resumable either way.
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
# mflux writes to a timestamped sibling folder: "$OUT_<date>_<time>/{checkpoints,preview,loss}"
exec "$VENV/bin/mflux-train" --config "$OUT/train.json"
