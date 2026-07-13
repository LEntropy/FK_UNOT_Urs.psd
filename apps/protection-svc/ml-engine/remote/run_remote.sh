#!/usr/bin/env bash
# Syncs code + input images to the GPU PC over SSH, runs style_cloak.py
# there (CUDA torch), then pulls the resulting image back to this machine's
# out/ directory. Requires SETUP_GPU_PC.md to have been done once, and
# remote.env to exist (copy remote.env.example -> remote.env and fill in).
#
# Usage: ./run_remote.sh [PRESET] [OUTPUT_FILENAME]
#   ./run_remote.sh L3_ANTI_TRAIN cloaked_gpu.png

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ ! -f remote.env ]; then
  echo "remote.env not found. Copy remote.env.example to remote.env and fill in GPU_HOST/GPU_USER first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source remote.env

PRESET="${1:-L3_ANTI_TRAIN}"
OUTPUT_NAME="${2:-cloaked_gpu.png}"
ENGINE_DIR=".."
REMOTE="$GPU_USER@$GPU_HOST"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

echo "==> [1/5] ensuring remote directories exist"
ssh "${SSH_OPTS[@]}" "$REMOTE" \
  "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '$GPU_REMOTE_DIR' | Out-Null; New-Item -ItemType Directory -Force -Path '$GPU_REMOTE_DIR/out' | Out-Null\""

echo "==> [2/5] syncing code + input images"
scp "${SSH_OPTS[@]}" -r "$ENGINE_DIR/src" "$ENGINE_DIR/scripts" "$ENGINE_DIR/requirements.txt" "$REMOTE:$GPU_REMOTE_DIR/"
if [ -f "$ENGINE_DIR/out/original.png" ]; then
  scp "${SSH_OPTS[@]}" "$ENGINE_DIR/out/original.png" "$ENGINE_DIR/out/style_target.png" "$REMOTE:$GPU_REMOTE_DIR/out/"
fi

echo "==> [3/5] ensuring remote venv + CUDA torch (first run downloads ~2-3GB, be patient)"
# cu128 is required for Blackwell (RTX 50-series, sm_120) GPUs — cu121/cu124
# builds don't include sm_120 kernels and fail at runtime with
# "CUDA error: no kernel image is available for execution on the device"
# even though torch.cuda.is_available() reports True. If your GPU is older,
# cu121 is smaller/faster to install and works fine — check `nvidia-smi` /
# your GPU's compute capability before downgrading this.
ssh "${SSH_OPTS[@]}" "$REMOTE" "powershell -NoProfile -Command \"\
cd '$GPU_REMOTE_DIR'; \
if (-not (Test-Path .venv)) { python -m venv .venv }; \
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip -q; \
.\\.venv\\Scripts\\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision -q; \
.\\.venv\\Scripts\\python.exe -m pip install Pillow numpy -q\""

echo "==> [4/5] generating test images on remote (only if missing) + running style_cloak.py on GPU"
ssh "${SSH_OPTS[@]}" "$REMOTE" "powershell -NoProfile -Command \"\
cd '$GPU_REMOTE_DIR'; \
if (-not (Test-Path out/original.png)) { .\\.venv\\Scripts\\python.exe scripts/generate_test_images.py }; \
.\\.venv\\Scripts\\python.exe src/style_cloak.py --preset $PRESET --output out/$OUTPUT_NAME\""

echo "==> [5/5] pulling result back"
mkdir -p "$ENGINE_DIR/out"
scp "${SSH_OPTS[@]}" "$REMOTE:$GPU_REMOTE_DIR/out/$OUTPUT_NAME" "$ENGINE_DIR/out/$OUTPUT_NAME"

echo "==> done: $ENGINE_DIR/out/$OUTPUT_NAME"
