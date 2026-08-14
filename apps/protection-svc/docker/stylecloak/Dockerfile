# syntax=docker/dockerfile:1
#
# On-demand image for the L1_PREVIEW/L2_PORTFOLIO/L3_ANTI_TRAIN tiers
# (style_cloak.py), moved off the GPU-PC-over-SSH path (remote_gpu.py's
# remote_cloak/remote_compute_metrics/remote_upscale) onto RunPod
# Serverless for the same reason strong_protection already made this move
# (PHASE4_SCOPING.md §6): a personally-owned GPU PC reachable only over
# SSH is a single point of failure (has to be powered on, has to have a
# live network path) that Serverless removes by construction.
#
# Deliberately NOT the same image as docker/strongprotect/Dockerfile --
# that one is ~15GB with SD1.5+SDXL checkpoints baked in for the dual-arch
# attack, which style_cloak.py never touches (it only needs a VGG19
# feature extractor, see model.py's StyleFeatureExtractor). Sharing that
# image would mean L1/L2 -- meant to be this project's *fast* tiers
# (seconds to ~1 minute) -- cold-starting a 15GB pull on every worker
# spin-up, which would make them slower than strong_protection itself.
# Kept small on purpose.
#
# Build (from apps/protection-svc/docker/stylecloak/):
#   DOCKER_BUILDKIT=1 docker build -t lentropy/dontai-stylecloak:latest .
#   docker push lentropy/dontai-stylecloak:latest

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1/libxcb1: super-image's
# own dependency chain pulls in opencv-python (data/loader.py's `import
# cv2`), which dynamically links X11/GL client libs at import time even
# though this image never renders anything -- missing them fails with
# "ImportError: libxcb.so.1: cannot open shared object file", found live
# on the first real build of this Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip curl ca-certificates \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Exactly what style_cloak.py/evaluate.py/upscale.py import -- no
# diffusers/transformers/peft (those are strong_protection's dependencies,
# not this tier's). super-image is upscale.py's EDSR super-resolution
# model (restores real resolution after cloak()'s smaller processing
# size, same as orchestrate.py's local/GPU-PC paths already did).
# open_clip_torch: style_cloak.py's cloak() also runs a 2-model CLIP
# ensemble (get_clip_ensemble/ConceptFeatureExtractor, CLIP_ENSEMBLE_
# CHECKPOINTS) for its concept-transfer loss, not just VGG19 -- missing
# this failed every real job on the first live smoke test with
# "ModuleNotFoundError: No module named 'open_clip'" (this Dockerfile's
# own header comment was wrong about VGG19 being the only model needed).
RUN pip install --no-cache-dir --break-system-packages \
      torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir --break-system-packages \
      pillow numpy super-image fastapi open_clip_torch

# Pre-warm pretrained weights into the image at build time -- otherwise
# every cold worker start would hit the network for VGG19 (torchvision's
# own hub), EDSR (super-image's own HF download), and both CLIP ensemble
# checkpoints (open_clip's own HF download) on the first real job, adding
# latency and a runtime failure mode this tier's whole "fast and reliable"
# point is to avoid. torch's hub cache defaults to ~/.cache/torch, HF's to
# ~/.cache/huggingface -- both persist in this image layer since this RUN
# runs before any USER switch (root's home).
RUN python3 -c "import torchvision.models as m; m.vgg19(weights=m.VGG19_Weights.IMAGENET1K_V1)" && \
    python3 -c "from super_image import EdsrModel; EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2); EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)" && \
    python3 -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

# The default style-target asset every real L1/L2/L3 upload actually
# compares against (orchestrate.py's own style_target_path default) --
# same asset as docker/strongprotect's ml-engine/out/style_target.png
# copy, needed here too since style_cloak.py's whole objective is drift
# *toward* this image's style.
RUN mkdir -p /workspace/ml-engine/out
COPY style_target.png /workspace/ml-engine/out/style_target.png

RUN mkdir -p /workspace/dontai-protection-svc/ml-engine/src
COPY style_cloak.py model.py evaluate.py upscale.py /workspace/dontai-protection-svc/ml-engine/src/

RUN cd /workspace/dontai-protection-svc/ml-engine/src && \
    python3 -c "import style_cloak, model, evaluate, upscale; print('import OK')"
