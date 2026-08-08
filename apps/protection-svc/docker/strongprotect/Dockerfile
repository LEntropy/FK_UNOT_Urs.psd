# syntax=docker/dockerfile:1
#
# On-demand image for PHASE4_SCOPING.md §6's strong_protection tier.
#
# Deliberately NOT based on runpod/pytorch (this project's own experiment
# pods all used that image) -- built and measured at 49.4GB, almost
# entirely jupyter/filebrowser/multi-Python-version (3.9-3.13) tooling
# this headless single-purpose worker never touches. Also deliberately
# does NOT clone kohya_ss/sd-scripts (unlike this project's *validation*
# pods, which needed train_network.py/sdxl_train_network.py for real LoRA
# training) -- multiarch_ensemble_attack() itself only ever imports torch/
# torchvision/diffusers/peft/PIL (see ensemble_attack_multiarch.py and its
# style_cloak.py/model.py dependencies), never sd-scripts. Production only
# runs the attack, never the LoRA-training validation pipeline, so kohya's
# ~1.5GB clone + its gradio/tensorboard/onnx GUI dependencies were pure
# dead weight here.
#
# Build (from apps/protection-svc/docker/strongprotect/):
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=civitai_token,env=CIVITAI_TOKEN \
#     -t lentropy/dontai-strongprotect:latest .
#   docker push lentropy/dontai-strongprotect:latest
#
# CIVITAI_TOKEN via BuildKit's --secret mount (not ARG/ENV) so it never
# lands in an image layer or `docker history` -- a plain ARG would leak it
# to anyone who pulls the (public) image.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Minimal SSH setup matching RunPod's PUBLIC_KEY convention -- replicated
# by hand since dropping the runpod/pytorch base (which did this via its
# own /start.sh) also dropped this. remote_multiarch_cloak() dispatches
# over SSH the same way as every other pod this project has used.
RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server python3 python3-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

RUN printf '#!/bin/bash\nmkdir -p ~/.ssh\nif [ -n "$PUBLIC_KEY" ]; then\n  echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys\n  chmod 600 ~/.ssh/authorized_keys\n  chmod 700 ~/.ssh\nfi\nssh-keygen -A\nexec /usr/sbin/sshd -D\n' > /start.sh \
    && chmod +x /start.sh
CMD ["/start.sh"]

WORKDIR /workspace

# The exact dependency set multiarch_ensemble_attack() actually imports --
# not kohya_ss's full requirements.txt (gradio/tensorboard/onnx/wandb,
# none of which a headless attack-only worker touches). Pinned to the
# same diffusers/transformers versions this project's validation runs
# used (PHASE4_SCOPING.md §6), torch to the cu128 build every experiment
# pod this session used. --break-system-packages: Ubuntu 24.04's system
# pip refuses unmanaged installs (PEP 668) otherwise -- fine here, this
# image has exactly one Python environment, no conflicting system tooling
# to protect.
RUN pip install --no-cache-dir --break-system-packages \
      torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir --break-system-packages \
      diffusers==0.32.2 transformers==4.54.1 peft pillow numpy

# Checkpoints -- SD1.5 is a public HF mirror (no auth), SDXL is
# Illustrious-XL from Civitai (gated behind an account, needs the token).
# --http1.1 --retry 5 --retry-all-errors: a multi-GB transfer inside a
# Docker build hit a transient HTTP/2 stream error (curl exit 92) once
# during this project's own build -- these two flags avoid depending on
# HTTP/2 framing surviving a large single-connection transfer and retry
# through any other transient network blip instead of failing the build.
RUN mkdir -p /workspace/checkpoints && \
    curl -sL --http1.1 --retry 5 --retry-all-errors -o /workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors \
      "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors"

RUN --mount=type=secret,id=civitai_token \
    curl -sL --http1.1 --retry 5 --retry-all-errors -H "Authorization: Bearer $(cat /run/secrets/civitai_token)" \
      -o /workspace/checkpoints/Illustrious-XL-v0.1.safetensors \
      "https://civitai.com/api/download/models/889818"

# ensemble_attack_multiarch.py's own two deps (model.py for style_cloak.py's
# top-level import), plus model_leak_detect.py (detection-svc's model-leak
# report pathway, remote_gpu.py's remote_detect_model_leak) -- it only
# imports diffusers/transformers/PIL/torch, all already installed above,
# no new dependency needed for it. aspl_attack.py (SD1.5) and
# aspl_attack_sdxl_only.py (SDXL, reuses ensemble_attack_multiarch.py's
# SDXLBranch) are the two single-architecture pixel-space attacks.
# aspl_attack_latent.py / aspl_attack_sdxl_only_latent.py (2026-08-08,
# same reuse pattern in latent space) and hybrid_protect.py (2026-08-08)
# are the four-stage latent-then-pixel composition strong_protection now
# runs -- see hybrid_protect.py's own module doc for why stage order and
# per-stage epsilon are load-bearing, and its STATUS note: this is
# n=1-validated (SD1.5 only), not the n=30+replication bar every other
# mechanism in this codebase cleared before shipping. Still not the whole
# ml-engine tree. protection_score.py (Test Lab's on-demand real-LoRA-
# training score) reuses aspl_attack.py's ASPLAttacker and
# ensemble_attack_multiarch.py's SDXLBranch directly -- same reasoning,
# no new dependency (peft/transformers/diffusers all already installed
# above for the attack scripts).
RUN mkdir -p /workspace/dontai-protection-svc/ml-engine/src
COPY ensemble_attack_multiarch.py style_cloak.py model.py model_leak_detect.py aspl_attack.py aspl_attack_sdxl_only.py aspl_attack_latent.py aspl_attack_sdxl_only_latent.py hybrid_protect.py protection_score.py /workspace/dontai-protection-svc/ml-engine/src/

# Sanity checks baked into the build itself -- fail the build (not a later
# pod boot) if a checkpoint download silently truncated, or the attack
# module can't even import in this trimmed dependency set.
RUN sd15_size=$(stat -c%s /workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors) && \
    sdxl_size=$(stat -c%s /workspace/checkpoints/Illustrious-XL-v0.1.safetensors) && \
    echo "SD1.5: ${sd15_size} bytes, SDXL: ${sdxl_size} bytes" && \
    [ "$sd15_size" -gt 2000000000 ] && [ "$sdxl_size" -gt 6000000000 ]
RUN cd /workspace/dontai-protection-svc/ml-engine/src && \
    python3 -c "import ensemble_attack_multiarch, aspl_attack, aspl_attack_sdxl_only, aspl_attack_latent, aspl_attack_sdxl_only_latent, hybrid_protect, protection_score; print('import OK')"
