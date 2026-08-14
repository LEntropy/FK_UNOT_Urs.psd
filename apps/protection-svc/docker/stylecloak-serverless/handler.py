"""RunPod Serverless handler for the L1_PREVIEW/L2_PORTFOLIO/L3_ANTI_TRAIN
style_cloak.py tier -- replaces remote_gpu.py's SSH-to-a-GPU-PC path
(remote_cloak/remote_compute_metrics/remote_upscale) the same way
docker/strongprotect-serverless/handler.py already replaced the dual-arch
attack's own SSH path (PHASE4_SCOPING.md §6).

Three actions, matching the three steps orchestrate.py's protect() already
runs in sequence for every real L1/L2/L3 upload:

- "cloak": style_cloak.py's cloak() -- the actual protection step.
- "compute_metrics": evaluate.py's compute_protection_metrics() --
  styleDriftScore/perceptualPsnrDb/styleSimilarityToOriginal, the numbers
  shown on the artwork page and Test Lab's "보호 강도 테스트" tab.
- "upscale": upscale.py's upscale_to_size() -- restores the real upload's
  original resolution after cloak() processed a smaller square (see
  orchestrate.py's MAX_PROCESSING_SIZE comment for why cloak() doesn't
  just run at full resolution directly).

All three run in-process (not subprocess, unlike hybrid_protect.py's own
handler) -- there's no multi-model VRAM-isolation concern here the way
there is chaining SD1.5 then SDXL attacks; this is always one model
(VGG19 for cloak/metrics, EDSR for upscale) per job, and RunPod's own
worker lifecycle handles teardown between jobs.
"""

import base64
import os
import sys
import tempfile

import runpod

ML_SRC = "/workspace/dontai-protection-svc/ml-engine/src"
STYLE_TARGET = "/workspace/ml-engine/out/style_target.png"

sys.path.insert(0, ML_SRC)
from style_cloak import cloak  # noqa: E402
from evaluate import compute_protection_metrics  # noqa: E402
from upscale import upscale_to_size  # noqa: E402


def _run_cloak(job_input: dict, tmp: str) -> dict:
    original_path = os.path.join(tmp, "original.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(job_input["original_b64"]))

    style_target_path = STYLE_TARGET
    if job_input.get("style_target_b64"):
        style_target_path = os.path.join(tmp, "style_target.png")
        with open(style_target_path, "wb") as f:
            f.write(base64.b64decode(job_input["style_target_b64"]))

    output_path = os.path.join(tmp, "cloaked.png")
    cloak(
        original_path=original_path,
        style_target_path=style_target_path,
        output_path=output_path,
        preset_name=job_input["preset_name"],
        size=job_input.get("size", 256),
        eot=job_input.get("eot", False),
        eot_samples=job_input.get("eot_samples", 2),
        perceptual_mask=job_input.get("perceptual_mask", False),
        use_amp=job_input.get("use_amp", False),
    )

    with open(output_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode()
    return {"output_b64": output_b64}


def _run_compute_metrics(job_input: dict, tmp: str) -> dict:
    original_path = os.path.join(tmp, "original.png")
    cloaked_path = os.path.join(tmp, "cloaked.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(job_input["original_b64"]))
    with open(cloaked_path, "wb") as f:
        f.write(base64.b64decode(job_input["cloaked_b64"]))

    style_target_path = STYLE_TARGET
    if job_input.get("style_target_b64"):
        style_target_path = os.path.join(tmp, "style_target.png")
        with open(style_target_path, "wb") as f:
            f.write(base64.b64decode(job_input["style_target_b64"]))

    return compute_protection_metrics(
        original_path=original_path,
        cloaked_path=cloaked_path,
        style_target_path=style_target_path,
        size=job_input.get("size", 256),
    )


def _run_upscale(job_input: dict, tmp: str) -> dict:
    input_path = os.path.join(tmp, "input.png")
    with open(input_path, "wb") as f:
        f.write(base64.b64decode(job_input["input_b64"]))

    output_path = os.path.join(tmp, "upscaled.png")
    used_sr = upscale_to_size(
        input_path=input_path,
        output_path=output_path,
        target_width=job_input["target_width"],
        target_height=job_input["target_height"],
    )

    with open(output_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode()
    return {"output_b64": output_b64, "used_sr": used_sr}


def handler(job):
    job_input = job["input"]
    action = job_input.get("action")

    with tempfile.TemporaryDirectory() as tmp:
        if action == "compute_metrics":
            return _run_compute_metrics(job_input, tmp)
        if action == "upscale":
            return _run_upscale(job_input, tmp)
        if action == "cloak":
            return _run_cloak(job_input, tmp)
        raise ValueError(f"unknown action {action!r} -- expected 'cloak', 'compute_metrics', or 'upscale'")


runpod.serverless.start({"handler": handler})
