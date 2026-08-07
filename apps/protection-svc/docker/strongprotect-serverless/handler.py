"""RunPod Serverless handler wrapping two jobs that reuse the same
checkpoints and attack scripts baked into lentropy/dontai-strongprotect:
latest (this image is FROM that one, see the sibling Dockerfile):

- "dual_arch_cloak" (default, unset action): the same dual-arch attack
  remote_dual_arch_cloak() dispatches over SSH -- built to empirically
  compare RunPod Serverless cost/reliability against the Pods-based
  approach (PHASE4_SCOPING.md SS6's 2026-08-07 update).
- "score_protection": the Test Lab's on-demand real-LoRA-training
  protection score (see protection_score.py's module doc) -- trains
  baseline-vs-protected LoRAs for a single artwork and CLIP-scores the
  result, run once per user request rather than n=30 for a statistical
  claim.

Both run as a subprocess (not an in-process import) for the same reason
the original dual_arch_cloak job did: each of aspl_attack.py/
aspl_attack_sdxl_only.py/protection_score.py loads its own full pipeline
into VRAM and expects to have it torn down cleanly on exit -- a fresh
process per job is the simplest way to guarantee no state (or VRAM)
leaks between RunPod job invocations sharing one worker.
"""

import base64
import json
import os
import subprocess
import tempfile

import runpod

SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
ML_SRC = "/workspace/dontai-protection-svc/ml-engine/src"


def _run_dual_arch_cloak(job_input: dict, tmp: str) -> dict:
    image_b64 = job_input["image_b64"]
    prompt = job_input.get("prompt", "artwork")
    sd15_preset = job_input.get("sd15_preset", "L3_ANTI_TRAIN")
    sdxl_preset = job_input.get("sdxl_preset", "SDXL_FULL")

    original_path = os.path.join(tmp, "original.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    intermediate_path = os.path.join(tmp, "sd15_attacked.png")
    final_path = os.path.join(tmp, "final.png")

    subprocess.run(
        [
            "python3", f"{ML_SRC}/aspl_attack.py",
            "--original", original_path,
            "--checkpoint", SD15_CKPT,
            "--prompt", prompt,
            "--output", intermediate_path,
            "--preset", sd15_preset,
        ],
        check=True,
    )

    subprocess.run(
        [
            "python3", f"{ML_SRC}/aspl_attack_sdxl_only.py",
            "--original", intermediate_path,
            "--sdxl-checkpoint", SDXL_CKPT,
            "--prompt", prompt,
            "--output", final_path,
            "--preset", sdxl_preset,
        ],
        check=True,
    )

    with open(final_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode()

    return {"output_b64": output_b64}


def _run_score_protection(job_input: dict, tmp: str) -> dict:
    original_path = os.path.join(tmp, "original.png")
    protected_path = os.path.join(tmp, "protected.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(job_input["original_b64"]))
    with open(protected_path, "wb") as f:
        f.write(base64.b64decode(job_input["protected_b64"]))

    result = subprocess.run(
        [
            "python3", f"{ML_SRC}/protection_score.py",
            "--original", original_path,
            "--protected", protected_path,
            "--sd15-checkpoint", SD15_CKPT,
            "--sdxl-checkpoint", SDXL_CKPT,
            "--prompt", job_input.get("prompt", "artwork"),
            "--seed", str(job_input.get("seed", 1)),
            "--train-steps", str(job_input.get("train_steps", 150)),
            "--num-samples", str(job_input.get("num_samples", 2)),
        ],
        check=True, capture_output=True, text=True,
    )
    # protection_score.py's CLI writes progress to stderr and exactly one
    # JSON document to stdout (see its own module doc) -- stdout should
    # have nothing else on it, but .strip() guards against a stray
    # trailing newline either way.
    return json.loads(result.stdout.strip())


def handler(job):
    job_input = job["input"]
    action = job_input.get("action", "dual_arch_cloak")

    with tempfile.TemporaryDirectory() as tmp:
        if action == "score_protection":
            return _run_score_protection(job_input, tmp)
        return _run_dual_arch_cloak(job_input, tmp)


runpod.serverless.start({"handler": handler})
