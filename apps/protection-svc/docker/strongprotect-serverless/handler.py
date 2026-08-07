"""RunPod Serverless handler wrapping the same dual-arch attack
remote_dual_arch_cloak() dispatches over SSH -- built to empirically compare
RunPod Serverless cost/reliability against the Pods-based approach
(PHASE4_SCOPING.md SS6's 2026-08-07 update). Reuses the checkpoints and
attack scripts already baked into lentropy/dontai-strongprotect:latest
(this image is FROM that one, see the sibling Dockerfile) -- no new
dependency on the attack logic itself, just a different invocation surface
(RunPod's job protocol instead of SSH+subprocess).
"""

import base64
import os
import subprocess
import tempfile

import runpod

SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
ML_SRC = "/workspace/dontai-protection-svc/ml-engine/src"


def handler(job):
    job_input = job["input"]
    image_b64 = job_input["image_b64"]
    prompt = job_input.get("prompt", "artwork")
    sd15_preset = job_input.get("sd15_preset", "L3_ANTI_TRAIN")
    sdxl_preset = job_input.get("sdxl_preset", "SDXL_FULL")

    with tempfile.TemporaryDirectory() as tmp:
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


runpod.serverless.start({"handler": handler})
