"""RunPod Serverless handler wrapping two jobs that reuse the same
checkpoints and attack scripts baked into lentropy/dontai-strongprotect:
latest (this image is FROM that one, see the sibling Dockerfile):

- "dual_arch_cloak" (default, unset action): the four-stage hybrid attack
  (hybrid_protect.py, 2026-08-08) -- latent-space SD1.5+SDXL passes
  (large budget, visually plausible via VAE decode) followed by small-
  epsilon pixel-space SD1.5+SDXL top-up passes. Replaces the earlier
  two-stage pixel-only chain (aspl_attack.py -> aspl_attack_sdxl_only.py)
  that this job originally ran. STATUS CAVEAT (see hybrid_protect.py's
  own module doc for the full reasoning): this composition is validated
  at n=1, SD1.5-only -- not the n=30-plus-independent-replication bar
  every other mechanism here cleared before shipping. Wired in anyway at
  the user's explicit, informed decision (2026-08-08) to ship ahead of
  full validation rather than wait; SDXL has not been re-checked at the
  epsilon values actually in production use.
- "score_protection": the Test Lab's on-demand real-LoRA-training
  protection score (see protection_score.py's module doc) -- trains
  baseline-vs-protected LoRAs for a single artwork and CLIP-scores the
  result, run once per user request rather than n=30 for a statistical
  claim.

hybrid_protect() and protection_score.py each run their own stages as
subprocesses (not in-process imports) for the same reason the original
dual_arch_cloak job did: each attack script loads its own full pipeline
into VRAM and expects to have it torn down cleanly on exit -- a fresh
process per stage is the simplest way to guarantee no state (or VRAM)
leaks between RunPod job invocations sharing one worker.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile

import runpod

SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
ML_SRC = "/workspace/dontai-protection-svc/ml-engine/src"

sys.path.insert(0, ML_SRC)
from hybrid_protect import hybrid_protect  # noqa: E402


def _run_dual_arch_cloak(job_input: dict, tmp: str) -> dict:
    image_b64 = job_input["image_b64"]
    prompt = job_input.get("prompt", "artwork")
    preset = job_input.get("hybrid_preset", "HYBRID_FULL")

    original_path = os.path.join(tmp, "original.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    final_path = os.path.join(tmp, "final.png")

    hybrid_protect(
        original_path=original_path,
        sd15_checkpoint=SD15_CKPT,
        sdxl_checkpoint=SDXL_CKPT,
        prompt=prompt,
        output_path=final_path,
        preset_name=preset,
        work_dir=os.path.join(tmp, "_hybrid_stages"),
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
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # subprocess.CalledProcessError's own str() doesn't include
        # stdout/stderr -- surfacing them explicitly here is the only way
        # to see protection_score.py's real traceback in RunPod's job
        # error output (found live: the first real smoke test only showed
        # "exit status 2" with no further detail without this).
        raise RuntimeError(
            f"protection_score.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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
