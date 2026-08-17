"""RunPod Serverless handler wrapping the jobs that reuse the same
checkpoints and attack scripts baked into lentropy/dontai-strongprotect:
latest (this image is FROM that one, see the sibling Dockerfile):

- "clean_cloak" (DEFAULT since 2026-08-13, unset action): clean_protect.py's
  four-stage native pixel-space-only chain (SD1.5 full -> SDXL full ->
  SD1.5 top-up -> SDXL top-up, no VAE-decode latent stage at all). See
  clean_protect.py's own module doc for the full story -- it replaces
  "dual_arch_cloak" because that mechanism's latent stage was confirmed
  causing real color/quality damage (an oil-painting-style distortion,
  sky reduced to magenta/green blotches) on a real user's deployed
  artwork. Same STATUS CAVEAT as dual_arch_cloak had: n=1, single-image
  validated, not this project's usual n=30-plus-replication bar -- wired
  in ahead of that (2026-08-13 decision) because shipping a visually
  honest n=1 mechanism beats leaving a known-broken one live. Optional
  job_input["epsilon"] (2026-08-15, upload-time "강도" control): see
  clean_protect.py's epsilon_override doc -- clamped there to [0.01, 0.30]
  regardless of what this job_input actually contains, so a bad/missing
  caller-side clamp can't reach the mechanism itself.
- "dual_arch_cloak" (LEGACY, explicit-only -- no longer the default):
  hybrid_protect.py's four-stage latent-then-pixel composition
  (2026-08-08). Kept only for an explicit rollback; do not point new
  callers at this action.
- "score_protection": the Test Lab's on-demand real-LoRA-training
  protection score (see protection_score.py's module doc) -- trains
  baseline-vs-protected LoRAs for a single artwork and CLIP-scores the
  result, run once per user request rather than n=30 for a statistical
  claim.
- "generate_lora": lora_generate.py, a standalone LoRA-training utility
  (see that script's own doc).

Every mechanism here runs its own stages as subprocesses (not in-process
imports), for the same reason across all of them: each attack script
loads its own full pipeline into VRAM and expects to have it torn down
cleanly on exit -- a fresh process per stage is the simplest way to
guarantee no state (or VRAM) leaks between RunPod job invocations sharing
one worker.
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
from caption_image import caption_image  # noqa: E402
from clean_protect import clean_protect  # noqa: E402
from hybrid_protect import hybrid_protect  # noqa: E402


def _run_clean_cloak(job_input: dict, tmp: str) -> dict:
    """2026-08-13 default: clean_protect.py's native pixel-space-only
    sequential chain -- see that module's own doc for why it replaces
    _run_dual_arch_cloak (hybrid_protect.py's latent stage was confirmed
    causing real color/quality damage on real user artwork)."""
    image_b64 = job_input["image_b64"]
    caller_prompt = job_input.get("prompt", "artwork")
    preset = job_input.get("clean_preset", "CLEAN_FULL")
    # Upload-time "강도" (intensity) override (2026-08-15) -- optional,
    # only present when a caller opted into a non-default epsilon (see
    # clean_protect.py's own epsilon_override doc). None here means "run
    # the preset unmodified", same as clean_protect() itself defaults to.
    epsilon_override = job_input.get("epsilon")

    original_path = os.path.join(tmp, "original.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    prompt = caption_image(original_path)
    print(f"[handler] caller-supplied prompt: {caller_prompt!r} -- attacking with content caption: {prompt!r}", flush=True)

    final_path = os.path.join(tmp, "final.png")

    clean_protect(
        original_path=original_path,
        sd15_checkpoint=SD15_CKPT,
        sdxl_checkpoint=SDXL_CKPT,
        prompt=prompt,
        output_path=final_path,
        preset_name=preset,
        work_dir=os.path.join(tmp, "_clean_stages"),
        epsilon_override=epsilon_override,
    )

    with open(final_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode()

    return {"output_b64": output_b64}


def _run_dual_arch_cloak(job_input: dict, tmp: str) -> dict:
    image_b64 = job_input["image_b64"]
    caller_prompt = job_input.get("prompt", "artwork")
    preset = job_input.get("hybrid_preset", "HYBRID_FULL")

    original_path = os.path.join(tmp, "original.png")
    with open(original_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    # Not caller_prompt (the artwork's own title, in practice) -- see
    # caption_image.py's module doc for why a title-as-prompt is
    # unreliable for training/attack purposes and protection_score.py's
    # identical fix on the measurement side. ASPL's surrogate fine-tuning
    # needs a prompt the image actually matches for its denoising loss to
    # mean anything; a real scraper's own auto-captioning tool wouldn't
    # have used the artist's title either.
    prompt = caption_image(original_path)
    print(f"[handler] caller-supplied prompt: {caller_prompt!r} -- attacking with content caption: {prompt!r}", flush=True)

    final_path = os.path.join(tmp, "final.png")

    hybrid_protect(
        original_path=original_path,
        sd15_checkpoint=SD15_CKPT,
        sdxl_checkpoint=SDXL_CKPT,
        prompt=prompt,
        output_path=final_path,
        preset_name=preset,
        work_dir=os.path.join(tmp, "_hybrid_stages"),
        # Advanced-options upload feature (2026-08-08) -- optional, only
        # present when a caller opted into a non-default epsilon (see
        # hybrid_protect()'s own doc on this override).
        latent_epsilon_override=job_input.get("latent_epsilon"),
        pixel_epsilon_override=job_input.get("pixel_epsilon"),
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


def _run_generate_lora(job_input: dict, tmp: str) -> dict:
    image_path = os.path.join(tmp, "image.png")
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(job_input["image_b64"]))
    output_path = os.path.join(tmp, "lora.safetensors")

    result = subprocess.run(
        [
            "python3", f"{ML_SRC}/lora_generate.py",
            "--image", image_path,
            "--output", output_path,
            "--sd15-checkpoint", SD15_CKPT,
            "--prompt", job_input.get("prompt", "artwork"),
            "--seed", str(job_input.get("seed", 1)),
            "--train-steps", str(job_input.get("train_steps", 150)),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"lora_generate.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    with open(output_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode()
    meta = json.loads(result.stdout.strip())
    return {"output_b64": output_b64, "contentPrompt": meta.get("contentPrompt")}


def handler(job):
    job_input = job["input"]
    # Default changed 2026-08-13: "clean_cloak" (clean_protect.py, no
    # latent-space stage) replaces "dual_arch_cloak" (hybrid_protect.py) as
    # the default -- hybrid_protect.py's latent stage was confirmed
    # producing real color/quality damage on real user artwork in
    # production. "dual_arch_cloak" stays available, unadvertised, only
    # for an explicit rollback -- not for new callers.
    action = job_input.get("action", "clean_cloak")

    with tempfile.TemporaryDirectory() as tmp:
        if action == "score_protection":
            return _run_score_protection(job_input, tmp)
        if action == "generate_lora":
            return _run_generate_lora(job_input, tmp)
        if action == "dual_arch_cloak":
            return _run_dual_arch_cloak(job_input, tmp)
        return _run_clean_cloak(job_input, tmp)


runpod.serverless.start({"handler": handler})
