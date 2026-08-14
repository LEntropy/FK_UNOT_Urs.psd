"""HTTP wrapper around orchestrate.py's protect() pipeline, implementing the
job-based API contract from apps/protection-svc/INTEGRATION.md:

    POST /protect            -> 202 { jobId, status: "queued" }
    GET  /protect/{jobId}     -> current job status/result

Why job-based, not synchronous: protect() takes anywhere from about a minute
(L1_PREVIEW at 256px) to potentially hours (L3_ANTI_TRAIN + EOT at 1024px+ --
see ml-engine/README.md's 1024px re-validation notes on time/VRAM cost). A
synchronous HTTP request can't sit open for that.

Concurrency: max_workers=1 in the executor below is deliberate, not an
oversight. ml-engine/README.md's 1024px re-validation found that GPU VRAM
usage from eot_samples x size can already approach an 8GB card's limit for
a SINGLE job. Running two protect() jobs at once on the same GPU risks an
out-of-memory crash, not just slowness. A real deployment needs either a
GPU worker pool (one job per GPU) or a queue that serializes jobs per GPU --
this is neither, just a single in-process worker sized to what's actually
safe on the one GPU this project has tested against.

Known gaps, not hidden:
- `imageUri` is a local file path in this PoC, not a real object-storage
  URI -- there's no S3-or-equivalent integration here.
- Job state now persists to SQLite (src/jobs_db.py) instead of an
  in-memory dict -- a *finished* job's status/result survives a restart.
  This does NOT make an interrupted GPU job resumable (no checkpoint
  mechanism exists in ml-engine/rust-core to resume a partial
  optimization from) -- any job genuinely mid-flight when the process
  dies gets marked failed with an honest "interrupted by restart" message
  on the next startup, rather than sitting in queued/processing forever
  with no way to tell "still running" from "died silently". See
  jobs_db.py's own module doc for the full reasoning.
- No auth, no per-tenant isolation -- this is a dev-loopback service.
"""

import os
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from orchestrate import ML_ENGINE_DIR, PRESETS, USE_REMOTE_GPU, choose_processing_size, protect, run_rust_core  # noqa: E402
import jobs_db  # noqa: E402

if USE_REMOTE_GPU:
    from remote_gpu import (
        remote_detect_model_leak,
        remote_measure_existing_images,
        serverless_score_protection,
        serverless_generate_lora,
    )

sys.path.insert(0, str(ML_ENGINE_DIR / "src"))
from tag_suggest import suggest_tags  # noqa: E402
from original_preview import make_original_preview  # noqa: E402

if not USE_REMOTE_GPU:
    from evaluate import compute_protection_metrics
    # model_leak_detect is NOT imported here at module scope, unlike
    # evaluate.py above -- it needs diffusers/transformers (a full
    # StableDiffusionPipeline + CLIP), which this project's local (non-
    # remote-GPU) dev venv has never carried (see generate_and_score.py's
    # own doc: "ml-engine's own .venv, which only has plain torch/Pillow").
    # Importing it eagerly here would crash server.py's startup entirely
    # for that dev mode even though most requests never touch this
    # endpoint. Imported lazily inside _run_model_leak_job instead.

app = FastAPI(title="protection-svc", version="0.1.0")

_executor = ThreadPoolExecutor(max_workers=1)  # see module docstring for why

# 2026-08-13: separate, higher-concurrency pool for job types that NEVER
# touch this process's own (possibly-absent) local GPU -- score-protection
# and lora-generation both raise immediately if USE_REMOTE_GPU isn't set
# (see their handlers below), meaning every real execution of either is
# 100% RunPod Serverless: this process just base64-encodes the input,
# POSTs a job, and polls GET /status in a loop. None of that contends for
# the single local GPU max_workers=1 exists to protect, so serializing
# these behind /protect and /detect-model-leak jobs (which DO still touch
# local GPU on their non-remote code paths) was pure unnecessary queueing
# depth, not actual safety -- a Test Lab user's score request could sit
# behind an unrelated slow local-GPU protect() job for no resource reason.
# 4 is a deliberately modest concurrency bump, not "unbounded": still a
# real cap (this is the queue the user asked for, sized to what RunPod's
# own worker pool can actually take on -- see the dontai-strongprotect
# endpoint's own workers.max=2 in its RunPod config; 4 gives headroom for
# a second endpoint or brief bursts without flooding either one, since
# RunPod's own QUEUE_DELAY autoscaling absorbs anything beyond what's
# currently running rather than rejecting it).
_cloud_executor = ThreadPoolExecutor(max_workers=4)

_JOBS_DB_PATH = os.environ.get("JOBS_DB_PATH", str(Path(__file__).parent / "data" / "jobs.db"))
_jobs_conn = jobs_db.connect(_JOBS_DB_PATH)

_interrupted_count = jobs_db.mark_interrupted_jobs_failed(_jobs_conn)
if _interrupted_count:
    print(f"[startup] marked {_interrupted_count} job(s) failed -- interrupted by a previous process's restart")


class ProtectRequest(BaseModel):
    imageUri: str
    protectionProfile: str = "L3_ANTI_TRAIN"
    eot: Optional[bool] = None  # None = apply INTEGRATION.md's per-preset default
    styleTargetUri: Optional[str] = None
    title: str = "Untitled artwork"
    creatorId: str = "creator_unknown"
    allowAiTraining: bool = False
    watermarkPayloadHex: str = "deadbeefcafef00d"
    # None (the only caller today, asset-service, never sets this) means
    # "pick from the real upload's own resolution" -- see orchestrate.py's
    # choose_processing_size for why that beat a fixed 256 on both quality
    # and protection strength, measured for real. An explicit value here
    # still overrides it, for callers (tests, the CLI, future experiments)
    # that want a specific processing size on purpose.
    size: Optional[int] = None
    # Opt-in only, off by default -- PHASE4_SCOPING.md §6: replaces
    # style-cloak with dual-arch ASPL attacks -- SD1.5 and SDXL, each
    # attacked independently and sequentially, both validated protection
    # effects (see PHASE4_SCOPING.md §6). Several minutes per job on a
    # dedicated A40-class pod (MULTIARCH_GPU_* env vars, see remote_gpu.py's
    # remote_dual_arch_cloak) instead of the GPU PC -- falls back to normal
    # style-cloak if that pod is unreachable or unconfigured, so this never
    # turns "protected" into "unprotected".
    strongProtection: bool = False
    # Advanced-options upload feature (2026-08-08) -- both None (the
    # default) means run at HYBRID_FULL's own preset values, not "no
    # protection." Ignored unless strongProtection is also true. See
    # hybrid_protect.py's own override doc for why this exists and what
    # it does and doesn't change.
    strongProtectionLatentEpsilon: Optional[float] = None
    strongProtectionPixelEpsilon: Optional[float] = None


def _run_job(job_id: str, req: ProtectRequest) -> None:
    jobs_db.set_processing(_jobs_conn, job_id)

    try:
        out_dir = str(Path(ML_ENGINE_DIR).parent / "out" / job_id)
        style_target = req.styleTargetUri or str(ML_ENGINE_DIR / "out" / "style_target.png")
        size = req.size if req.size is not None else choose_processing_size(req.imageUri)

        result = protect(
            input_path=req.imageUri,
            out_dir=out_dir,
            preset_name=req.protectionProfile,
            style_target_path=style_target,
            title=req.title,
            creator_id=req.creatorId,
            allow_ai_training=req.allowAiTraining,
            watermark_payload_hex=req.watermarkPayloadHex,
            size=size,
            eot=req.eot,
            strong_protection=req.strongProtection,
            strong_protection_latent_epsilon=req.strongProtectionLatentEpsilon,
            strong_protection_pixel_epsilon=req.strongProtectionPixelEpsilon,
            # Cancel-upload feature (2026-08-14): as soon as the RunPod job
            # actually exists, remember its id so POST /protect/{job_id}/
            # cancel below has something real to cancel, not just this
            # row's own status.
            on_runpod_job_submitted=lambda runpod_job_id, endpoint_id: jobs_db.set_runpod_job_id(
                _jobs_conn, job_id, runpod_job_id, endpoint_id
            ),
        )
        jobs_db.set_completed(_jobs_conn, job_id, result)  # result already has "status": "completed"
    except Exception as exc:  # noqa: BLE001 -- report failure via job status, don't just kill the thread silently
        jobs_db.set_failed(_jobs_conn, job_id, str(exc), traceback.format_exc())


class DetectModelLeakRequest(BaseModel):
    originalImageUri: str
    suspectLoraUri: str
    prompts: list[str]
    numSamples: int = 4
    resolution: int = 512
    genSeed: int = 42


def _local_detect_model_leak(**kwargs) -> dict:
    """Thin indirection point so tests can monkeypatch the local-GPU code
    path without importing model_leak_detect itself -- that module's
    top-level `from transformers import ...` isn't satisfiable in this
    project's local test venv (see the module-scope note above), same
    reasoning as evaluate.py/compute_protection_metrics being importable
    at module scope while this deliberately isn't."""
    from model_leak_detect import detect_model_leak

    return detect_model_leak(**kwargs)


def _run_model_leak_job(job_id: str, req: DetectModelLeakRequest) -> None:
    jobs_db.set_processing(_jobs_conn, job_id)
    try:
        if USE_REMOTE_GPU:
            result = remote_detect_model_leak(
                original_path=req.originalImageUri,
                suspect_lora_path=req.suspectLoraUri,
                prompts=req.prompts,
                num_samples=req.numSamples,
                resolution=req.resolution,
                gen_seed=req.genSeed,
            )
        else:
            result = _local_detect_model_leak(
                checkpoint_path=os.environ.get(
                    "SD15_CHECKPOINT", str(ML_ENGINE_DIR / "checkpoints" / "v1-5-pruned-emaonly-fp16.safetensors")
                ),
                suspect_lora_path=req.suspectLoraUri,
                original_image_path=req.originalImageUri,
                prompts=req.prompts,
                num_samples=req.numSamples,
                resolution=req.resolution,
                gen_seed=req.genSeed,
            )
        result["status"] = "completed"
        jobs_db.set_completed(_jobs_conn, job_id, result)
    except Exception as exc:  # noqa: BLE001 -- report failure via job status, mirrors _run_job
        jobs_db.set_failed(_jobs_conn, job_id, str(exc), traceback.format_exc())


class ScoreProtectionRequest(BaseModel):
    originalImageUri: str
    protectedImageUri: str
    prompt: str = "artwork"
    seed: int = 1
    trainSteps: int = 150
    numSamples: int = 2


def _run_score_protection_job(job_id: str, req: ScoreProtectionRequest) -> None:
    jobs_db.set_processing(_jobs_conn, job_id)
    try:
        if not USE_REMOTE_GPU:
            # This feature only ever runs against strong_protection artworks
            # (asset-service gates it to usedStrongProtection == true), which
            # itself only ever completed via the RunPod Serverless path --
            # there's no local-GPU fallback to reach for here, unlike
            # /detect-model-leak's dual local/remote code paths.
            raise RuntimeError("score-protection requires USE_REMOTE_GPU (RunPod Serverless) to be configured")

        result = serverless_score_protection(
            original_path=req.originalImageUri,
            protected_path=req.protectedImageUri,
            prompt=req.prompt,
            seed=req.seed,
            train_steps=req.trainSteps,
            num_samples=req.numSamples,
        )
        result["status"] = "completed"
        jobs_db.set_completed(_jobs_conn, job_id, result)
    except Exception as exc:  # noqa: BLE001 -- report failure via job status, mirrors _run_model_leak_job
        jobs_db.set_failed(_jobs_conn, job_id, str(exc), traceback.format_exc())


class LoraGenerationRequest(BaseModel):
    imageUri: str
    prompt: str = "artwork"
    seed: int = 1
    trainSteps: int = 150


def _run_lora_generation_job(job_id: str, req: LoraGenerationRequest) -> None:
    jobs_db.set_processing(_jobs_conn, job_id)
    try:
        if not USE_REMOTE_GPU:
            # Same reasoning as score-protection's own check just above --
            # lora_generate.py needs the same heavy diffusers/peft/SD1.5
            # checkpoint stack that only exists in the RunPod Serverless
            # strong_protection worker, no local fallback.
            raise RuntimeError("generate-lora requires USE_REMOTE_GPU (RunPod Serverless) to be configured")

        out_dir = Path(ML_ENGINE_DIR).parent / "out" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "lora.safetensors")

        result = serverless_generate_lora(
            image_path=req.imageUri,
            output_path=output_path,
            prompt=req.prompt,
            seed=req.seed,
            train_steps=req.trainSteps,
        )
        result["status"] = "completed"
        jobs_db.set_completed(_jobs_conn, job_id, result)
    except Exception as exc:  # noqa: BLE001 -- report failure via job status, mirrors _run_score_protection_job
        jobs_db.set_failed(_jobs_conn, job_id, str(exc), traceback.format_exc())


class SuggestTagsRequest(BaseModel):
    imageUri: str
    topK: int = 10


class RemeasureRequest(BaseModel):
    originalImageUri: str
    cloakedImageUri: str
    # Same default as ProtectRequest's own styleTargetUri handling
    # (_run_job above) -- real uploads never pass a per-artwork style
    # target (see orchestrate.py's own note that auto-selection is a
    # no-op under USE_REMOTE_GPU), so the fixed default asset is what
    # every real protect() job actually compared against, and reusing it
    # here keeps a re-test comparable to the original measurement.
    styleTargetUri: Optional[str] = None
    size: int = 256


class OriginalPreviewRequest(BaseModel):
    imageUri: str
    watermarkPayloadHex: str = "deadbeefcafef00d"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/suggest-tags")
def suggest_tags_endpoint(req: SuggestTagsRequest):
    """Fast, synchronous (unlike /protect) -- a single CLIP forward pass
    against a precomputed tag-embedding matrix, meant to run in the
    upload-preview UI before the user commits to the slow /protect job.
    See ml-engine/src/tag_suggest.py's module doc for the mechanism."""
    if not Path(req.imageUri).exists():
        raise HTTPException(400, f"imageUri {req.imageUri!r} not found (local file path in this PoC, see module docstring)")

    tags = suggest_tags(req.imageUri, top_k=req.topK)
    return {"tags": tags}


@app.post("/original-preview")
def original_preview_endpoint(req: OriginalPreviewRequest):
    """Creator-opt-in original-preview derivative (coin-system feature,
    2026-08-10) -- synchronous like /suggest-tags above, not job-based:
    original_preview.py is pure PIL (resize + watermark overlay), no
    GPU/model involved, so this finishes in well under a second even at
    the 2048px cap. See ml-engine/src/original_preview.py's module doc for
    what this derivative actually is (never the real original bytes).

    Embeds the same invisible watermarkPayloadHex as the real protected
    image (rust-core's `embed`, same as orchestrate.py's own step 2/4) on
    top of the visible tiled watermark original_preview.py already drew --
    if this derivative leaks, detection-svc's existing matching logic can
    still trace it back, no new payload scheme needed.
    """
    if not Path(req.imageUri).exists():
        raise HTTPException(400, f"imageUri {req.imageUri!r} not found (local file path in this PoC, see module docstring)")

    out_dir = Path(ML_ENGINE_DIR).parent / "out" / f"original_preview_{uuid.uuid4().hex[:16]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / "preview.png"
    watermarked_path = out_dir / "preview_watermarked.png"

    make_original_preview(req.imageUri, str(preview_path))
    run_rust_core(
        "embed",
        "--input", str(preview_path),
        "--output", str(watermarked_path),
        "--payload-hex", req.watermarkPayloadHex,
        "--strength", "24.0",
    )

    from PIL import Image as _Image

    with _Image.open(watermarked_path) as im:
        width, height = im.size

    return {"previewUri": str(watermarked_path), "width": width, "height": height}


@app.post("/remeasure")
def remeasure(req: RemeasureRequest):
    """Live, synchronous re-run of the "테스트 랩" 보호 강도 테스트 tab's
    protection-strength measurement, on demand -- distinct from the
    styleDriftScore/perceptualPsnrDb/styleSimilarityToOriginal stored on
    the artwork at upload time (protect()'s own one-shot measurement).
    Fast (a few VGG19 forward passes plus one SSH round trip under
    USE_REMOTE_GPU, no cloak/training step), so this runs synchronously
    rather than through the job-based /protect flow.
    """
    if not Path(req.originalImageUri).exists():
        raise HTTPException(400, f"originalImageUri {req.originalImageUri!r} not found")
    if not Path(req.cloakedImageUri).exists():
        raise HTTPException(400, f"cloakedImageUri {req.cloakedImageUri!r} not found")

    style_target = req.styleTargetUri or str(ML_ENGINE_DIR / "out" / "style_target.png")
    if not Path(style_target).exists():
        raise HTTPException(400, f"styleTargetUri {style_target!r} not found")

    try:
        if USE_REMOTE_GPU:
            metrics = remote_measure_existing_images(
                original_path=req.originalImageUri,
                cloaked_path=req.cloakedImageUri,
                style_target_path=style_target,
                size=req.size,
            )
        else:
            metrics = compute_protection_metrics(
                original_path=req.originalImageUri,
                cloaked_path=req.cloakedImageUri,
                style_target_path=style_target,
                size=req.size,
            )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as a real error, unlike protect()'s own best-effort metrics step (there, a missing measurement shouldn't fail an upload; here, the measurement IS the whole request)
        raise HTTPException(502, f"re-measurement failed: {exc}") from None

    return metrics


@app.post("/protect", status_code=202)
def create_protect_job(req: ProtectRequest):
    if req.protectionProfile not in PRESETS:
        raise HTTPException(
            400, f"unknown protectionProfile {req.protectionProfile!r}, expected one of {list(PRESETS)}"
        )
    if not Path(req.imageUri).exists():
        raise HTTPException(400, f"imageUri {req.imageUri!r} not found (local file path in this PoC, see module docstring)")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    jobs_db.create_job(_jobs_conn, job_id, req.model_dump())

    _executor.submit(_run_job, job_id, req)
    return {"jobId": job_id, "status": "queued"}


@app.get("/protect/{job_id}")
def get_protect_job(job_id: str):
    job = jobs_db.get_job(_jobs_conn, job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return job


@app.post("/protect/{job_id}/cancel")
def cancel_protect_job(job_id: str):
    """Cancel-upload feature (2026-08-14, asset-service's own POST
    /artworks/:id/cancel is the real entry point a creator hits -- this is
    what it calls in turn). Marks the job 'cancelled' in jobs_db
    immediately (jobs_db.request_cancel's own doc explains the race guard
    that keeps a late-finishing background thread from resurrecting it),
    then best-effort asks RunPod to actually stop the GPU job if one was
    ever submitted for it (strong_protection jobs only -- style_cloak's
    local/SSH path has no equivalent mid-flight stop today, see this
    route's own 200 response either way: from the caller's perspective the
    job IS cancelled now regardless of whether RunPod's side finishes a
    beat later).
    """
    row = jobs_db.request_cancel(_jobs_conn, job_id)
    if row is None:
        existing = jobs_db.get_job(_jobs_conn, job_id)
        if existing is None:
            raise HTTPException(404, f"no job {job_id!r}")
        return {"jobId": job_id, "status": existing["status"], "alreadyTerminal": True}

    runpod_job_id = row["runpod_job_id"]
    runpod_endpoint_id = row["runpod_endpoint_id"]
    if runpod_job_id and runpod_endpoint_id and USE_REMOTE_GPU:
        from remote_gpu import cancel_runpod_job

        cancel_runpod_job(runpod_endpoint_id, runpod_job_id)

    return {"jobId": job_id, "status": "cancelled"}


@app.post("/detect-model-leak", status_code=202)
def create_detect_model_leak_job(req: DetectModelLeakRequest):
    """detection-svc's model-leak report pathway (PHASE4_SCOPING.md
    detection-tracking follow-up): given a suspect LoRA file and the
    registered artwork's own (published, protected) image, does
    generating from the suspect LoRA come out anomalously close to it?
    See ml-engine/src/model_leak_detect.py's module doc for the mechanism.

    Job-based like /protect, not synchronous like /remeasure -- unlike
    /remeasure's few VGG19 forward passes, this actually generates images
    (num_samples x len(prompts) x 2 conditions), the same order of cost
    class as a real /protect job, submitted to the same single-worker
    executor (so it never contends with an in-flight /protect job for the
    same GPU).
    """
    if not Path(req.originalImageUri).exists():
        raise HTTPException(400, f"originalImageUri {req.originalImageUri!r} not found")
    if not Path(req.suspectLoraUri).exists():
        raise HTTPException(400, f"suspectLoraUri {req.suspectLoraUri!r} not found")
    if not req.prompts:
        raise HTTPException(400, "prompts must be non-empty")

    job_id = f"leakjob_{uuid.uuid4().hex[:12]}"
    jobs_db.create_job(_jobs_conn, job_id, req.model_dump())

    _executor.submit(_run_model_leak_job, job_id, req)
    return {"jobId": job_id, "status": "queued"}


@app.get("/detect-model-leak/{job_id}")
def get_detect_model_leak_job(job_id: str):
    job = jobs_db.get_job(_jobs_conn, job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return job


@app.post("/score-protection", status_code=202)
def create_score_protection_job(req: ScoreProtectionRequest):
    """Test Lab's on-demand real-LoRA-training protection score (see
    protection_score.py's module doc for the mechanism). Job-based like
    /protect and /detect-model-leak, not synchronous like /remeasure --
    this trains four LoRAs (SD1.5/SDXL x baseline/protected), the
    slowest job class this service exposes, submitted to the same
    single-worker executor so it never contends with an in-flight
    /protect or /detect-model-leak job for the same GPU.
    """
    if not Path(req.originalImageUri).exists():
        raise HTTPException(400, f"originalImageUri {req.originalImageUri!r} not found")
    if not Path(req.protectedImageUri).exists():
        raise HTTPException(400, f"protectedImageUri {req.protectedImageUri!r} not found")

    job_id = f"scorejob_{uuid.uuid4().hex[:12]}"
    jobs_db.create_job(_jobs_conn, job_id, req.model_dump())

    _cloud_executor.submit(_run_score_protection_job, job_id, req)  # Serverless-only, see _cloud_executor's own comment
    return {"jobId": job_id, "status": "queued"}


@app.get("/score-protection/{job_id}")
def get_score_protection_job(job_id: str):
    job = jobs_db.get_job(_jobs_conn, job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return job


@app.post("/lora-jobs", status_code=202)
def create_lora_generation_job(req: LoraGenerationRequest):
    """Coin-system feature (2026-08-10): trains a real, downloadable SD1.5
    LoRA on a single artwork image (see ml-engine/src/lora_generate.py's
    module doc). Job-based like /score-protection, not synchronous --
    one LoRA training run, same order of cost as a single arm of
    /score-protection's four, submitted to the same Serverless-only
    _cloud_executor pool (see its own comment for why this and
    /score-protection are split out from the local-GPU-risk executor).
    """
    if not Path(req.imageUri).exists():
        raise HTTPException(400, f"imageUri {req.imageUri!r} not found (local file path in this PoC, see module docstring)")

    job_id = f"lorajob_{uuid.uuid4().hex[:12]}"
    jobs_db.create_job(_jobs_conn, job_id, req.model_dump())

    _cloud_executor.submit(_run_lora_generation_job, job_id, req)
    return {"jobId": job_id, "status": "queued"}


@app.get("/lora-jobs/{job_id}")
def get_lora_generation_job(job_id: str):
    job = jobs_db.get_job(_jobs_conn, job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return job


if __name__ == "__main__":
    import os

    import uvicorn

    # Default 8000 for local dev; override via env when 8000 is already
    # taken by something else on the host (e.g. a pre-existing unrelated
    # service on a shared deployment box).
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
