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
- Job state is an in-memory dict -- lost on restart. A real deployment needs
  persistent job state, not just this (the Postgres/Redis stack
  PROJECT_DESIGN.md already calls for elsewhere would be the natural home).
- No auth, no per-tenant isolation -- this is a dev-loopback service.
"""

import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from orchestrate import ML_ENGINE_DIR, PRESETS, protect  # noqa: E402

app = FastAPI(title="protection-svc", version="0.1.0")

_executor = ThreadPoolExecutor(max_workers=1)  # see module docstring for why
_jobs: dict[str, dict] = {}
_jobs_lock = Lock()


class ProtectRequest(BaseModel):
    imageUri: str
    protectionProfile: str = "L3_ANTI_TRAIN"
    eot: Optional[bool] = None  # None = apply INTEGRATION.md's per-preset default
    styleTargetUri: Optional[str] = None
    title: str = "Untitled artwork"
    creatorId: str = "creator_unknown"
    allowAiTraining: bool = False
    watermarkPayloadHex: str = "deadbeefcafef00d"
    size: int = 256


def _run_job(job_id: str, req: ProtectRequest) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"

    try:
        out_dir = str(Path(ML_ENGINE_DIR).parent / "out" / job_id)
        style_target = req.styleTargetUri or str(ML_ENGINE_DIR / "out" / "style_target.png")

        result = protect(
            input_path=req.imageUri,
            out_dir=out_dir,
            preset_name=req.protectionProfile,
            style_target_path=style_target,
            title=req.title,
            creator_id=req.creatorId,
            allow_ai_training=req.allowAiTraining,
            watermark_payload_hex=req.watermarkPayloadHex,
            size=req.size,
            eot=req.eot,
        )
        with _jobs_lock:
            _jobs[job_id].update(result)  # result already has "status": "completed"
    except Exception as exc:  # noqa: BLE001 -- report failure via job status, don't just kill the thread silently
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["traceback"] = traceback.format_exc()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/protect", status_code=202)
def create_protect_job(req: ProtectRequest):
    if req.protectionProfile not in PRESETS:
        raise HTTPException(
            400, f"unknown protectionProfile {req.protectionProfile!r}, expected one of {list(PRESETS)}"
        )
    if not Path(req.imageUri).exists():
        raise HTTPException(400, f"imageUri {req.imageUri!r} not found (local file path in this PoC, see module docstring)")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    with _jobs_lock:
        _jobs[job_id] = {"jobId": job_id, "status": "queued", "queuedAt": time.time()}

    _executor.submit(_run_job, job_id, req)
    return {"jobId": job_id, "status": "queued"}


@app.get("/protect/{job_id}")
def get_protect_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
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
