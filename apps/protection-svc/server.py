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
from orchestrate import ML_ENGINE_DIR, PRESETS, choose_processing_size, protect  # noqa: E402
import jobs_db  # noqa: E402

app = FastAPI(title="protection-svc", version="0.1.0")

_executor = ThreadPoolExecutor(max_workers=1)  # see module docstring for why
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
        )
        jobs_db.set_completed(_jobs_conn, job_id, result)  # result already has "status": "completed"
    except Exception as exc:  # noqa: BLE001 -- report failure via job status, don't just kill the thread silently
        jobs_db.set_failed(_jobs_conn, job_id, str(exc), traceback.format_exc())


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
    jobs_db.create_job(_jobs_conn, job_id, req.model_dump())

    _executor.submit(_run_job, job_id, req)
    return {"jobId": job_id, "status": "queued"}


@app.get("/protect/{job_id}")
def get_protect_job(job_id: str):
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
