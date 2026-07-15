"""detection-svc: Phase 3 infringement detection & evidence packaging
(PROJECT_DESIGN.md §3-7, §7). Same job-based HTTP shape as protection-svc's
server.py (POST -> 202 {caseId, status} / GET .../{caseId} to poll), but
cases persist in SQLite (src/db.py) rather than an in-memory dict --
evidence must survive a restart.

Scope: this implements runbook steps 1-3 of §7 (탐지/신고 접수 -> 자동 증거
수집 -> 증거 패키지 생성). Steps 4-6 (권리자 알림, 테이크다운, 케이스 추적)
are product/human workflow, not automated here.
"""

import os
import sys
import traceback
import uuid
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))

from asset_client import ArtworkNotFoundError, get_artwork  # noqa: E402
from db import add_evidence, connect, create_case, get_case, set_case_status  # noqa: E402
from evidence_bundle import build_bundle, write_json, write_pdf_best_effort  # noqa: E402
from evidence_capture import capture  # noqa: E402
from phash_match import is_likely_match  # noqa: E402
from rust_watermark import detect_watermark  # noqa: E402
from vision import vision_configured, web_detect_matching_urls  # noqa: E402

ASSET_SERVICE_URL = os.environ.get("ASSET_SERVICE_URL", "http://localhost:3002")
PHASH_MATCH_THRESHOLD = int(os.environ.get("PHASH_MATCH_THRESHOLD", "20"))
DEFAULT_WATERMARK_HEX = os.environ.get("DEFAULT_WATERMARK_HEX", "deadbeefcafef00d")
OUT_DIR = Path(os.environ.get("DETECTION_OUT_DIR", str(Path(__file__).parent / "out")))
DB_PATH = os.environ.get("DETECTION_DB_PATH", str(Path(__file__).parent / "data" / "detection.db"))

app = FastAPI(title="detection-svc", version="0.1.0")
_db = connect(DB_PATH)


class ScanResponse(BaseModel):
    caseId: str
    status: str


class ReportRequest(BaseModel):
    artworkId: str
    suspectUrl: str


# Runbook steps 4-6 (PROJECT_DESIGN.md §7: 권리자 알림, 대응 옵션 안내, 케이스
# 상태 추적) are human/product workflow, not automated here -- see
# RUNBOOK.md for the actual checklist a person follows. This endpoint is
# the minimum this service can offer to support that: a place to record
# which manual step a case has reached, so "케이스 상태 추적" means
# something more than a paper trail kept outside the system entirely.
MANUAL_STATUSES = {"NOTIFIED", "RESOLVED", "ESCALATED"}


class UpdateCaseRequest(BaseModel):
    status: str
    note: str | None = None


def _run_case_for_urls(case_id: str, artwork: dict, candidate_urls: list[str]) -> None:
    try:
        registered_hash = artwork.get("perceptualHash")
        # asset-service generates and stores this per-artwork now (was
        # previously dropped entirely, forcing every case to check against
        # one project-wide constant regardless of which artwork the
        # candidate URL was actually a copy of). Still fall back for
        # artworks created before that fix / rows with it unset.
        watermark_hex = artwork.get("watermarkPayloadHex") or DEFAULT_WATERMARK_HEX
        any_evidence = False

        for url in candidate_urls:
            case_out_dir = OUT_DIR / case_id / _safe_slug(url)
            import asyncio

            captured = asyncio.run(capture(url, case_out_dir))

            phash_distance = None
            is_match = False
            if captured.image_path and registered_hash:
                is_match, phash_distance = is_likely_match(registered_hash, captured.image_path, PHASH_MATCH_THRESHOLD)

            watermark_result = None
            if captured.image_path:
                try:
                    wm = detect_watermark(captured.image_path, watermark_hex)
                    watermark_result = {
                        "recoveredHex": wm.recovered_hex,
                        "avgConfidence": wm.avg_confidence,
                        "minConfidence": wm.min_confidence,
                        "bitErrorRate": wm.bit_error_rate,
                        "isMatch": wm.is_match,
                    }
                    if wm.is_match:
                        is_match = True
                except (FileNotFoundError, RuntimeError):
                    watermark_result = None

            if not is_match:
                continue

            any_evidence = True
            bundle = build_bundle(
                artwork=artwork,
                source_url=url,
                detected_at=captured.captured_at,
                phash_distance=phash_distance,
                watermark_result=watermark_result,
                headers=captured.headers,
                screenshot_path=captured.screenshot_path,
            )
            write_json(bundle, case_out_dir / "bundle.json")
            write_pdf_best_effort(bundle, case_out_dir / "bundle.pdf")

            confidence = 1.0 - (phash_distance / 256.0) if phash_distance is not None else None
            evidence_type = "watermark_match" if watermark_result and watermark_result["isMatch"] else "phash_match"
            add_evidence(_db, case_id, evidence_type, url, confidence, str(case_out_dir))

        set_case_status(_db, case_id, "EVIDENCE_READY" if any_evidence else "NO_MATCH_FOUND")
    except Exception as exc:  # noqa: BLE001 -- report failure via case status, mirrors protection-svc/server.py
        set_case_status(_db, case_id, "FAILED", f"{exc}\n{traceback.format_exc()}")


def _safe_slug(url: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in url)[:80] or "url"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan/{artwork_id}", status_code=202, response_model=ScanResponse)
async def scan_artwork(artwork_id: str):
    try:
        artwork = await get_artwork(ASSET_SERVICE_URL, artwork_id)
    except ArtworkNotFoundError:
        raise HTTPException(404, f"no artwork {artwork_id!r} on asset-service")

    protected_uri = artwork.get("protectedImageUri")
    if not protected_uri or not Path(protected_uri).exists():
        raise HTTPException(400, f"artwork {artwork_id!r} has no reachable protectedImageUri to scan from")

    case_id = f"case_{uuid.uuid4().hex[:12]}"
    create_case(_db, case_id, artwork_id, "scan")

    candidate_urls = web_detect_matching_urls(protected_uri) if vision_configured() else []
    Thread(target=_run_case_for_urls, args=(case_id, artwork, candidate_urls), daemon=True).start()

    return {"caseId": case_id, "status": "queued"}


@app.post("/reports", status_code=202, response_model=ScanResponse)
async def submit_report(req: ReportRequest):
    try:
        artwork = await get_artwork(ASSET_SERVICE_URL, req.artworkId)
    except ArtworkNotFoundError:
        raise HTTPException(404, f"no artwork {req.artworkId!r} on asset-service")

    case_id = f"case_{uuid.uuid4().hex[:12]}"
    create_case(_db, case_id, req.artworkId, "report")

    Thread(target=_run_case_for_urls, args=(case_id, artwork, [req.suspectUrl]), daemon=True).start()

    return {"caseId": case_id, "status": "queued"}


@app.get("/cases/{case_id}")
def get_case_status(case_id: str):
    case = get_case(_db, case_id)
    if case is None:
        raise HTTPException(404, f"no case {case_id!r}")
    return case


@app.patch("/cases/{case_id}")
def update_case_status(case_id: str, req: UpdateCaseRequest):
    """Records a manual runbook step (§7 steps 4-6 -- RUNBOOK.md has the
    actual checklist a person follows before calling this). Only lets a
    case move forward from EVIDENCE_READY through NOTIFIED/RESOLVED/
    ESCALATED -- not a general-purpose status override, and not reachable
    from OPEN/NO_MATCH_FOUND/FAILED, which are automated-only states.
    """
    if req.status not in MANUAL_STATUSES:
        raise HTTPException(400, f"status must be one of {sorted(MANUAL_STATUSES)}")

    case = get_case(_db, case_id)
    if case is None:
        raise HTTPException(404, f"no case {case_id!r}")
    if case["status"] not in ({"EVIDENCE_READY"} | MANUAL_STATUSES):
        raise HTTPException(409, f"case {case_id!r} is {case['status']!r}, not eligible for a manual status update")

    set_case_status(_db, case_id, req.status, note=req.note)
    return get_case(_db, case_id)


@app.get("/evidence/{case_id}")
def get_evidence(case_id: str):
    case = get_case(_db, case_id)
    if case is None:
        raise HTTPException(404, f"no case {case_id!r}")

    bundles = []
    for record in case["evidence"]:
        bundle_path = Path(record["artifact_uri"]) / "bundle.json"
        if bundle_path.exists():
            import json

            bundles.append(json.loads(bundle_path.read_text()))
    return {"caseId": case_id, "status": case["status"], "bundles": bundles}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8003")))
