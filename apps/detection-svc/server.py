"""detection-svc: Phase 3 infringement detection & evidence packaging
(PROJECT_DESIGN.md §3-7, §7). Same job-based HTTP shape as protection-svc's
server.py (POST -> 202 {caseId, status} / GET .../{caseId} to poll), but
cases persist in SQLite (src/db.py) rather than an in-memory dict --
evidence must survive a restart.

Scope: this implements runbook steps 1-3 of §7 (탐지/신고 접수 -> 자동 증거
수집 -> 증거 패키지 생성), plus two follow-ups that make those steps less
dependent on a human remembering to trigger them:
- Periodic auto-rescan (_auto_scan_loop): every registered artwork gets a
  background re-check on a rolling interval, not just at upload time or
  whenever a creator happens to click "웹에서 자동 검색".
- Evidence-ready email (notify_client.notify_evidence_ready): the creator
  doesn't have to keep the Test Lab tab open and polling to learn a case
  found something.
Runbook steps 4-6 proper (권리자의 실제 대응 판단, 테이크다운 접수, 케이스
추적 상태 갱신) remain product/human workflow, not automated here -- the
email above tells a creator a case needs their attention, it doesn't act
on their behalf.
"""

import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from threading import Thread

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))

from asset_client import ArtworkNotFoundError, get_artwork  # noqa: E402
from c2pa_verify import verify_c2pa  # noqa: E402
from db import add_evidence, connect, create_case, get_case, get_last_scan_time, set_case_status  # noqa: E402
from dmca_notice import build_dmca_notice, is_dmca_applicable  # noqa: E402
from evidence_bundle import build_bundle, write_json, write_pdf_best_effort  # noqa: E402
from evidence_capture import capture  # noqa: E402
from evidence_signing import sign_bundle  # noqa: E402
from notify_client import notify_evidence_ready  # noqa: E402
from phash_match import is_likely_match  # noqa: E402
from protection_client import LeakDetectionTimeoutError, download_suspect_model, poll_leak_detection_job, submit_leak_detection_job  # noqa: E402
from rust_watermark import detect_watermark  # noqa: E402
from vision import vision_configured, web_detect_matching_urls  # noqa: E402

ASSET_SERVICE_URL = os.environ.get("ASSET_SERVICE_URL", "http://localhost:3002")
PROTECTION_SVC_URL = os.environ.get("PROTECTION_SVC_URL", "http://localhost:8010")
PHASH_MATCH_THRESHOLD = int(os.environ.get("PHASH_MATCH_THRESHOLD", "20"))
DEFAULT_WATERMARK_HEX = os.environ.get("DEFAULT_WATERMARK_HEX", "deadbeefcafef00d")
OUT_DIR = Path(os.environ.get("DETECTION_OUT_DIR", str(Path(__file__).parent / "out")))
DB_PATH = os.environ.get("DETECTION_DB_PATH", str(Path(__file__).parent / "data" / "detection.db"))
# Same magnitude as a real LoRA .safetensors file (SD1.5 rank-32 LoRAs run
# tens of MB; generous headroom for higher ranks) without leaving the cap
# effectively unbounded -- see protection_client.download_suspect_model's
# doc for why this exists at all.
MODEL_LEAK_MAX_BYTES = int(os.environ.get("MODEL_LEAK_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2GB
MODEL_LEAK_POLL_TIMEOUT_SECONDS = float(os.environ.get("MODEL_LEAK_POLL_TIMEOUT_SECONDS", "1800"))  # 30min
# Opt-in, default off -- a background thread doing real outbound HTTP
# (asset-service + Vision API) at module-import time would be a surprising
# side effect for anything that just imports this module (every test file
# does exactly that). Production deployment sets this to "1".
AUTO_SCAN_ENABLED = os.environ.get("AUTO_SCAN_ENABLED") == "1"
AUTO_SCAN_POLL_INTERVAL_SECONDS = float(os.environ.get("AUTO_SCAN_POLL_INTERVAL_SECONDS", str(6 * 3600)))  # check every 6h
AUTO_RESCAN_INTERVAL_SECONDS = float(os.environ.get("AUTO_RESCAN_INTERVAL_SECONDS", str(7 * 24 * 3600)))  # re-scan an artwork at most once a week

app = FastAPI(title="detection-svc", version="0.1.1")
_db = connect(DB_PATH)


class ScanResponse(BaseModel):
    caseId: str
    status: str


class ReportRequest(BaseModel):
    artworkId: str
    suspectUrl: str


class ModelLeakReportRequest(BaseModel):
    artworkId: str
    suspectModelUrl: str  # a .safetensors LoRA file someone else published, suspected of being trained on this artwork


def _build_generic_prompts(artwork: dict) -> list[str]:
    """Deliberately trigger-word-free prompts (see ml-engine/src/
    model_leak_detect.py's module doc for why) -- built from the artwork's
    own title/tags, the only subject description this service has (same
    "no real per-image caption in the data model" gap
    remote_multiarch_cloak's docstring already flags for strong_protection).
    Two variants (title alone, title+tags) rather than one, so a single
    bad prompt doesn't decide the whole verdict."""
    title = artwork.get("title") or "an illustration"
    tags = artwork.get("tags")
    if isinstance(tags, str):
        import json as _json

        try:
            tags = _json.loads(tags)
        except (ValueError, TypeError):
            tags = []
    tags = tags or []

    prompts = [title]
    if tags:
        prompts.append(f"{title}, {', '.join(tags[:3])}")
    return prompts


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


def _notify_if_evidence_ready(artwork: dict, case_id: str, evidence_type: str) -> None:
    """Best-effort -- wrapped so a notification failure (api-gateway down,
    SMTP unreachable) can never turn an already-EVIDENCE_READY case back
    into FAILED. notify_client.notify_evidence_ready itself already
    catches httpx errors and returns False rather than raising; this
    outer guard is for anything else (a malformed artwork dict, etc)."""
    try:
        notify_evidence_ready(artwork.get("creatorId", ""), artwork.get("title") or "Untitled", case_id, evidence_type)
    except Exception:  # noqa: BLE001 -- notification is enrichment, never worth failing a case over
        pass


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

            # Second, independent provenance signal alongside pHash/watermark
            # -- but only supplementary evidence, never its own trigger for
            # is_match. A C2PA manifest is far easier for a redistribution to
            # strip than the invisible watermark is to defeat, so "no
            # manifest" proves nothing either way, and even a present,
            # DONTAI-signed manifest is corroboration, not independently
            # sufficient (see c2pa_verify.py's module doc).
            c2pa_result = None
            if captured.image_path:
                try:
                    image_format = Path(captured.image_path).suffix.lstrip(".")
                    c2pa = verify_c2pa(captured.image_path, image_format)
                    c2pa_result = {
                        "hasManifest": c2pa.has_manifest,
                        "signedByDontai": c2pa.signed_by_dontai,
                        "ownership": c2pa.ownership,
                        "validationIssues": c2pa.validation_issues,
                    }
                except (FileNotFoundError, RuntimeError):
                    c2pa_result = None

            if not is_match:
                continue

            any_evidence = True
            bundle = build_bundle(
                artwork=artwork,
                source_url=url,
                detected_at=captured.captured_at,
                phash_distance=phash_distance,
                watermark_result=watermark_result,
                c2pa_result=c2pa_result,
                headers=captured.headers,
                screenshot_path=captured.screenshot_path,
            )
            # Best-effort, like the screenshot/PDF steps around it -- a
            # signing failure (api-gateway down, KMS unreachable) degrades
            # to an unsigned bundle rather than losing the whole case.
            bundle["signature"] = sign_bundle({k: v for k, v in bundle.items() if k != "signature"})
            write_json(bundle, case_out_dir / "bundle.json")
            write_pdf_best_effort(bundle, case_out_dir / "bundle.pdf")

            confidence = 1.0 - (phash_distance / 256.0) if phash_distance is not None else None
            evidence_type = "watermark_match" if watermark_result and watermark_result["isMatch"] else "phash_match"
            add_evidence(_db, case_id, evidence_type, url, confidence, str(case_out_dir))

        set_case_status(_db, case_id, "EVIDENCE_READY" if any_evidence else "NO_MATCH_FOUND")
        if any_evidence:
            _notify_if_evidence_ready(artwork, case_id, "copy")
    except Exception as exc:  # noqa: BLE001 -- report failure via case status, mirrors protection-svc/server.py
        set_case_status(_db, case_id, "FAILED", f"{exc}\n{traceback.format_exc()}")


def _run_model_leak_case(case_id: str, artwork: dict, suspect_model_url: str) -> None:
    try:
        protected_uri = artwork.get("protectedImageUri")
        if not protected_uri or not Path(protected_uri).exists():
            set_case_status(_db, case_id, "FAILED", "artwork has no reachable protectedImageUri to compare against")
            return

        case_out_dir = OUT_DIR / case_id
        case_out_dir.mkdir(parents=True, exist_ok=True)
        model_path = case_out_dir / "suspect_model.safetensors"

        try:
            download_suspect_model(suspect_model_url, str(model_path), MODEL_LEAK_MAX_BYTES)
        except (httpx.HTTPError, ValueError) as exc:
            set_case_status(_db, case_id, "FAILED", f"could not download suspect model: {exc}")
            return

        prompts = _build_generic_prompts(artwork)
        job_id = submit_leak_detection_job(PROTECTION_SVC_URL, protected_uri, str(model_path), prompts)

        try:
            job = poll_leak_detection_job(PROTECTION_SVC_URL, job_id, MODEL_LEAK_POLL_TIMEOUT_SECONDS)
        except LeakDetectionTimeoutError as exc:
            set_case_status(_db, case_id, "FAILED", str(exc))
            return

        if job["status"] != "completed":
            set_case_status(_db, case_id, "FAILED", job.get("error", "model-leak detection job failed"))
            return

        verdict = job.get("verdict")
        any_evidence = verdict == "SUSPECTED_LEAK"

        bundle = build_bundle(
            artwork=artwork,
            source_url=suspect_model_url,
            detected_at=time.time(),
            model_leak_result=job,
        )
        bundle["signature"] = sign_bundle({k: v for k, v in bundle.items() if k != "signature"})
        write_json(bundle, case_out_dir / "bundle.json")
        write_pdf_best_effort(bundle, case_out_dir / "bundle.pdf")

        evidence_type = "model_leak" if any_evidence else "model_leak_no_match"
        add_evidence(_db, case_id, evidence_type, suspect_model_url, job.get("meanDelta"), str(case_out_dir))

        set_case_status(_db, case_id, "EVIDENCE_READY" if any_evidence else "NO_MATCH_FOUND")
        if any_evidence:
            _notify_if_evidence_ready(artwork, case_id, "model_leak")
    except Exception as exc:  # noqa: BLE001 -- report failure via case status, mirrors _run_case_for_urls
        set_case_status(_db, case_id, "FAILED", f"{exc}\n{traceback.format_exc()}")


def _safe_slug(url: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in url)[:80] or "url"


def _list_all_artworks() -> list[dict]:
    """No creatorId filter -- the auto-scan loop needs every published
    artwork on the platform, not one creator's, unlike everything else in
    this file (which only ever looks at one artwork a caller already named).
    """
    resp = httpx.get(f"{ASSET_SERVICE_URL}/artworks", timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def _run_auto_scan_pass() -> None:
    """One sweep: for every artwork due for a re-check (never scanned, or
    last scanned more than AUTO_RESCAN_INTERVAL_SECONDS ago), run the same
    web-search + evidence pipeline a manual POST /scan/{artworkId} would.
    Sequential, not parallelized across artworks -- this already runs on
    its own background thread and this project's scale doesn't need more
    concurrency than that; parallelizing would also mean juggling several
    Vision API calls at once for no real benefit yet.
    """
    now = time.time()
    for artwork in _list_all_artworks():
        protected_uri = artwork.get("protectedImageUri")
        if not protected_uri or not Path(protected_uri).exists():
            continue  # nothing published yet to check copies of

        last_scan = get_last_scan_time(_db, artwork["id"])
        if last_scan is not None and now - last_scan < AUTO_RESCAN_INTERVAL_SECONDS:
            continue

        case_id = f"case_{uuid.uuid4().hex[:12]}"
        create_case(_db, case_id, artwork["id"], "auto_scan")
        candidate_urls = web_detect_matching_urls(protected_uri) if vision_configured() else []
        _run_case_for_urls(case_id, artwork, candidate_urls)


def _auto_scan_loop() -> None:
    while True:
        time.sleep(AUTO_SCAN_POLL_INTERVAL_SECONDS)
        try:
            _run_auto_scan_pass()
        except Exception:  # noqa: BLE001 -- one bad sweep (asset-service down, etc) must not kill the loop forever
            traceback.print_exc()


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


@app.post("/model-leak-reports", status_code=202, response_model=ScanResponse)
async def submit_model_leak_report(req: ModelLeakReportRequest):
    """Detects whether a suspect LoRA/model file was trained on this
    artwork -- the "someone trained a model on my art" threat the original
    /scan and /reports endpoints don't cover (those only find re-posted
    copies of the image itself, via reverse image search / pHash /
    watermark). Compares against the artwork's *published* protected
    image (protectedImageUri), not the private original -- that's the
    only version a real infringer could ever have scraped and trained on.
    See ml-engine/src/model_leak_detect.py's module doc for the detection
    mechanism and protection_client.py for why this is the one place
    detection-svc calls out to protection-svc.
    """
    try:
        artwork = await get_artwork(ASSET_SERVICE_URL, req.artworkId)
    except ArtworkNotFoundError:
        raise HTTPException(404, f"no artwork {req.artworkId!r} on asset-service")

    case_id = f"case_{uuid.uuid4().hex[:12]}"
    create_case(_db, case_id, req.artworkId, "model_report")

    Thread(target=_run_model_leak_case, args=(case_id, artwork, req.suspectModelUrl), daemon=True).start()

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


@app.get("/cases/{case_id}/dmca-notice")
def get_dmca_notice(case_id: str):
    """RUNBOOK.md Step 5's DMCA template, filled in from each real evidence
    bundle this case produced -- see dmca_notice.py's module doc for what's
    auto-filled vs left as a bracketed placeholder, and why a model-leak
    bundle gets a note instead of a notice (it isn't a "URL hosting a copy
    of this work" situation)."""
    case = get_case(_db, case_id)
    if case is None:
        raise HTTPException(404, f"no case {case_id!r}")

    notices = []
    for record in case["evidence"]:
        bundle_path = Path(record["artifact_uri"]) / "bundle.json"
        if not bundle_path.exists():
            continue
        import json

        bundle = json.loads(bundle_path.read_text())
        if is_dmca_applicable(bundle):
            notices.append({"sourceUrl": bundle.get("discoveredUrl"), "notice": build_dmca_notice(bundle), "note": None})
        else:
            notices.append(
                {
                    "sourceUrl": bundle.get("discoveredUrl"),
                    "notice": None,
                    "note": "모델 학습 유출 의심 건은 DMCA 통지 대상이 아닙니다 -- RUNBOOK.md Step 5의 "
                    "'Suspected AI training dataset inclusion' 절차를 따르세요.",
                }
            )
    return {"caseId": case_id, "notices": notices}


if AUTO_SCAN_ENABLED:
    Thread(target=_auto_scan_loop, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8003")))
