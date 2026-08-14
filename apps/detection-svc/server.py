"""detection-svc: Phase 3 infringement detection & evidence packaging
(PROJECT_DESIGN.md §3-7, §7). Same job-based HTTP shape as protection-svc's
server.py (POST -> 202 {caseId, status} / GET .../{caseId} to poll), but
cases persist in SQLite (src/db.py) rather than an in-memory dict --
evidence must survive a restart.

Scope: this implements runbook steps 1-3 of §7 (탐지/신고 접수 -> 자동 증거
수집 -> 증거 패키지 생성), plus several follow-ups that make those steps less
dependent on a human remembering to trigger them:
- Periodic auto-rescan, two independent mechanisms:
  - _auto_scan_loop (AUTO_SCAN_ENABLED, off by default): a background
    thread that re-checks each artwork only once it's actually due
    (get_last_scan_time), on a simple sleep loop.
  - MonitorScheduler (MONITOR_ENABLED, off by default): a persistent-
    cadence scheduler (src/monitor_scheduler.py) that unconditionally
    sweeps every artwork every MONITOR_INTERVAL_SECONDS, applies Vision
    quota + candidate dedup, and records per-artwork monitoring_state for
    GET /monitor/status -- started/stopped via this app's lifespan and
    the /monitor/start,stop,run endpoints.
  Kept both rather than picking one: neither is a strict superset of the
  other yet (the first is cheaper and skips artworks that aren't due; the
  second gives an auditable per-artwork monitoring_state and quota/dedup
  bookkeeping), and each is independently opt-in via its own env var, so
  running both, either, or neither is a one-line config change.
- Evidence-ready email (notify_client.notify_evidence_ready): the creator
  doesn't have to keep the Test Lab tab open and polling to learn a case
  found something.
- 256-bit pHash BK-tree index (src/phash_index.py, PUT /index/artworks/{id}
  and POST /index/search) -- a persistent, searchable catalog of every
  registered artwork's perceptual hash, independent of any one case.
- Robust (mirror/rotation-invariant) fingerprinting and an explainable
  MATCH/REVIEW/NO_MATCH/INCONCLUSIVE verdict (src/robust_fingerprint.py,
  src/verdict.py) -- computed and recorded on every evidence bundle as
  additional signal, alongside (not yet replacing) the original pHash/
  watermark is_match check that decides whether a bundle gets built at
  all; see _run_case_for_urls's own doc for why that trigger hasn't
  changed.
- Evidence manifest integrity (src/evidence_integrity.py): every evidence
  directory gets a SHA-256 manifest.json of every artifact in it,
  optionally Ed25519-signed (EVIDENCE_SIGNING_KEY) and optionally sealed
  read-only (EVIDENCE_LOCAL_WRITE_ONCE) -- a local development emulation
  of write-once storage, not a substitute for real Object Lock/WORM. This
  is independent of, and in addition to, evidence_signing.py's own
  bundle-level Ed25519 signature via api-gateway's KMS-backed endpoint.
Runbook steps 4-6 proper (권리자의 실제 대응 판단, 테이크다운 접수, 케이스
추적 상태 갱신) remain product/human workflow, not automated here -- the
email above tells a creator a case needs their attention, it doesn't act
on their behalf.
"""

import asyncio
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # python-dotenv is a dev convenience, not a hard requirement
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from asset_client import ArtworkNotFoundError, get_artwork, list_artworks  # noqa: E402
from blockchain_client import anchor_evidence_bundle  # noqa: E402
from c2pa_verify import verify_c2pa  # noqa: E402
from c2pa_verifier import verify_c2pa as verify_c2pa_generic  # noqa: E402
from db import (  # noqa: E402
    add_evidence,
    connect,
    consume_vision_quota,
    create_case,
    filter_new_candidates,
    get_case,
    get_last_scan_time,
    get_monitoring_states,
    get_vision_usage,
    set_case_status,
    update_monitoring_state,
)
from dmca_notice import build_dmca_notice, is_dmca_applicable  # noqa: E402
from evidence_bundle import build_bundle, write_evidence_manifest, write_json, write_pdf_best_effort  # noqa: E402
from evidence_capture import capture  # noqa: E402
from evidence_integrity import EvidenceSigner, seal_directory, verify_evidence_directory  # noqa: E402
from evidence_signing import sign_bundle  # noqa: E402
from evidence_store import LocalWormEvidenceStore  # noqa: E402
from monitor_scheduler import MonitorScheduler  # noqa: E402
from notify_client import notify_evidence_ready  # noqa: E402
from phash_index import PHashIndex  # noqa: E402
from phash_match import is_likely_match  # noqa: E402
from protection_client import LeakDetectionTimeoutError, download_suspect_model, poll_leak_detection_job, submit_leak_detection_job  # noqa: E402
from robust_fingerprint import compare_robust_fingerprint  # noqa: E402
from rust_watermark import detect_watermark  # noqa: E402
from verdict import decide_verdict  # noqa: E402
from vision import vision_configured, web_detect_matching_urls  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent


def _local_path(value: str) -> str:
    if value == ":memory:":
        return value
    path = Path(value)
    return str(path if path.is_absolute() else BASE_DIR / path)


ASSET_SERVICE_URL = os.environ.get("ASSET_SERVICE_URL", "http://localhost:3002")
PROTECTION_SVC_URL = os.environ.get("PROTECTION_SVC_URL", "http://localhost:8010")
PHASH_MATCH_THRESHOLD = int(os.environ.get("PHASH_MATCH_THRESHOLD", "20"))
DEFAULT_WATERMARK_HEX = os.environ.get("DEFAULT_WATERMARK_HEX", "deadbeefcafef00d")
OUT_DIR = Path(_local_path(os.environ.get("DETECTION_OUT_DIR", "./out")))
DB_PATH = _local_path(os.environ.get("DETECTION_DB_PATH", "./data/detection.db"))
INDEX_PATH = _local_path(os.environ.get("PHASH_INDEX_PATH", "./data/phash-index.json"))
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
# Opt-in, default off -- see blockchain_client.py's module doc: a real
# (if testnet) relayer gas cost per EVIDENCE_READY case, and this
# project's own history already hit the relayer running low on funds
# unexpectedly once.
EVIDENCE_ANCHOR_ENABLED = os.environ.get("EVIDENCE_ANCHOR_ENABLED") == "1"
# Second, independent monitoring mechanism -- see this module's own
# docstring for why both this and AUTO_SCAN_ENABLED exist.
MONITOR_INTERVAL_SECONDS = int(os.environ.get("MONITOR_INTERVAL_SECONDS", "86400"))
MONITOR_RETRY_SECONDS = int(os.environ.get("MONITOR_RETRY_SECONDS", "300"))
MONITOR_ENABLED = os.environ.get("MONITOR_ENABLED", "false").lower() in {"1", "true", "yes"}
SIGNING_KEY_PATH_RAW = os.environ.get("EVIDENCE_SIGNING_KEY")
SIGNING_KEY_PATH = _local_path(SIGNING_KEY_PATH_RAW) if SIGNING_KEY_PATH_RAW else None
LOCAL_WRITE_ONCE = os.environ.get("EVIDENCE_LOCAL_WRITE_ONCE", "false").lower() in {"1", "true", "yes"}
# WORM evidence storage (2026-08-14, ported from the monitoring_detection
# handoff's evidence_store.py -- see that module's own doc). "filesystem"
# (default) keeps evidence directories exactly where they've always lived
# (OUT_DIR/case_id/...), matching every existing deployment's behavior with
# zero config changes required. "local_worm" additionally copies each
# case's evidence into a write-once-emulated store (unique object id,
# overwrite refusal, per-file SHA-256 manifest, append-only audit log) --
# still not a substitute for real Object Lock (see MOTECTION_OVERVIEW.md
# §10's own "실제 WORM/Object Lock... 미구현" note), but stronger than a
# plain writable directory for local/staging use.
EVIDENCE_STORE = os.environ.get("EVIDENCE_STORE", "filesystem").lower()
EVIDENCE_WORM_DIR = _local_path(os.environ.get("EVIDENCE_WORM_DIR", "./worm"))
EVIDENCE_RETENTION_DAYS = max(1, int(os.environ.get("EVIDENCE_RETENTION_DAYS", "365")))
VISION_MONTHLY_LIMIT = min(999, max(0, int(os.environ.get("VISION_MONTHLY_LIMIT", "900"))))
# Optional, scoped bearer auth for candidate image/page fetches -- sent
# only to the exact hostnames in CANDIDATE_AUTH_HOSTS, never to arbitrary
# candidates or redirect destinations (see evidence_capture._auth_headers).
# NOTE: currently unused/experimental in this service -- kept as documented,
# empty-by-default placeholders (see MOTECTION_HANDOFF.md's own
# "현재 사용하지 않음" note on this exact pair of variables). This is unrelated
# to api-gateway's DETECTION_SERVICE_TOKEN.
CANDIDATE_AUTH_BEARER_TOKEN = os.environ.get("CANDIDATE_AUTH_BEARER_TOKEN") or None
CANDIDATE_AUTH_HOSTS = {
    host.strip().lower() for host in os.environ.get("CANDIDATE_AUTH_HOSTS", "").split(",") if host.strip()
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if MONITOR_ENABLED:
        _scheduler.start()
    yield
    _scheduler.stop()


app = FastAPI(title="detection-svc", version="0.3.0", lifespan=lifespan)
_db = connect(DB_PATH)
_index = PHashIndex(INDEX_PATH)
_signer = (
    EvidenceSigner.from_pem(SIGNING_KEY_PATH, os.environ.get("EVIDENCE_SIGNING_KEY_ID", "local-ed25519"))
    if SIGNING_KEY_PATH
    else None
)
_worm_store = LocalWormEvidenceStore(EVIDENCE_WORM_DIR, EVIDENCE_RETENTION_DAYS) if EVIDENCE_STORE == "local_worm" else None


class ScanResponse(BaseModel):
    caseId: str
    status: str


class ReportRequest(BaseModel):
    artworkId: str
    suspectUrl: str


class ModelLeakReportRequest(BaseModel):
    artworkId: str
    suspectModelUrl: str  # a .safetensors LoRA file someone else published, suspected of being trained on this artwork


class IndexArtworkRequest(BaseModel):
    perceptualHash: str


class IndexSearchRequest(BaseModel):
    perceptualHash: str
    maxDistance: int = 20
    limit: int = 10


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


def _anchor_if_enabled(bundle_without_signature: dict, owner_address: str) -> dict | None:
    """Returns None immediately (no network call at all) unless
    EVIDENCE_ANCHOR_ENABLED -- see blockchain_client.py's module doc for
    why this defaults off. Wrapped the same best-effort way as
    _notify_if_evidence_ready: a failure here must never fail an
    otherwise-complete case."""
    if not EVIDENCE_ANCHOR_ENABLED:
        return None
    try:
        import json

        bundle_content = json.dumps(bundle_without_signature, sort_keys=True, default=str)
        return anchor_evidence_bundle(bundle_content, owner_address)
    except Exception:  # noqa: BLE001 -- anchoring is enrichment, never worth failing a case over
        return None


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
        access_issues: list[str] = []

        for url in candidate_urls:
            case_out_dir = OUT_DIR / case_id / _safe_slug(url)

            if CANDIDATE_AUTH_BEARER_TOKEN and CANDIDATE_AUTH_HOSTS:
                captured = asyncio.run(
                    capture(url, case_out_dir, bearer_token=CANDIDATE_AUTH_BEARER_TOKEN, auth_hosts=CANDIDATE_AUTH_HOSTS)
                )
            else:
                captured = asyncio.run(capture(url, case_out_dir))

            # evidence_capture.capture() may now return more than one
            # candidate image for a single URL (srcset/background-image/
            # JS-rendered network images on an SPA) -- check every one and
            # keep whichever the pHash/watermark check actually matches.
            # getattr with defaults, not direct attribute access: some
            # existing tests stub capture() with a plain namedtuple that
            # only has the original (image_path, headers, screenshot_path,
            # captured_at) fields, predating image_paths/access_status.
            candidate_image_paths = getattr(captured, "image_paths", None) or (
                [captured.image_path] if captured.image_path else []
            )
            if not candidate_image_paths:
                access_issues.append(getattr(captured, "access_status", "FETCH_ERROR"))
                continue

            found_match = False
            analysis = None  # (phash_distance, watermark_result, c2pa_result, fingerprint_result, verdict_result, image_path)
            for candidate_image_path in candidate_image_paths:
                phash_distance = None
                is_match = False
                if registered_hash:
                    is_match, phash_distance = is_likely_match(registered_hash, candidate_image_path, PHASH_MATCH_THRESHOLD)

                watermark_result = None
                try:
                    wm = detect_watermark(candidate_image_path, watermark_hex)
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
                try:
                    image_format = Path(candidate_image_path).suffix.lstrip(".")
                    c2pa = verify_c2pa(candidate_image_path, image_format)
                    c2pa_result = {
                        "hasManifest": c2pa.has_manifest,
                        "signedByDontai": c2pa.signed_by_dontai,
                        "ownership": c2pa.ownership,
                        "validationIssues": c2pa.validation_issues,
                    }
                except (FileNotFoundError, RuntimeError):
                    c2pa_result = None

                # Multi-view (mirror/rotate) fingerprint and the explainable
                # verdict aggregation are new, purely additive signals --
                # recorded on the bundle for a human reviewer, but (for now)
                # informational rather than part of the is_match trigger
                # above, so this merge doesn't change which cases reach
                # EVIDENCE_READY. Broad except (not just FileNotFoundError/
                # RuntimeError) because PIL/c2pa-python can raise their own
                # decode errors on a candidate image that downloaded fine
                # but isn't a valid/parseable image.
                fingerprint_result = None
                if registered_hash:
                    try:
                        fingerprint_result = compare_robust_fingerprint(
                            registered_hash, candidate_image_path, PHASH_MATCH_THRESHOLD
                        ).to_dict()
                    except Exception:  # noqa: BLE001 -- supplementary signal, same posture as c2pa/screenshot
                        fingerprint_result = None

                c2pa_status = None
                try:
                    c2pa_status = verify_c2pa_generic(candidate_image_path).status
                except Exception:  # noqa: BLE001 -- supplementary signal only
                    c2pa_status = None

                verdict_result = decide_verdict(
                    phash_distance=phash_distance,
                    watermark_match=watermark_result["isMatch"] if watermark_result else None,
                    fingerprint_distance=fingerprint_result["distance"] if fingerprint_result else None,
                    c2pa_status=c2pa_status,
                    match_threshold=PHASH_MATCH_THRESHOLD,
                )

                if analysis is None or is_match:
                    analysis = (phash_distance, watermark_result, c2pa_result, fingerprint_result, verdict_result, candidate_image_path)
                if is_match:
                    found_match = True
                    break

            if not found_match:
                continue
            phash_distance, watermark_result, c2pa_result, fingerprint_result, verdict_result, matched_image_path = analysis

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
                fingerprint_result=fingerprint_result,
                verdict_result=verdict_result,
            )
            bundle["matchedImagePath"] = matched_image_path
            # Best-effort, like the screenshot/PDF steps around it -- a
            # signing failure (api-gateway down, KMS unreachable) degrades
            # to an unsigned bundle rather than losing the whole case.
            bundle_without_signature = {k: v for k, v in bundle.items() if k != "signature"}
            bundle["signature"] = sign_bundle(bundle_without_signature)
            bundle["evidenceAnchor"] = _anchor_if_enabled(bundle_without_signature, artwork.get("ownerWalletAddress", ""))
            write_json(bundle, case_out_dir / "bundle.json")
            write_pdf_best_effort(bundle, case_out_dir / "bundle.pdf")

            # File-level integrity manifest (independent of the bundle's own
            # `signature` field above) -- see evidence_integrity.py's module
            # doc. Optionally Ed25519-signed and optionally sealed read-only,
            # both opt-in and both local-dev emulations, not a substitute
            # for real KMS/WORM.
            manifest_path = write_evidence_manifest(case_out_dir)
            if _signer:
                _signer.sign_manifest(manifest_path)
            if LOCAL_WRITE_ONCE:
                seal_directory(case_out_dir)

            confidence = 1.0 - (phash_distance / 256.0) if phash_distance is not None else None
            evidence_type = "watermark_match" if watermark_result and watermark_result["isMatch"] else "phash_match"
            artifact_uri = _store_evidence_artifact(case_out_dir, case_id, url)
            add_evidence(_db, case_id, evidence_type, url, confidence, artifact_uri)

        if any_evidence:
            final_status = "EVIDENCE_READY"
        elif access_issues and all(status == "AUTH_REQUIRED" for status in access_issues):
            final_status = "AUTH_REQUIRED"
        elif access_issues and all(status in {"ACCESS_DENIED", "AUTH_REQUIRED"} for status in access_issues):
            final_status = "ACCESS_DENIED"
        else:
            final_status = "NO_MATCH_FOUND"
        set_case_status(_db, case_id, final_status)
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
        bundle_without_signature = {k: v for k, v in bundle.items() if k != "signature"}
        bundle["signature"] = sign_bundle(bundle_without_signature)
        bundle["evidenceAnchor"] = _anchor_if_enabled(bundle_without_signature, artwork.get("ownerWalletAddress", ""))
        write_json(bundle, case_out_dir / "bundle.json")
        write_pdf_best_effort(bundle, case_out_dir / "bundle.pdf")

        evidence_type = "model_leak" if any_evidence else "model_leak_no_match"
        artifact_uri = _store_evidence_artifact(case_out_dir, case_id, suspect_model_url)
        add_evidence(_db, case_id, evidence_type, suspect_model_url, job.get("meanDelta"), artifact_uri)

        set_case_status(_db, case_id, "EVIDENCE_READY" if any_evidence else "NO_MATCH_FOUND")
        if any_evidence:
            _notify_if_evidence_ready(artwork, case_id, "model_leak")
    except Exception as exc:  # noqa: BLE001 -- report failure via case status, mirrors _run_case_for_urls
        set_case_status(_db, case_id, "FAILED", f"{exc}\n{traceback.format_exc()}")


def _safe_slug(url: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in url)[:80] or "url"


def _store_evidence_artifact(case_out_dir: Path, case_id: str, evidence_id: str) -> str:
    """Returns the artifact_uri to record in the evidence table -- the
    plain on-disk case_out_dir when EVIDENCE_STORE=filesystem (the default,
    every existing deployment's behavior), or the WORM store's own uri once
    the directory's been copied into it when EVIDENCE_STORE=local_worm (see
    this module's own EVIDENCE_STORE doc). Not best-effort like the
    signing/sealing steps around its call sites -- a WORM put failure here
    means the evidence was never actually retained the way the operator
    configured it to be, which should surface as a real case FAILED status
    (via the caller's own try/except), not a silently downgraded artifact_uri.
    """
    if not _worm_store:
        return str(case_out_dir)
    stored = _worm_store.put(case_out_dir, case_id=case_id, evidence_id=_safe_slug(evidence_id))
    return stored.uri


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


def _scan_all_artworks() -> None:
    """Scheduled sweep for MonitorScheduler (MONITOR_ENABLED) -- unlike
    _run_auto_scan_pass (AUTO_SCAN_ENABLED) above, this unconditionally
    re-scans every artwork every MONITOR_INTERVAL_SECONDS, applies Vision
    quota (consume_vision_quota) and candidate dedup (filter_new_candidates),
    keeps the pHash index (src/phash_index.py) up to date, and records
    per-artwork monitoring_state for GET /monitor/status. See this module's
    own docstring for why both mechanisms are kept.
    """
    artworks = asyncio.run(list_artworks(ASSET_SERVICE_URL))
    next_scan = time.time() + MONITOR_INTERVAL_SECONDS
    for artwork in artworks:
        artwork_id = artwork.get("id")
        if not artwork_id:
            continue
        protected_uri = artwork.get("protectedImageUri")
        if not protected_uri or not Path(protected_uri).exists():
            update_monitoring_state(_db, artwork_id, status="SKIPPED_NO_PROTECTED_IMAGE", next_scan_at=next_scan)
            continue
        try:
            registered_hash = artwork.get("perceptualHash")
            if registered_hash:
                try:
                    _index.upsert(artwork_id, registered_hash)
                except ValueError:
                    pass  # legacy short PoC hash predating the 256-bit format
            candidate_urls = []
            if vision_configured() and VISION_MONTHLY_LIMIT > 0 and consume_vision_quota(_db, VISION_MONTHLY_LIMIT):
                candidate_urls = web_detect_matching_urls(protected_uri)
            candidate_urls = filter_new_candidates(_db, artwork_id, candidate_urls)
            case_id = f"case_{uuid.uuid4().hex[:12]}"
            create_case(_db, case_id, artwork_id, "auto_scan")
            Thread(target=_run_case_for_urls, args=(case_id, artwork, candidate_urls), daemon=True).start()
            update_monitoring_state(_db, artwork_id, status=f"QUEUED:{case_id}:{len(candidate_urls)}", next_scan_at=next_scan)
        except Exception as exc:  # noqa: BLE001 -- one bad artwork must not abort the whole sweep
            update_monitoring_state(_db, artwork_id, status="FAILED", next_scan_at=next_scan, error=str(exc))


_scheduler = MonitorScheduler(_scan_all_artworks, MONITOR_INTERVAL_SECONDS, MONITOR_RETRY_SECONDS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/details")
def health_details():
    return {
        "status": "ok",
        "indexedArtworks": len(_index),
        "monitor": _scheduler.status(),
        "autoScan": {
            "enabled": AUTO_SCAN_ENABLED,
            "pollIntervalSeconds": AUTO_SCAN_POLL_INTERVAL_SECONDS,
            "rescanIntervalSeconds": AUTO_RESCAN_INTERVAL_SECONDS,
        },
        "evidenceSigning": bool(_signer),
        "evidenceStore": {"backend": EVIDENCE_STORE, "retentionDays": EVIDENCE_RETENTION_DAYS},
        "vision": {"configured": vision_configured(), **get_vision_usage(_db, VISION_MONTHLY_LIMIT)},
    }


@app.get("/vision/usage")
def vision_usage():
    return {"configured": vision_configured(), **get_vision_usage(_db, VISION_MONTHLY_LIMIT)}


@app.post("/scan/{artwork_id}", status_code=202, response_model=ScanResponse)
async def scan_artwork(artwork_id: str):
    try:
        artwork = await get_artwork(ASSET_SERVICE_URL, artwork_id)
    except ArtworkNotFoundError:
        raise HTTPException(404, f"no artwork {artwork_id!r} on asset-service")

    protected_uri = artwork.get("protectedImageUri")
    if not protected_uri or not Path(protected_uri).exists():
        raise HTTPException(400, f"artwork {artwork_id!r} has no reachable protectedImageUri to scan from")

    registered_hash = artwork.get("perceptualHash")
    if registered_hash:
        try:
            _index.upsert(artwork_id, registered_hash)
        except ValueError:
            pass  # legacy short PoC hash predating the 256-bit format; scanning still proceeds

    case_id = f"case_{uuid.uuid4().hex[:12]}"
    create_case(_db, case_id, artwork_id, "scan")

    candidate_urls = []
    if vision_configured() and VISION_MONTHLY_LIMIT > 0 and consume_vision_quota(_db, VISION_MONTHLY_LIMIT):
        candidate_urls = web_detect_matching_urls(protected_uri)
    candidate_urls = filter_new_candidates(_db, artwork_id, candidate_urls)
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


@app.get("/evidence/{case_id}/verify")
def verify_evidence(case_id: str):
    """Independent of GET /evidence/{caseId}'s bundle-level `signature` --
    re-checks each evidence directory's file-level manifest.json (SHA-256
    per file, optional Ed25519 signature) against what's actually on disk
    right now. See evidence_integrity.py's module doc."""
    case = get_case(_db, case_id)
    if case is None:
        raise HTTPException(404, f"no case {case_id!r}")
    results = []
    for record in case["evidence"]:
        manifest = Path(record["artifact_uri"]) / "manifest.json"
        if manifest.exists():
            results.append({"artifactUri": record["artifact_uri"], **verify_evidence_directory(record["artifact_uri"])})
    return {"caseId": case_id, "manifests": results}


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


@app.put("/index/artworks/{artwork_id}")
def index_artwork(artwork_id: str, req: IndexArtworkRequest):
    """Standalone entry point into the pHash BK-tree index (src/phash_index.py)
    -- /scan already keeps this up to date for scanned artworks, but a
    caller (or a backfill script) can index an artwork directly without
    triggering a scan."""
    _index.upsert(artwork_id, req.perceptualHash)
    return {"artworkId": artwork_id, "indexed": True, "indexSize": len(_index)}


@app.post("/index/search")
def search_index(req: IndexSearchRequest):
    """Reverse lookup: "which registered artworks does this hash resemble" --
    the inverse direction of the usual /scan flow (one artwork -> many
    candidate URLs). Useful for e.g. checking a newly uploaded artwork
    against the existing catalog for near-duplicates before it's even
    published."""
    if not 0 <= req.maxDistance <= 256 or not 1 <= req.limit <= 100:
        raise HTTPException(400, "maxDistance must be 0..256 and limit must be 1..100")
    return {"matches": _index.search(req.perceptualHash, req.maxDistance, req.limit)}


@app.post("/monitor/run", status_code=202)
def run_monitor_now():
    Thread(target=_scheduler.run_once, daemon=True).start()
    return {"status": "queued"}


@app.get("/monitor/status")
def monitor_status():
    return {**_scheduler.status(), "artworks": get_monitoring_states(_db)}


@app.post("/monitor/start")
def start_monitor():
    _scheduler.start()
    return _scheduler.status()


@app.post("/monitor/stop")
def stop_monitor():
    _scheduler.stop()
    return _scheduler.status()


if AUTO_SCAN_ENABLED:
    Thread(target=_auto_scan_loop, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8003")))
