"""HTTP-contract tests for server.py. asset-service and Vision calls are
mocked (monkeypatch on the names server.py imported directly, since that's
where Python actually looks them up) -- no real network, matching
asset-service's own vi.mock-based test approach in spirit.

The background evidence-collection Thread is stubbed to a synchronous
no-op here: the pipeline pieces it calls (phash_match, evidence_bundle,
rust_watermark) already have their own focused unit tests. This file only
verifies the request/response contract: status codes, body shapes, and the
404 paths.
"""

import os
import sys
from pathlib import Path

os.environ["DETECTION_DB_PATH"] = ":memory:"
os.environ["DETECTION_OUT_DIR"] = "/tmp/detection-svc-test-out"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import server
from asset_client import ArtworkNotFoundError

FAKE_ARTWORK = {
    "id": "ast_abc",
    "title": "Starry Fields",
    "creatorId": "user_creator1",
    "perceptualHash": "0xaaaa",
    "protectedImageUri": __file__,  # any existing file path works for the "exists" check
    "ownerWalletAddress": "0x1234567890abcdef1234567890ABCDEF12345678",
    "ownershipRecords": [],
}


@pytest.fixture
def client(monkeypatch):
    # in-memory sqlite means each TestClient instance needs its own
    # connection; server.py opens _db once at import time, which is fine
    # for :memory: since the module-level connection persists for the
    # whole test session (mirrors production's single-process assumption).
    monkeypatch.setattr(server, "_run_case_for_urls", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_run_model_leak_case", lambda *a, **kw: None)
    return TestClient(server.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_unknown_case_is_404(client):
    resp = client.get("/cases/case_does_not_exist")
    assert resp.status_code == 404


def test_scan_unknown_artwork_is_404(client, monkeypatch):
    async def fake_get_artwork(url, artwork_id):
        raise ArtworkNotFoundError(artwork_id)

    monkeypatch.setattr(server, "get_artwork", fake_get_artwork)

    resp = client.post("/scan/ast_missing")
    assert resp.status_code == 404


def test_scan_known_artwork_returns_202_and_case_id(client, monkeypatch):
    async def fake_get_artwork(url, artwork_id):
        return FAKE_ARTWORK

    monkeypatch.setattr(server, "get_artwork", fake_get_artwork)
    monkeypatch.setattr(server, "vision_configured", lambda: False)

    resp = client.post("/scan/ast_abc")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["caseId"].startswith("case_")

    case = client.get(f"/cases/{body['caseId']}").json()
    assert case["artwork_id"] == "ast_abc"
    assert case["trigger"] == "scan"


def test_scan_artwork_without_protected_image_is_400(client, monkeypatch):
    async def fake_get_artwork(url, artwork_id):
        return {**FAKE_ARTWORK, "protectedImageUri": "/no/such/file.png"}

    monkeypatch.setattr(server, "get_artwork", fake_get_artwork)

    resp = client.post("/scan/ast_abc")
    assert resp.status_code == 400


def test_report_creates_case_with_report_trigger(client, monkeypatch):
    async def fake_get_artwork(url, artwork_id):
        return FAKE_ARTWORK

    monkeypatch.setattr(server, "get_artwork", fake_get_artwork)

    resp = client.post("/reports", json={"artworkId": "ast_abc", "suspectUrl": "https://example.com/x.png"})
    assert resp.status_code == 202
    case_id = resp.json()["caseId"]

    case = client.get(f"/cases/{case_id}").json()
    assert case["trigger"] == "report"


def test_evidence_for_unknown_case_is_404(client):
    resp = client.get("/evidence/case_does_not_exist")
    assert resp.status_code == 404


def test_dmca_notice_for_unknown_case_is_404(client):
    resp = client.get("/cases/case_does_not_exist/dmca-notice")
    assert resp.status_code == 404


def test_dmca_notice_fills_in_a_real_copy_bundle_and_flags_a_model_leak_bundle(client, tmp_path):
    import json

    from db import add_evidence, create_case

    case_id = "case_dmca_1"
    create_case(server._db, case_id, "ast_abc", "report")

    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    (copy_dir / "bundle.json").write_text(
        json.dumps(
            {
                "originalHash": "0xaaaa",
                "watermarkDetection": None,
                "c2paDetection": None,
                "modelLeakDetection": None,
                "discoveredUrl": "https://example.com/stolen.png",
                "discoveredAt": 1_700_000_000.0,
                "phashDistance": 4,
                "onchainTransaction": None,
                "registeredAt": None,
            }
        )
    )
    add_evidence(server._db, case_id, "phash_match", "https://example.com/stolen.png", 0.9, str(copy_dir))

    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    (leak_dir / "bundle.json").write_text(
        json.dumps({"modelLeakDetection": {"verdict": "SUSPECTED_LEAK"}, "discoveredUrl": "https://civitai.com/models/1"})
    )
    add_evidence(server._db, case_id, "model_leak", "https://civitai.com/models/1", 0.1, str(leak_dir))

    res = client.get(f"/cases/{case_id}/dmca-notice")
    assert res.status_code == 200
    notices = res.json()["notices"]
    assert len(notices) == 2

    copy_notice = next(n for n in notices if n["sourceUrl"] == "https://example.com/stolen.png")
    assert copy_notice["note"] is None
    assert "https://example.com/stolen.png" in copy_notice["notice"]
    assert "0xaaaa" in copy_notice["notice"]

    leak_notice = next(n for n in notices if n["sourceUrl"] == "https://civitai.com/models/1")
    assert leak_notice["notice"] is None
    assert "AI training dataset" in leak_notice["note"]


def test_patch_case_moves_evidence_ready_to_notified(client):
    from db import create_case, set_case_status

    case_id = "case_manual1"
    create_case(server._db, case_id, "ast_x", "scan")
    set_case_status(server._db, case_id, "EVIDENCE_READY")

    resp = client.patch(f"/cases/{case_id}", json={"status": "NOTIFIED", "note": "emailed creator 2026-01-01"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "NOTIFIED"
    assert body["note"] == "emailed creator 2026-01-01"


def test_patch_case_rejects_status_not_in_the_manual_set(client):
    from db import create_case, set_case_status

    case_id = "case_manual2"
    create_case(server._db, case_id, "ast_x", "scan")
    set_case_status(server._db, case_id, "EVIDENCE_READY")

    resp = client.patch(f"/cases/{case_id}", json={"status": "PUBLISHED"})
    assert resp.status_code == 400


def test_patch_case_rejects_transition_from_a_non_evidence_ready_automated_state(client):
    from db import create_case

    case_id = "case_manual3"
    create_case(server._db, case_id, "ast_x", "scan")  # still OPEN

    resp = client.patch(f"/cases/{case_id}", json={"status": "NOTIFIED"})
    assert resp.status_code == 409


def test_patch_unknown_case_is_404(client):
    resp = client.patch("/cases/case_does_not_exist", json={"status": "NOTIFIED"})
    assert resp.status_code == 404


# _run_case_for_urls itself is stubbed out for every test above (see this
# file's module docstring) -- none of them exercise the real wiring that
# calls verify_c2pa(), builds the c2paDetection dict from its result, and
# falls back to None on failure. These two tests call it directly instead,
# mocking only its immediate dependencies (capture/is_likely_match/
# detect_watermark/verify_c2pa/sign_bundle/write_json/write_pdf_best_effort),
# so build_bundle's actual c2pa_result handling runs for real.
class _FakeWatermarkResult:
    def __init__(self):
        self.recovered_hex = "deadbeef"
        self.avg_confidence = 0.5
        self.min_confidence = 0.5
        self.bit_error_rate = 0.5
        self.is_match = False  # phash alone drives is_match in these tests


def _stub_common_run_case_deps(monkeypatch, tmp_path):
    from evidence_capture import CapturedEvidence

    async def fake_capture(url, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "candidate_image.png"
        image_path.write_bytes(b"fake")
        return CapturedEvidence(str(image_path), {"content-type": "image/png"}, None, 1_700_000_000.0)

    captured_bundles = []

    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    monkeypatch.setattr(server, "capture", fake_capture)
    monkeypatch.setattr(server, "is_likely_match", lambda registered_hash, path, threshold: (True, 5))
    monkeypatch.setattr(server, "detect_watermark", lambda path, hexpayload: _FakeWatermarkResult())
    monkeypatch.setattr(server, "sign_bundle", lambda bundle: {"signature": "fake-sig", "publicKeyPem": "fake-pem", "algorithm": "ed25519"})
    monkeypatch.setattr(server, "write_json", lambda bundle, path: captured_bundles.append(bundle))
    monkeypatch.setattr(server, "write_pdf_best_effort", lambda bundle, path: None)
    monkeypatch.setattr(server, "notify_evidence_ready", lambda *a, **kw: False)
    return captured_bundles


def test_run_case_for_urls_wires_a_real_c2pa_match_into_the_bundle(monkeypatch, tmp_path):
    from db import create_case, get_case
    from c2pa_verify import C2paVerifyResult

    captured_bundles = _stub_common_run_case_deps(monkeypatch, tmp_path)
    fake_result = C2paVerifyResult(
        manifest={
            "active_manifest": "urn:c2pa:abc",
            "manifests": {
                "urn:c2pa:abc": {
                    "assertions": [{"label": "com.dontai.ownership", "data": {"title": "t", "doNotTrain": True}}],
                    "signature_info": {"issuer": "DONTAI"},
                }
            },
        },
        validation_issues=["signingCredential.untrusted: signing certificate untrusted"],
    )
    monkeypatch.setattr(server, "verify_c2pa", lambda path, fmt: fake_result)

    case_id = "case_c2pa_wired_match"
    create_case(server._db, case_id, "ast_abc", "report")

    server._run_case_for_urls(case_id, FAKE_ARTWORK, ["https://example.com/found.png"])

    assert len(captured_bundles) == 1
    c2pa_detection = captured_bundles[0]["c2paDetection"]
    assert c2pa_detection == {
        "hasManifest": True,
        "signedByDontai": True,
        "ownership": {"title": "t", "doNotTrain": True},
        "validationIssues": ["signingCredential.untrusted: signing certificate untrusted"],
    }

    case = get_case(server._db, case_id)
    assert case["status"] == "EVIDENCE_READY"


def test_model_leak_report_unknown_artwork_is_404(client, monkeypatch):
    async def fake_get_artwork(url, artwork_id):
        raise ArtworkNotFoundError(artwork_id)

    monkeypatch.setattr(server, "get_artwork", fake_get_artwork)

    resp = client.post("/model-leak-reports", json={"artworkId": "ast_missing", "suspectModelUrl": "https://example.com/x.safetensors"})
    assert resp.status_code == 404


def test_model_leak_report_creates_case_with_model_report_trigger(client, monkeypatch):
    async def fake_get_artwork(url, artwork_id):
        return FAKE_ARTWORK

    monkeypatch.setattr(server, "get_artwork", fake_get_artwork)

    resp = client.post("/model-leak-reports", json={"artworkId": "ast_abc", "suspectModelUrl": "https://example.com/x.safetensors"})
    assert resp.status_code == 202
    case_id = resp.json()["caseId"]

    case = client.get(f"/cases/{case_id}").json()
    assert case["artwork_id"] == "ast_abc"
    assert case["trigger"] == "model_report"


def test_build_generic_prompts_uses_title_and_parses_json_string_tags():
    prompts = server._build_generic_prompts({"title": "Starry Fields", "tags": '["oil painting", "landscape", "night"]'})
    assert prompts[0] == "Starry Fields"
    assert prompts[1] == "Starry Fields, oil painting, landscape, night"


def test_build_generic_prompts_falls_back_when_no_tags():
    prompts = server._build_generic_prompts({"title": "Untitled Work"})
    assert prompts == ["Untitled Work"]


# _run_model_leak_case's own immediate dependencies (download_suspect_model,
# submit_leak_detection_job, poll_leak_detection_job, sign_bundle,
# write_json, write_pdf_best_effort) are mocked here -- same posture as
# _stub_common_run_case_deps above -- so the real case-status/evidence-
# record wiring runs for real.
def _stub_model_leak_deps(monkeypatch, tmp_path, job_result):
    captured_bundles = []

    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    monkeypatch.setattr(server, "download_suspect_model", lambda url, dest, max_bytes: Path(dest).write_bytes(b"fake"))
    monkeypatch.setattr(server, "submit_leak_detection_job", lambda *a, **kw: "leakjob_fake123")
    monkeypatch.setattr(server, "poll_leak_detection_job", lambda *a, **kw: job_result)
    monkeypatch.setattr(server, "sign_bundle", lambda bundle: {"signature": "fake-sig", "publicKeyPem": "fake-pem", "algorithm": "ed25519"})
    monkeypatch.setattr(server, "write_json", lambda bundle, path: captured_bundles.append(bundle))
    monkeypatch.setattr(server, "write_pdf_best_effort", lambda bundle, path: None)
    monkeypatch.setattr(server, "notify_evidence_ready", lambda *a, **kw: False)
    return captured_bundles


def test_run_model_leak_case_suspected_leak_reaches_evidence_ready(monkeypatch, tmp_path):
    from db import create_case, get_case

    job_result = {
        "status": "completed",
        "perPrompt": [{"prompt": "t", "avgBaseSimilarity": 0.5, "avgSuspectSimilarity": 0.7, "delta": 0.2}],
        "meanDelta": 0.2,
        "stdevDelta": 0.0,
        "verdict": "SUSPECTED_LEAK",
        "threshold": 0.03,
    }
    captured_bundles = _stub_model_leak_deps(monkeypatch, tmp_path, job_result)

    case_id = "case_leak_suspected"
    create_case(server._db, case_id, "ast_abc", "model_report")

    server._run_model_leak_case(case_id, FAKE_ARTWORK, "https://example.com/suspect.safetensors")

    case = get_case(server._db, case_id)
    assert case["status"] == "EVIDENCE_READY"
    assert len(case["evidence"]) == 1
    assert case["evidence"][0]["evidence_type"] == "model_leak"
    assert case["evidence"][0]["confidence"] == 0.2

    assert len(captured_bundles) == 1
    assert captured_bundles[0]["modelLeakDetection"] == job_result


def test_run_model_leak_case_no_evidence_reaches_no_match_found(monkeypatch, tmp_path):
    from db import create_case, get_case

    job_result = {"status": "completed", "perPrompt": [], "meanDelta": 0.0, "stdevDelta": 0.0, "verdict": "NO_EVIDENCE", "threshold": 0.03}
    _stub_model_leak_deps(monkeypatch, tmp_path, job_result)

    case_id = "case_leak_no_evidence"
    create_case(server._db, case_id, "ast_abc", "model_report")

    server._run_model_leak_case(case_id, FAKE_ARTWORK, "https://example.com/suspect.safetensors")

    case = get_case(server._db, case_id)
    assert case["status"] == "NO_MATCH_FOUND"
    assert case["evidence"][0]["evidence_type"] == "model_leak_no_match"


def test_run_model_leak_case_fails_without_reachable_protected_image(monkeypatch, tmp_path):
    from db import create_case, get_case

    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    case_id = "case_leak_no_image"
    create_case(server._db, case_id, "ast_abc", "model_report")

    server._run_model_leak_case(case_id, {**FAKE_ARTWORK, "protectedImageUri": "/no/such/file.png"}, "https://example.com/x.safetensors")

    case = get_case(server._db, case_id)
    assert case["status"] == "FAILED"


def test_run_model_leak_case_fails_when_download_raises(monkeypatch, tmp_path):
    from db import create_case, get_case

    monkeypatch.setattr(server, "OUT_DIR", tmp_path)

    def raising_download(url, dest, max_bytes):
        raise ValueError("suspect model exceeds size limit")

    monkeypatch.setattr(server, "download_suspect_model", raising_download)

    case_id = "case_leak_download_fail"
    create_case(server._db, case_id, "ast_abc", "model_report")

    server._run_model_leak_case(case_id, FAKE_ARTWORK, "https://example.com/huge.safetensors")

    case = get_case(server._db, case_id)
    assert case["status"] == "FAILED"
    assert "size limit" in case["error_message"]


def test_run_model_leak_case_fails_when_protection_svc_job_itself_failed(monkeypatch, tmp_path):
    from db import create_case, get_case

    job_result = {"status": "failed", "error": "GPU exploded"}
    _stub_model_leak_deps(monkeypatch, tmp_path, job_result)

    case_id = "case_leak_job_failed"
    create_case(server._db, case_id, "ast_abc", "model_report")

    server._run_model_leak_case(case_id, FAKE_ARTWORK, "https://example.com/x.safetensors")

    case = get_case(server._db, case_id)
    assert case["status"] == "FAILED"
    assert "GPU exploded" in case["error_message"]


def test_run_model_leak_case_fails_on_poll_timeout(monkeypatch, tmp_path):
    from db import create_case, get_case
    from protection_client import LeakDetectionTimeoutError

    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    monkeypatch.setattr(server, "download_suspect_model", lambda url, dest, max_bytes: Path(dest).write_bytes(b"fake"))
    monkeypatch.setattr(server, "submit_leak_detection_job", lambda *a, **kw: "leakjob_fake123")

    def raising_poll(*a, **kw):
        raise LeakDetectionTimeoutError("protection-svc job leakjob_fake123 did not finish within 1800s")

    monkeypatch.setattr(server, "poll_leak_detection_job", raising_poll)

    case_id = "case_leak_timeout"
    create_case(server._db, case_id, "ast_abc", "model_report")

    server._run_model_leak_case(case_id, FAKE_ARTWORK, "https://example.com/x.safetensors")

    case = get_case(server._db, case_id)
    assert case["status"] == "FAILED"
    assert "did not finish" in case["error_message"]


def test_run_case_for_urls_notifies_the_creator_when_evidence_is_found(monkeypatch, tmp_path):
    from db import create_case

    _stub_common_run_case_deps(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(server, "notify_evidence_ready", lambda creator_id, title, case_id, evidence_type: calls.append((creator_id, title, case_id, evidence_type)))

    case_id = "case_notify_copy"
    create_case(server._db, case_id, "ast_abc", "report")
    server._run_case_for_urls(case_id, FAKE_ARTWORK, ["https://example.com/found.png"])

    assert calls == [("user_creator1", "Starry Fields", case_id, "copy")]


def test_run_case_for_urls_does_not_notify_on_no_match(monkeypatch, tmp_path):
    from db import create_case

    _stub_common_run_case_deps(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "is_likely_match", lambda registered_hash, path, threshold: (False, 200))
    calls = []
    monkeypatch.setattr(server, "notify_evidence_ready", lambda *a, **kw: calls.append(a))

    case_id = "case_notify_no_match"
    create_case(server._db, case_id, "ast_abc", "report")
    server._run_case_for_urls(case_id, FAKE_ARTWORK, ["https://example.com/found.png"])

    assert calls == []


def test_anchor_if_enabled_returns_none_without_calling_anchor_when_disabled(monkeypatch):
    monkeypatch.setattr(server, "EVIDENCE_ANCHOR_ENABLED", False)
    calls = []
    monkeypatch.setattr(server, "anchor_evidence_bundle", lambda *a, **kw: calls.append(a))

    result = server._anchor_if_enabled({"a": 1}, "0xOWNER")

    assert result is None
    assert calls == []


def test_anchor_if_enabled_calls_through_when_enabled(monkeypatch):
    monkeypatch.setattr(server, "EVIDENCE_ANCHOR_ENABLED", True)
    captured = {}

    def fake_anchor(content, owner_address):
        captured["content"] = content
        captured["owner_address"] = owner_address
        return {"txHash": "0xTX", "blockNumber": 1, "contentHash": "0xHASH"}

    monkeypatch.setattr(server, "anchor_evidence_bundle", fake_anchor)

    result = server._anchor_if_enabled({"a": 1}, "0xOWNER")

    assert result == {"txHash": "0xTX", "blockNumber": 1, "contentHash": "0xHASH"}
    assert captured["owner_address"] == "0xOWNER"
    assert '"a": 1' in captured["content"]


def test_anchor_if_enabled_swallows_a_failure_instead_of_raising(monkeypatch):
    monkeypatch.setattr(server, "EVIDENCE_ANCHOR_ENABLED", True)

    def raising_anchor(*a, **kw):
        raise RuntimeError("blockchain-svc unreachable")

    monkeypatch.setattr(server, "anchor_evidence_bundle", raising_anchor)

    assert server._anchor_if_enabled({"a": 1}, "0xOWNER") is None


def test_run_case_for_urls_leaves_evidence_anchor_none_by_default(monkeypatch, tmp_path):
    from db import create_case

    captured_bundles = _stub_common_run_case_deps(monkeypatch, tmp_path)

    case_id = "case_anchor_default"
    create_case(server._db, case_id, "ast_abc", "report")
    server._run_case_for_urls(case_id, FAKE_ARTWORK, ["https://example.com/found.png"])

    assert captured_bundles[0]["evidenceAnchor"] is None


def test_run_case_for_urls_sets_evidence_anchor_when_enabled(monkeypatch, tmp_path):
    from db import create_case

    captured_bundles = _stub_common_run_case_deps(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "EVIDENCE_ANCHOR_ENABLED", True)
    monkeypatch.setattr(server, "anchor_evidence_bundle", lambda content, owner_address: {"txHash": "0xTX", "blockNumber": 1, "contentHash": "0xHASH"})

    case_id = "case_anchor_enabled"
    create_case(server._db, case_id, "ast_abc", "report")
    server._run_case_for_urls(case_id, FAKE_ARTWORK, ["https://example.com/found.png"])

    assert captured_bundles[0]["evidenceAnchor"] == {"txHash": "0xTX", "blockNumber": 1, "contentHash": "0xHASH"}


def test_notify_if_evidence_ready_swallows_a_missing_field_instead_of_raising(monkeypatch):
    def raising_notify(*a, **kw):
        raise RuntimeError("api-gateway unreachable")

    monkeypatch.setattr(server, "notify_evidence_ready", raising_notify)
    server._notify_if_evidence_ready(FAKE_ARTWORK, "case_x", "copy")  # must not raise


def test_run_model_leak_case_notifies_the_creator_on_suspected_leak(monkeypatch, tmp_path):
    from db import create_case

    job_result = {"status": "completed", "perPrompt": [], "meanDelta": 0.1, "stdevDelta": 0.0, "verdict": "SUSPECTED_LEAK", "threshold": 0.03}
    _stub_model_leak_deps(monkeypatch, tmp_path, job_result)
    calls = []
    monkeypatch.setattr(server, "notify_evidence_ready", lambda creator_id, title, case_id, evidence_type: calls.append((creator_id, title, case_id, evidence_type)))

    case_id = "case_notify_leak"
    create_case(server._db, case_id, "ast_abc", "model_report")
    server._run_model_leak_case(case_id, FAKE_ARTWORK, "https://example.com/suspect.safetensors")

    assert calls == [("user_creator1", "Starry Fields", case_id, "model_leak")]


# _run_auto_scan_pass: mocks _list_all_artworks and _run_case_for_urls
# directly (that function's own real behavior is already covered by the
# tests above) -- this only tests the due-for-rescan decision itself. Each
# test below uses its own never-before-used artwork id, not FAKE_ARTWORK's
# shared "ast_abc" -- server._db is one shared in-memory connection for the
# whole test file (see the module docstring), and several earlier tests
# already create real "scan"-triggered cases for "ast_abc", which would
# silently make a "never scanned before" assumption false depending on
# test execution order.
def test_auto_scan_pass_scans_an_artwork_that_was_never_scanned_before(monkeypatch):
    artwork = {**FAKE_ARTWORK, "id": "ast_autoscan_never_scanned"}
    monkeypatch.setattr(server, "_list_all_artworks", lambda: [artwork])
    calls = []
    monkeypatch.setattr(server, "_run_case_for_urls", lambda case_id, artwork, urls: calls.append(artwork["id"]))
    monkeypatch.setattr(server, "vision_configured", lambda: False)

    server._run_auto_scan_pass()

    assert calls == ["ast_autoscan_never_scanned"]


def test_auto_scan_pass_skips_an_artwork_scanned_recently(monkeypatch):
    from db import create_case

    artwork = {**FAKE_ARTWORK, "id": "ast_autoscan_recent"}
    create_case(server._db, "case_recent_scan", artwork["id"], "scan")  # created just now

    monkeypatch.setattr(server, "_list_all_artworks", lambda: [artwork])
    calls = []
    monkeypatch.setattr(server, "_run_case_for_urls", lambda case_id, artwork, urls: calls.append(artwork["id"]))

    server._run_auto_scan_pass()

    assert calls == []


def test_auto_scan_pass_rescans_an_artwork_whose_last_scan_is_stale(monkeypatch):
    from db import connect

    # A fresh in-memory db with a deliberately old case row, so this test
    # doesn't depend on AUTO_RESCAN_INTERVAL_SECONDS being small enough to
    # already be stale relative to "just now".
    monkeypatch.setattr(server, "_db", connect(":memory:"))
    from db import create_case, set_case_status

    create_case(server._db, "case_old_scan", "ast_abc", "scan")
    server._db.execute("UPDATE cases SET created_at = 0 WHERE id = 'case_old_scan'")
    server._db.commit()

    monkeypatch.setattr(server, "_list_all_artworks", lambda: [FAKE_ARTWORK])
    calls = []
    monkeypatch.setattr(server, "_run_case_for_urls", lambda case_id, artwork, urls: calls.append(artwork["id"]))
    monkeypatch.setattr(server, "vision_configured", lambda: False)

    server._run_auto_scan_pass()

    assert calls == ["ast_abc"]


def test_auto_scan_pass_skips_artworks_with_no_reachable_protected_image(monkeypatch):
    monkeypatch.setattr(server, "_list_all_artworks", lambda: [{**FAKE_ARTWORK, "protectedImageUri": "/no/such/file.png"}])
    calls = []
    monkeypatch.setattr(server, "_run_case_for_urls", lambda case_id, artwork, urls: calls.append(artwork["id"]))

    server._run_auto_scan_pass()

    assert calls == []


def test_run_case_for_urls_falls_back_to_none_c2pa_when_verify_raises(monkeypatch, tmp_path):
    """verify_c2pa can raise (rust-core binary missing, non-zero exit) --
    real production behavior found live before this was wired in (see
    c2pa_verify.py's module doc). A C2PA verification failure must not
    block or fail the case; it should just show up as c2paDetection: None,
    distinct from a real {"hasManifest": False, ...} result."""
    from db import create_case, get_case

    captured_bundles = _stub_common_run_case_deps(monkeypatch, tmp_path)

    def raising_verify_c2pa(path, fmt):
        raise RuntimeError("rust-core c2pa-verify failed: simulated failure")

    monkeypatch.setattr(server, "verify_c2pa", raising_verify_c2pa)

    case_id = "case_c2pa_wired_failure"
    create_case(server._db, case_id, "ast_abc", "report")

    server._run_case_for_urls(case_id, FAKE_ARTWORK, ["https://example.com/found.png"])

    assert len(captured_bundles) == 1
    assert captured_bundles[0]["c2paDetection"] is None

    case = get_case(server._db, case_id)
    assert case["status"] == "EVIDENCE_READY"
