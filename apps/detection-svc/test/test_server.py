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
