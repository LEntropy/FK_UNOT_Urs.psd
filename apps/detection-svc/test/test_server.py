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
