"""_run_case_for_urls previously always checked candidate images against one
project-wide DEFAULT_WATERMARK_HEX constant, regardless of which artwork a
case was actually about -- asset-service now generates and stores a real
per-artwork watermarkPayloadHex (apps/asset-service/src/routes/artworks.ts),
so this verifies detect_watermark is called with *that*, falling back to
the constant only when an artwork row doesn't have one set.
"""

import os
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("DETECTION_DB_PATH", ":memory:")
os.environ.setdefault("DETECTION_OUT_DIR", "/tmp/detection-svc-test-out")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402

FakeCapture = namedtuple("FakeCapture", ["image_path", "captured_at", "headers", "screenshot_path"])


def _run(monkeypatch, artwork: dict) -> str:
    """Runs _run_case_for_urls against one candidate URL, returns whatever
    watermark_hex detect_watermark was actually called with."""
    calls = {}

    async def fake_capture(url, out_dir):
        return FakeCapture(image_path=__file__, captured_at="2026-01-01T00:00:00Z", headers={}, screenshot_path=None)

    def fake_detect_watermark(image_path, watermark_hex):
        calls["watermark_hex"] = watermark_hex
        return MagicMock(recovered_hex=watermark_hex, avg_confidence=1.0, min_confidence=1.0, bit_error_rate=0.0, is_match=True)

    monkeypatch.setattr(server, "capture", fake_capture)
    monkeypatch.setattr(server, "detect_watermark", fake_detect_watermark)
    monkeypatch.setattr(server, "is_likely_match", lambda *a, **kw: (False, None))
    monkeypatch.setattr(server, "build_bundle", lambda **kw: {})
    monkeypatch.setattr(server, "write_json", lambda *a, **kw: None)
    monkeypatch.setattr(server, "write_pdf_best_effort", lambda *a, **kw: None)
    monkeypatch.setattr(server, "add_evidence", lambda *a, **kw: None)
    monkeypatch.setattr(server, "set_case_status", lambda *a, **kw: None)

    server._run_case_for_urls("case_test", artwork, ["http://example.com/copy.png"])
    return calls["watermark_hex"]


def test_uses_the_artworks_own_watermark_payload_hex(monkeypatch):
    artwork = {"id": "ast_1", "perceptualHash": None, "watermarkPayloadHex": "abc123def456"}
    assert _run(monkeypatch, artwork) == "abc123def456"


def test_falls_back_to_the_default_constant_when_unset(monkeypatch):
    artwork = {"id": "ast_2", "perceptualHash": None}  # no watermarkPayloadHex key at all
    assert _run(monkeypatch, artwork) == server.DEFAULT_WATERMARK_HEX
