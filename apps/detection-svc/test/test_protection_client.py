"""protection_client.py's own unit tests -- mocks httpx directly (same
convention as test_evidence_signing.py), no real network."""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import protection_client  # noqa: E402


class _FakeStreamResponse:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def iter_bytes(self, chunk_size):
        yield from self._chunks


def test_download_suspect_model_writes_the_full_file(monkeypatch, tmp_path):
    chunks = [b"a" * 10, b"b" * 10, b"c" * 5]
    monkeypatch.setattr(httpx, "stream", lambda method, url, **kw: _FakeStreamResponse(chunks))

    dest = tmp_path / "suspect.safetensors"
    protection_client.download_suspect_model("https://example.com/x.safetensors", str(dest), max_bytes=1000)

    assert dest.read_bytes() == b"a" * 10 + b"b" * 10 + b"c" * 5


def test_download_suspect_model_aborts_once_it_exceeds_the_byte_cap(monkeypatch, tmp_path):
    chunks = [b"x" * 500, b"y" * 500, b"z" * 500]  # 1500 bytes total
    monkeypatch.setattr(httpx, "stream", lambda method, url, **kw: _FakeStreamResponse(chunks))

    dest = tmp_path / "suspect.safetensors"
    with pytest.raises(ValueError, match="exceeds"):
        protection_client.download_suspect_model("https://example.com/huge.safetensors", str(dest), max_bytes=1000)


def test_download_suspect_model_surfaces_a_non_200_as_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "stream", lambda method, url, **kw: _FakeStreamResponse([], status_code=404))

    dest = tmp_path / "suspect.safetensors"
    with pytest.raises(httpx.HTTPStatusError):
        protection_client.download_suspect_model("https://example.com/missing.safetensors", str(dest), max_bytes=1000)


class _FakeJsonResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._body


def test_submit_leak_detection_job_posts_the_expected_payload_and_returns_job_id(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeJsonResponse({"jobId": "leakjob_abc123", "status": "queued"})

    monkeypatch.setattr(httpx, "post", fake_post)

    job_id = protection_client.submit_leak_detection_job(
        "http://localhost:8010", "/path/original.png", "/path/suspect.safetensors", ["prompt one", "prompt two"]
    )

    assert job_id == "leakjob_abc123"
    assert captured["url"] == "http://localhost:8010/detect-model-leak"
    assert captured["json"] == {
        "originalImageUri": "/path/original.png",
        "suspectLoraUri": "/path/suspect.safetensors",
        "prompts": ["prompt one", "prompt two"],
    }


def test_poll_leak_detection_job_returns_once_completed(monkeypatch):
    responses = [
        {"status": "processing"},
        {"status": "processing"},
        {"status": "completed", "verdict": "SUSPECTED_LEAK", "meanDelta": 0.1},
    ]

    def fake_get(url, timeout):
        return _FakeJsonResponse(responses.pop(0))

    monkeypatch.setattr(httpx, "get", fake_get)

    result = protection_client.poll_leak_detection_job("http://localhost:8010", "leakjob_abc", timeout_seconds=5, poll_interval_seconds=0)

    assert result == {"status": "completed", "verdict": "SUSPECTED_LEAK", "meanDelta": 0.1}
    assert responses == []  # all three were consumed


def test_poll_leak_detection_job_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _FakeJsonResponse({"status": "processing"}))

    with pytest.raises(protection_client.LeakDetectionTimeoutError):
        protection_client.poll_leak_detection_job("http://localhost:8010", "leakjob_abc", timeout_seconds=0.05, poll_interval_seconds=0.01)
