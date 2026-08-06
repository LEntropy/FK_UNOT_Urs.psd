"""notify_client.py's own unit tests -- mocks httpx directly (same
convention as test_evidence_signing.py/test_protection_client.py), no real
network."""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import notify_client  # noqa: E402


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._body


def test_notify_evidence_ready_posts_the_expected_payload_and_returns_sent(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"sent": True})

    monkeypatch.setattr(httpx, "post", fake_post)

    result = notify_client.notify_evidence_ready("user_1", "Starry Fields", "case_abc", "copy")

    assert result is True
    assert captured["url"] == "http://localhost:4000/internal/notify-evidence-ready"
    assert captured["json"] == {
        "creatorId": "user_1",
        "artworkTitle": "Starry Fields",
        "caseId": "case_abc",
        "evidenceType": "copy",
    }


def test_notify_evidence_ready_returns_false_when_api_gateway_says_not_sent(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: _FakeResponse({"sent": False}))

    result = notify_client.notify_evidence_ready("user_1", "Starry Fields", "case_abc", "model_leak")

    assert result is False


def test_notify_evidence_ready_returns_false_not_raise_on_http_error(monkeypatch):
    def raising_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", raising_post)

    result = notify_client.notify_evidence_ready("user_1", "Starry Fields", "case_abc", "copy")

    assert result is False


def test_notify_evidence_ready_returns_false_on_non_2xx_status(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: _FakeResponse({"error": "boom"}, status_code=500))

    result = notify_client.notify_evidence_ready("user_1", "Starry Fields", "case_abc", "copy")

    assert result is False
