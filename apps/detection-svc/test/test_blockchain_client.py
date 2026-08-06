"""blockchain_client.py's own unit tests -- mocks httpx directly (same
convention as test_protection_client.py/test_notify_client.py), no real
network, no real chain call."""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import blockchain_client  # noqa: E402


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._body


def test_anchor_evidence_bundle_posts_the_expected_payload_and_returns_the_receipt(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"contentHash": "0xHASH", "txHash": "0xTX", "blockNumber": 42})

    monkeypatch.setattr(httpx, "post", fake_post)

    result = blockchain_client.anchor_evidence_bundle('{"a": 1}', "0x1234567890abcdef1234567890ABCDEF12345678")

    assert result == {"txHash": "0xTX", "blockNumber": 42, "contentHash": "0xHASH"}
    assert captured["url"] == "http://localhost:3001/assets/register"
    assert captured["json"] == {
        "ownerAddress": "0x1234567890abcdef1234567890ABCDEF12345678",
        "doNotTrain": False,
        "content": '{"a": 1}',
    }


def test_anchor_evidence_bundle_returns_none_on_http_error(monkeypatch):
    def raising_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", raising_post)

    result = blockchain_client.anchor_evidence_bundle("{}", "0x1234567890abcdef1234567890ABCDEF12345678")

    assert result is None


def test_anchor_evidence_bundle_returns_none_on_non_2xx_status(monkeypatch):
    monkeypatch.setattr(
        httpx, "post", lambda url, json, timeout: _FakeResponse({"error": "insufficient funds"}, status_code=502)
    )

    result = blockchain_client.anchor_evidence_bundle("{}", "0x1234567890abcdef1234567890ABCDEF12345678")

    assert result is None


def test_anchor_evidence_bundle_returns_none_on_an_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: _FakeResponse({"unexpected": "shape"}))

    result = blockchain_client.anchor_evidence_bundle("{}", "0x1234567890abcdef1234567890ABCDEF12345678")

    assert result is None
