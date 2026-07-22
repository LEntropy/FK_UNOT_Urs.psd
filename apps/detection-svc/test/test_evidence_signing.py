import httpx
import pytest

import evidence_signing


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json_body


def test_sign_bundle_returns_api_gateway_response(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"signature": "abc123", "publicKeyPem": "-----BEGIN PUBLIC KEY-----...", "algorithm": "ed25519"})

    monkeypatch.setattr(httpx, "post", fake_post)

    result = evidence_signing.sign_bundle({"originalHash": "0xaaaa"})

    assert result == {"signature": "abc123", "publicKeyPem": "-----BEGIN PUBLIC KEY-----...", "algorithm": "ed25519"}
    assert captured["url"] == f"{evidence_signing.API_GATEWAY_URL}/internal/sign-evidence"
    assert captured["json"] == {"originalHash": "0xaaaa"}


def test_sign_bundle_returns_none_on_connection_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    assert evidence_signing.sign_bundle({"originalHash": "0xaaaa"}) is None


def test_sign_bundle_returns_none_on_http_error_status(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({"error": "signing failed"}, status_code=500)

    monkeypatch.setattr(httpx, "post", fake_post)

    assert evidence_signing.sign_bundle({"originalHash": "0xaaaa"}) is None
