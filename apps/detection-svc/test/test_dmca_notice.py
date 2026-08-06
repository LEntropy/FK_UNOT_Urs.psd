import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmca_notice import build_dmca_notice, is_dmca_applicable  # noqa: E402

COPY_BUNDLE = {
    "originalHash": "0xaaaa",
    "protectedHash": "0xaaaa",
    "registeredAt": "2026-01-01T00:00:00.000Z",
    "rightsHolder": "0x1234567890abcdef1234567890ABCDEF12345678",
    "watermarkDetection": {
        "recoveredHex": "deadbeef",
        "avgConfidence": 0.97,
        "minConfidence": 0.9,
        "bitErrorRate": 0.02,
        "isMatch": True,
    },
    "c2paDetection": None,
    "modelLeakDetection": None,
    "discoveredUrl": "https://example.com/stolen.png",
    "discoveredAt": 1_700_000_000.0,
    "phashDistance": 4,
    "screenshotPath": None,
    "httpHeaders": None,
    "onchainTransaction": {
        "chain": "polygon-amoy",
        "registryAddress": "0xREGISTRY",
        "txHash": "0xTXHASH",
        "blockNumber": 123,
    },
    "signature": {"signature": "sig", "publicKeyPem": "pem", "algorithm": "ed25519"},
}

MODEL_LEAK_BUNDLE = {**COPY_BUNDLE, "modelLeakDetection": {"verdict": "SUSPECTED_LEAK", "meanDelta": 0.1}}


def test_is_dmca_applicable_true_for_a_copy_bundle():
    assert is_dmca_applicable(COPY_BUNDLE) is True


def test_is_dmca_applicable_false_for_a_model_leak_bundle():
    assert is_dmca_applicable(MODEL_LEAK_BUNDLE) is False


def test_build_dmca_notice_fills_every_known_field():
    notice = build_dmca_notice(COPY_BUNDLE)

    assert "0xaaaa" in notice
    assert "polygon-amoy / 0xREGISTRY" in notice
    assert "0xTXHASH" in notice
    assert "2026-01-01T00:00:00.000Z" in notice
    assert "https://example.com/stolen.png" in notice
    assert "distance: 4 / 256" in notice
    assert "deadbeef" in notice
    assert "97%" in notice
    # Still-manual fields stay as brackets, not silently dropped.
    assert "[host/platform's designated DMCA agent or abuse contact]" in notice
    assert "[rights holder name / contact]" in notice
    assert "Signature: [name]" in notice


def test_build_dmca_notice_falls_back_gracefully_on_missing_fields():
    sparse_bundle = {
        "originalHash": None,
        "watermarkDetection": None,
        "onchainTransaction": None,
        "registeredAt": None,
        "discoveredUrl": None,
        "discoveredAt": None,
        "phashDistance": None,
    }

    notice = build_dmca_notice(sparse_bundle)

    assert "[unavailable]" in notice
    assert "[not registered on-chain]" in notice
    assert "[not measured]" in notice
    assert "[No watermark match recorded for this bundle]" in notice
    assert "[unknown]" in notice
