import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_bundle import write_evidence_manifest
from evidence_capture import _ImageFinder, _auth_headers, _looks_like_login, _validate_public_url
from phash_match import compute_perceptual_hash, hamming_distance
from robust_fingerprint import compare_robust_fingerprint
from verdict import decide_verdict


def _pattern() -> Image.Image:
    y, x = np.indices((96, 128))
    pixels = np.stack(((x * 3) % 255, (y * 5) % 255, ((x + y) * 7) % 255), axis=2).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def test_phash_is_256_bit_and_stable_for_same_image():
    image = _pattern()
    value = compute_perceptual_hash(image)
    assert value.startswith("0x")
    assert len(value) == 66
    assert hamming_distance(value, compute_perceptual_hash(image.copy())) == 0


def test_robust_fingerprint_recognizes_mirrored_candidate(tmp_path):
    original = _pattern()
    registered = compute_perceptual_hash(original)
    candidate = tmp_path / "mirrored.png"
    ImageOps.mirror(original).save(candidate)
    result = compare_robust_fingerprint(registered, candidate)
    assert result.matched is True
    assert result.transform == "mirror"
    assert result.distance == 0


def test_verdict_explains_match_and_inconclusive():
    match = decide_verdict(phash_distance=3, watermark_match=True, fingerprint_distance=3, c2pa_status="VALID")
    assert match["verdict"] == "MATCH"
    assert match["confidence"] == 1.0
    assert match["reasons"]
    unknown = decide_verdict(phash_distance=None, watermark_match=None, fingerprint_distance=None, c2pa_status="UNAVAILABLE")
    assert unknown["verdict"] == "INCONCLUSIVE"


def test_manifest_hashes_artifacts_but_not_itself(tmp_path):
    (tmp_path / "candidate.png").write_bytes(b"candidate")
    (tmp_path / "headers.json").write_text("{}", encoding="utf-8")
    manifest_path = write_evidence_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["signatureStatus"] == "UNSIGNED"
    assert {item["path"] for item in manifest["files"]} == {"candidate.png", "headers.json"}
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_private_evidence_targets_are_blocked():
    import pytest

    with pytest.raises(ValueError):
        _validate_public_url("http://127.0.0.1/private.png")


def test_html_image_finder_prioritizes_open_graph_image():
    finder = _ImageFinder()
    finder.feed('<img src="first.png"><meta property="og:image" content="canonical.png">')
    assert finder.urls == ["canonical.png", "first.png"]


def test_bearer_token_is_only_sent_to_allowlisted_host():
    allowed = {"images.example.com"}
    assert _auth_headers("https://images.example.com/a.png", "secret", allowed) == {
        "Authorization": "Bearer secret"
    }
    assert _auth_headers("https://redirected.example.net/a.png", "secret", allowed) == {}


def test_login_page_detection_uses_redirect_url_or_password_form():
    assert _looks_like_login("https://example.com/login") is True
    assert _looks_like_login("https://example.com/page", '<input type="password">') is True
    assert _looks_like_login("https://example.com/public", "<img src='x.png'>") is False
