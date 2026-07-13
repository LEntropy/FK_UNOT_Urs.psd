from evidence_bundle import build_bundle, write_json


FAKE_ARTWORK = {
    "id": "ast_abc",
    "perceptualHash": "0xaaaa",
    "ownerWalletAddress": "0x1234567890abcdef1234567890ABCDEF12345678",
    "ownershipRecords": [
        {
            "chain": "polygon-amoy",
            "registryAddress": "0xregistry",
            "txHash": "0xtxhash",
            "blockNumber": 42,
            "registeredAt": 1_700_000_000.0,
        }
    ],
}


def test_build_bundle_has_project_design_fields():
    bundle = build_bundle(
        artwork=FAKE_ARTWORK,
        source_url="https://example.com/found.png",
        detected_at=1_700_000_100.0,
        phash_distance=3,
        watermark_result={"isMatch": True, "bitErrorRate": 0.0},
        headers={"content-type": "image/png"},
        screenshot_path="/out/case_1/screenshot.png",
    )

    # PROJECT_DESIGN.md §3-7's exact field list, English key names.
    assert bundle["originalHash"] == "0xaaaa"
    assert bundle["protectedHash"] == "0xaaaa"
    assert bundle["registeredAt"] == 1_700_000_000.0
    assert bundle["rightsHolder"] == FAKE_ARTWORK["ownerWalletAddress"]
    assert bundle["watermarkDetection"]["isMatch"] is True
    assert bundle["discoveredUrl"] == "https://example.com/found.png"
    assert bundle["discoveredAt"] == 1_700_000_100.0
    assert bundle["phashDistance"] == 3
    assert bundle["screenshotPath"] == "/out/case_1/screenshot.png"
    assert bundle["httpHeaders"] == {"content-type": "image/png"}
    assert bundle["onchainTransaction"]["txHash"] == "0xtxhash"
    assert bundle["onchainTransaction"]["blockNumber"] == 42
    # Signing is explicitly not implemented yet -- see module docstring.
    assert bundle["signature"] is None


def test_build_bundle_handles_no_onchain_record():
    artwork = {**FAKE_ARTWORK, "ownershipRecords": []}
    bundle = build_bundle(
        artwork=artwork,
        source_url=None,
        detected_at=1_700_000_100.0,
        phash_distance=None,
        watermark_result=None,
        headers=None,
        screenshot_path=None,
    )
    assert bundle["onchainTransaction"] is None
    assert bundle["registeredAt"] is None


def test_write_json_roundtrip(tmp_path):
    bundle = build_bundle(
        artwork=FAKE_ARTWORK,
        source_url="https://example.com/found.png",
        detected_at=1_700_000_100.0,
        phash_distance=3,
        watermark_result=None,
        headers=None,
        screenshot_path=None,
    )
    out_path = tmp_path / "bundle.json"
    write_json(bundle, out_path)

    import json

    loaded = json.loads(out_path.read_text())
    assert loaded["discoveredUrl"] == "https://example.com/found.png"
