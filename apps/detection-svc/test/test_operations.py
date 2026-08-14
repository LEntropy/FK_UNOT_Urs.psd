import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db import connect, consume_vision_quota, filter_new_candidates, get_monitoring_states, get_vision_usage, update_monitoring_state
from evidence_integrity import EvidenceSigner, verify_evidence_directory, verify_manifest_signature
from evidence_store import LocalWormEvidenceStore
from monitor_scheduler import MonitorScheduler
from phash_index import PHashIndex


def _hash(number: int) -> str:
    return "0x" + f"{number:064x}"


def test_phash_index_search_and_persistence(tmp_path):
    path = tmp_path / "index.json"
    index = PHashIndex(path)
    index.upsert("art-a", _hash(0))
    index.upsert("art-b", _hash(3))
    index.upsert("far", _hash((1 << 200) - 1))
    assert [item["artworkId"] for item in index.search(_hash(1), 2)] == ["art-a", "art-b"]
    restored = PHashIndex(path)
    assert len(restored) == 3
    assert restored.search(_hash(0), 0)[0]["artworkId"] == "art-a"


def test_candidate_dedup_and_monitoring_state(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    assert filter_new_candidates(conn, "art-1", ["https://a/x", "https://a/x", "https://b/y"]) == [
        "https://a/x", "https://b/y"
    ]
    assert filter_new_candidates(conn, "art-1", ["https://a/x"]) == []
    assert filter_new_candidates(conn, "art-2", ["https://a/x"]) == ["https://a/x"]
    update_monitoring_state(conn, "art-1", status="QUEUED", next_scan_at=123.0)
    assert get_monitoring_states(conn)[0]["last_status"] == "QUEUED"


def test_vision_monthly_hard_limit_is_persistent(tmp_path):
    path = str(tmp_path / "db.sqlite")
    conn = connect(path)
    assert consume_vision_quota(conn, 2) is True
    assert consume_vision_quota(conn, 2) is True
    assert consume_vision_quota(conn, 2) is False
    assert get_vision_usage(conn, 2)["remaining"] == 0
    conn.close()
    reopened = connect(path)
    assert get_vision_usage(reopened, 2)["used"] == 2


def test_manifest_signature_detects_tampering(tmp_path):
    evidence = tmp_path / "candidate.bin"
    evidence.write_bytes(b"evidence")
    import hashlib
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"algorithm": "SHA-256", "signatureStatus": "UNSIGNED", "files": [
        {"path": "candidate.bin", "sha256": hashlib.sha256(b"evidence").hexdigest(), "size": 8}
    ]}), encoding="utf-8")
    signer = EvidenceSigner.generate(tmp_path / "signing.pem", "test-key")
    signer.sign_manifest(manifest)
    assert verify_manifest_signature(manifest) == {"valid": True, "status": "VALID", "keyId": "test-key"}
    assert verify_evidence_directory(tmp_path)["valid"] is True
    evidence.write_bytes(b"tampered")
    assert verify_evidence_directory(tmp_path)["valid"] is False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].append({"path": "added", "sha256": "0" * 64, "size": 1})
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_manifest_signature(manifest)["status"] == "INVALID"


def test_scheduler_tracks_success_and_failure():
    calls = []
    scheduler = MonitorScheduler(lambda: calls.append("ok"), 60)
    scheduler.run_once()
    assert calls == ["ok"]
    assert scheduler.status()["lastCompletedAt"] is not None

    failing = MonitorScheduler(lambda: (_ for _ in ()).throw(RuntimeError("offline")), 60)
    with pytest.raises(RuntimeError):
        failing.run_once()
    assert failing.status()["lastError"] == "offline"


def test_local_worm_refuses_overwrite_and_detects_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "bundle.json").write_text('{"ok":true}', encoding="utf-8")
    store = LocalWormEvidenceStore(tmp_path / "worm", retention_days=30)
    stored = store.put(source, case_id="case-1", evidence_id="candidate-1")
    assert stored.object_id == "case-1/candidate-1"
    assert store.verify(stored.object_id)["valid"] is True
    with pytest.raises(FileExistsError):
        store.put(source, case_id="case-1", evidence_id="candidate-1")

    copied = Path(stored.uri) / "bundle.json"
    copied.chmod(0o644)
    copied.write_text('{"ok":false}', encoding="utf-8")
    assert store.verify(stored.object_id)["status"] == "INVALID"


def test_local_worm_records_retention_and_audit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.bin").write_bytes(b"evidence")
    store = LocalWormEvidenceStore(tmp_path / "worm", retention_days=365)
    stored = store.put(source, case_id="case-2", evidence_id="candidate-2")
    manifest = json.loads((Path(stored.uri) / "worm-manifest.json").read_text(encoding="utf-8"))
    assert manifest["retentionMode"] == "COMPLIANCE_EMULATION"
    assert manifest["retainedUntil"] == stored.retained_until
    audit = (tmp_path / "worm" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(audit[0])["action"] == "PUT"
