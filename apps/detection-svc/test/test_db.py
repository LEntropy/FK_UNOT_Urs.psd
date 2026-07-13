"""Case state machine tests (OPEN -> EVIDENCE_READY / NO_MATCH_FOUND / FAILED)
against a real sqlite file (tmp_path), no mocking needed -- this is pure
storage logic, same spirit as asset-service's testDb.ts in-memory approach.
"""

from db import add_evidence, connect, create_case, get_case, set_case_status


def test_case_starts_open(tmp_path):
    db = connect(str(tmp_path / "test.db"))
    create_case(db, "case_1", "ast_abc", "scan")

    case = get_case(db, "case_1")
    assert case["status"] == "OPEN"
    assert case["artwork_id"] == "ast_abc"
    assert case["trigger"] == "scan"
    assert case["evidence"] == []


def test_case_transitions_to_evidence_ready_with_records(tmp_path):
    db = connect(str(tmp_path / "test.db"))
    create_case(db, "case_2", "ast_abc", "report")

    add_evidence(db, "case_2", "phash_match", "https://example.com/found.png", 0.95, "/out/case_2/found")
    set_case_status(db, "case_2", "EVIDENCE_READY")

    case = get_case(db, "case_2")
    assert case["status"] == "EVIDENCE_READY"
    assert len(case["evidence"]) == 1
    assert case["evidence"][0]["evidence_type"] == "phash_match"
    assert case["evidence"][0]["source_url"] == "https://example.com/found.png"


def test_case_transitions_to_failed_with_error_message(tmp_path):
    db = connect(str(tmp_path / "test.db"))
    create_case(db, "case_3", "ast_abc", "scan")

    set_case_status(db, "case_3", "FAILED", "asset-service unreachable")

    case = get_case(db, "case_3")
    assert case["status"] == "FAILED"
    assert case["error_message"] == "asset-service unreachable"


def test_get_case_returns_none_for_unknown_id(tmp_path):
    db = connect(str(tmp_path / "test.db"))
    assert get_case(db, "does_not_exist") is None
