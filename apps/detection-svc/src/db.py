"""SQLite persistence for detection-svc, matching PROJECT_DESIGN.md's §4
`evidence_records` table plus a `cases` table for the §7 runbook state
machine (OPEN -> EVIDENCE_READY | FAILED; steps 4-6 of the runbook --
notification, takedown, tracking -- are product/human work, out of scope
for this service).

Uses plain stdlib sqlite3, not an ORM: two small tables, no relations
beyond a foreign key, no query complexity that would justify one (contrast
with asset-service's Drizzle usage, which manages a bigger schema).

Evidence, unlike protection-svc's in-memory job dict, must survive a
restart -- losing a legal evidence record is a worse failure mode than
losing a protect-job's in-flight status, so this is real persistent
storage from the start, not a documented "known gap".
"""

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    artwork_id TEXT NOT NULL,
    status TEXT NOT NULL,          -- OPEN | EVIDENCE_READY | NO_MATCH_FOUND | FAILED
    trigger TEXT NOT NULL,         -- scan | report
    error_message TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(id),
    evidence_type TEXT NOT NULL,   -- reverse_image | phash_match | watermark_match
    source_url TEXT,
    confidence REAL,
    artifact_uri TEXT,
    detected_at REAL NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_case(conn: sqlite3.Connection, case_id: str, artwork_id: str, trigger: str) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO cases (id, artwork_id, status, trigger, created_at, updated_at) VALUES (?, ?, 'OPEN', ?, ?, ?)",
        (case_id, artwork_id, trigger, now, now),
    )
    conn.commit()


def set_case_status(conn: sqlite3.Connection, case_id: str, status: str, error_message: str | None = None) -> None:
    conn.execute(
        "UPDATE cases SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
        (status, error_message, time.time(), case_id),
    )
    conn.commit()


def add_evidence(
    conn: sqlite3.Connection,
    case_id: str,
    evidence_type: str,
    source_url: str | None,
    confidence: float | None,
    artifact_uri: str | None,
) -> None:
    conn.execute(
        "INSERT INTO evidence_records (case_id, evidence_type, source_url, confidence, artifact_uri, detected_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (case_id, evidence_type, source_url, confidence, artifact_uri, time.time()),
    )
    conn.commit()


def get_case(conn: sqlite3.Connection, case_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    case = dict(row)
    evidence = conn.execute(
        "SELECT * FROM evidence_records WHERE case_id = ? ORDER BY detected_at", (case_id,)
    ).fetchall()
    case["evidence"] = [dict(e) for e in evidence]
    return case
