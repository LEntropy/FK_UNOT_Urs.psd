"""SQLite persistence for protection-svc's protect jobs, replacing the
in-memory dict server.py used to rely on entirely (documented there as a
known gap: "Job state is an in-memory dict -- lost on restart").

Mirrors detection-svc's src/db.py pattern (plain stdlib sqlite3, no ORM --
one small table, no query complexity that would justify one).

Persistence alone doesn't make an interrupted GPU job resumable -- a
protect() call killed mid-optimization can't safely pick back up from an
arbitrary point, and there's no checkpoint mechanism in ml-engine/rust-core
to resume from. What this actually buys: a job's *final* status/result
survives a restart (so GET /protect/{jobId} still works after a restart
for anything that finished before the crash), and any job genuinely
interrupted mid-flight gets marked FAILED with an honest "interrupted by
restart" message on the next startup (server.py's mark_interrupted_jobs_failed)
instead of sitting in "queued"/"processing" forever with no way to
distinguish "still running" from "died silently".
"""

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,          -- queued | processing | completed | failed
    request_json TEXT NOT NULL,    -- the original ProtectRequest, for potential retry tooling
    result_json TEXT,              -- set once completed
    error TEXT,
    traceback TEXT,
    queued_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_job(conn: sqlite3.Connection, job_id: str, request: dict) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO jobs (job_id, status, request_json, queued_at, updated_at) VALUES (?, 'queued', ?, ?, ?)",
        (job_id, json.dumps(request), now, now),
    )
    conn.commit()


def set_processing(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'processing', updated_at = ? WHERE job_id = ?", (time.time(), job_id)
    )
    conn.commit()


def set_completed(conn: sqlite3.Connection, job_id: str, result: dict) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'completed', result_json = ?, updated_at = ? WHERE job_id = ?",
        (json.dumps(result), time.time(), job_id),
    )
    conn.commit()


def set_failed(conn: sqlite3.Connection, job_id: str, error: str, traceback_str: str | None = None) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'failed', error = ?, traceback = ?, updated_at = ? WHERE job_id = ?",
        (error, traceback_str, time.time(), job_id),
    )
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return _row_to_job_dict(row)


def mark_interrupted_jobs_failed(conn: sqlite3.Connection) -> int:
    """Called once at startup (server.py). Any job still 'queued' or
    'processing' from a previous process's lifetime was, by definition,
    interrupted -- this process just started, so nothing it hasn't been
    told about via a fresh POST /protect can legitimately be mid-flight
    yet. Returns the number of jobs marked, for a real startup log line
    instead of a silent no-op."""
    now = time.time()
    cursor = conn.execute(
        "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE status IN ('queued', 'processing')",
        ("interrupted by a protection-svc process restart -- no checkpoint/resume exists for a partial GPU job", now),
    )
    conn.commit()
    return cursor.rowcount


def _row_to_job_dict(row: sqlite3.Row) -> dict:
    # Reconstructs the same shape GET /protect/{jobId} always returned
    # (jobId/status/result fields flattened together), not a raw DB row --
    # callers (asset-service's pollProtectJob) shouldn't need to know this
    # is now backed by SQLite instead of an in-memory dict.
    job = {"jobId": row["job_id"], "status": row["status"]}
    if row["result_json"]:
        job.update(json.loads(row["result_json"]))
        job["status"] = row["status"]  # result_json's own "status": "completed" would otherwise win the update
    if row["error"] is not None:
        job["error"] = row["error"]
    if row["traceback"] is not None:
        job["traceback"] = row["traceback"]
    job["queuedAt"] = row["queued_at"]
    return job
