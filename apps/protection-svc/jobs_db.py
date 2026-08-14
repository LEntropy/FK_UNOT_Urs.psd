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
    status TEXT NOT NULL,          -- queued | processing | completed | failed | cancelled
    request_json TEXT NOT NULL,    -- the original ProtectRequest, for potential retry tooling
    result_json TEXT,              -- set once completed
    error TEXT,
    traceback TEXT,
    queued_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

# 2026-08-14: cancel-upload feature. Added via ALTER TABLE (not folded into
# SCHEMA above) so an existing jobs.db from before this feature doesn't need
# a real migration tool -- CREATE TABLE IF NOT EXISTS above is a no-op on an
# existing table, so these columns would silently never appear without this.
# runpod_job_id/runpod_endpoint_id: set by server.py's on_runpod_job_submitted
# callback (remote_gpu.serverless_dual_arch_cloak) as soon as a RunPod
# Serverless job is actually submitted -- lets request_cancel below reach the
# real in-flight RunPod job, not just this row's own bookkeeping.
_MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN runpod_job_id TEXT",
    "ALTER TABLE jobs ADD COLUMN runpod_endpoint_id TEXT",
]


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    for migration in _MIGRATIONS:
        col = migration.split("ADD COLUMN ")[1].split(" ")[0]
        if col not in existing_cols:
            conn.execute(migration)
    conn.commit()
    return conn


def create_job(conn: sqlite3.Connection, job_id: str, request: dict) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO jobs (job_id, status, request_json, queued_at, updated_at) VALUES (?, 'queued', ?, ?, ?)",
        (job_id, json.dumps(request), now, now),
    )
    conn.commit()


def set_processing(conn: sqlite3.Connection, job_id: str) -> None:
    # Same race guard as set_completed/set_failed below -- a job cancelled
    # while still 'queued' (before its worker thread even started) must not
    # get flipped back to 'processing' when that thread finally runs.
    if _is_cancelled(conn, job_id):
        return
    conn.execute(
        "UPDATE jobs SET status = 'processing', updated_at = ? WHERE job_id = ?", (time.time(), job_id)
    )
    conn.commit()


def set_completed(conn: sqlite3.Connection, job_id: str, result: dict) -> None:
    # Guard against a race with request_cancel below: the background thread
    # running protect() has no way to know it was cancelled mid-flight
    # (there's no checkpoint/interrupt mechanism, see this module's own
    # doc) -- it just keeps running and eventually calls this. Once a job
    # is 'cancelled', a late completion must not resurrect it as
    # 'completed' out from under a caller that already moved on.
    if _is_cancelled(conn, job_id):
        return
    conn.execute(
        "UPDATE jobs SET status = 'completed', result_json = ?, updated_at = ? WHERE job_id = ?",
        (json.dumps(result), time.time(), job_id),
    )
    conn.commit()


def set_failed(conn: sqlite3.Connection, job_id: str, error: str, traceback_str: str | None = None) -> None:
    if _is_cancelled(conn, job_id):
        return
    conn.execute(
        "UPDATE jobs SET status = 'failed', error = ?, traceback = ?, updated_at = ? WHERE job_id = ?",
        (error, traceback_str, time.time(), job_id),
    )
    conn.commit()


def _is_cancelled(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return row is not None and row["status"] == "cancelled"


def set_runpod_job_id(conn: sqlite3.Connection, job_id: str, runpod_job_id: str, runpod_endpoint_id: str) -> None:
    """Called by server.py's on_runpod_job_submitted callback as soon as
    remote_gpu.serverless_dual_arch_cloak actually submits to RunPod --
    well before completion, so request_cancel below has a real target to
    cancel even if called seconds after the job starts."""
    conn.execute(
        "UPDATE jobs SET runpod_job_id = ?, runpod_endpoint_id = ?, updated_at = ? WHERE job_id = ?",
        (runpod_job_id, runpod_endpoint_id, time.time(), job_id),
    )
    conn.commit()


def request_cancel(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Marks a queued/processing job cancelled immediately (server.py's own
    /protect/{job_id}/cancel route). Returns the job row (with its
    runpod_job_id/runpod_endpoint_id, if any, so the caller can also
    best-effort cancel the real RunPod job) if there was something to
    cancel, or None if the job was already terminal (completed/failed/
    cancelled) or doesn't exist -- nothing to do in that case."""
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None or row["status"] not in ("queued", "processing"):
        return None
    conn.execute(
        "UPDATE jobs SET status = 'cancelled', error = ?, updated_at = ? WHERE job_id = ?",
        ("cancelled by user", time.time(), job_id),
    )
    conn.commit()
    return dict(row)


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
