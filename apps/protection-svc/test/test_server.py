"""Tests server.py's HTTP contract (INTEGRATION.md's job-based API) against
a mocked orchestrate.protect() -- no torch/GPU actually exercised here,
that's ml-engine's own (manual, GPU-dependent) experiment scripts' job.
"""

import os
import sys
import time
from pathlib import Path

import pytest

# jobs_db.connect() (and the resulting mark_interrupted_jobs_failed() call)
# happens at server.py's *module import* time -- must point JOBS_DB_PATH at
# a throwaway file before that import, same reasoning as blockchain-svc's
# TS tests setting process.env before a dynamic import.
_TEST_JOBS_DB = str(Path(__file__).parent / "test_jobs.db")
if os.path.exists(_TEST_JOBS_DB):
    os.remove(_TEST_JOBS_DB)
os.environ["JOBS_DB_PATH"] = _TEST_JOBS_DB

sys.path.insert(0, str(Path(__file__).parent.parent))
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    server._jobs_conn.execute("DELETE FROM jobs")
    server._jobs_conn.commit()
    return TestClient(server.app)


@pytest.fixture
def real_image(tmp_path):
    # A real, small, valid PNG -- not just a file that exists. _run_job now
    # calls choose_processing_size() (a real PIL.Image.open()) before ever
    # reaching the mocked-out `protect()` below, so this needs to be a real
    # decodable image, not a placeholder that only satisfies the route
    # handler's Path.exists() check.
    from PIL import Image

    p = tmp_path / "source.png"
    Image.new("RGB", (64, 48), (200, 50, 50)).save(p)
    return str(p)


def wait_for_terminal_status(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/protect/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} never reached a terminal status")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_job_rejects_unknown_preset(client, real_image):
    res = client.post("/protect", json={"imageUri": real_image, "protectionProfile": "NOT_A_REAL_PRESET"})
    assert res.status_code == 400
    assert "NOT_A_REAL_PRESET" in res.json()["detail"]


def test_create_job_rejects_missing_image(client):
    res = client.post("/protect", json={"imageUri": "C:/definitely/not/a/real/path.png"})
    assert res.status_code == 400
    assert "not found" in res.json()["detail"]


def test_get_unknown_job_404s(client):
    res = client.get("/protect/job_doesnotexist")
    assert res.status_code == 404


def test_happy_path_reaches_completed(client, real_image, monkeypatch):
    def fake_protect(**kwargs):
        return {
            "status": "completed",
            "protectedImageUri": "out/job_x/watermarked.png",
            "perceptualHash": "0xaaaa",
            "metadataHash": "0xbbbb",
            "appliedPreset": kwargs["preset_name"],
            "eotUsed": False,
            "variants": [],
        }

    monkeypatch.setattr(server, "protect", fake_protect)

    create = client.post("/protect", json={"imageUri": real_image, "protectionProfile": "L1_PREVIEW"})
    assert create.status_code == 202
    job_id = create.json()["jobId"]
    assert create.json()["status"] == "queued"

    final = wait_for_terminal_status(client, job_id)
    assert final["status"] == "completed"
    assert final["perceptualHash"] == "0xaaaa"
    assert final["appliedPreset"] == "L1_PREVIEW"


def test_protect_failure_is_reported_as_failed_status_not_a_500(client, real_image, monkeypatch):
    def fake_protect(**kwargs):
        raise RuntimeError("GPU out of memory")

    monkeypatch.setattr(server, "protect", fake_protect)

    create = client.post("/protect", json={"imageUri": real_image})
    job_id = create.json()["jobId"]

    final = wait_for_terminal_status(client, job_id)
    assert final["status"] == "failed"
    assert "GPU out of memory" in final["error"]
    assert "traceback" in final


def test_style_target_defaults_when_not_provided(client, real_image, monkeypatch):
    captured = {}

    def fake_protect(**kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(server, "protect", fake_protect)

    create = client.post("/protect", json={"imageUri": real_image})
    wait_for_terminal_status(client, create.json()["jobId"])

    assert captured["style_target_path"] == str(server.ML_ENGINE_DIR / "out" / "style_target.png")


def test_interrupted_jobs_are_marked_failed_on_next_startup(client, real_image, monkeypatch):
    """The actual resumability fix (task: job state survives a restart):
    a job left 'processing' (as if the process died mid-job, before ever
    reaching a terminal status) gets marked failed, with an honest
    "interrupted by restart" message, the next time mark_interrupted_jobs_failed
    runs -- rather than a GET on it hanging in "processing" forever with no
    way to tell "still running" from "died silently"."""
    import jobs_db

    def hangs_forever(**kwargs):
        raise AssertionError("should never actually run in this test")

    monkeypatch.setattr(server, "protect", hangs_forever)

    # Simulate a job that was queued and started processing, then the
    # process died before _run_job ever updated it again -- bypass the real
    # POST /protect + executor path entirely to construct exactly that state.
    jobs_db.create_job(server._jobs_conn, "job_interrupted", {"imageUri": real_image})
    jobs_db.set_processing(server._jobs_conn, "job_interrupted")

    marked = jobs_db.mark_interrupted_jobs_failed(server._jobs_conn)
    assert marked == 1

    res = client.get("/protect/job_interrupted")
    assert res.json()["status"] == "failed"
    assert "interrupted" in res.json()["error"]


def test_a_completed_jobs_result_survives_being_marked_interrupted_again(client, real_image, monkeypatch):
    """mark_interrupted_jobs_failed only touches queued/processing rows --
    a job that already finished (successfully or not) before the restart
    must not be silently overwritten just because the check runs again."""
    import jobs_db

    def fake_protect(**kwargs):
        return {"status": "completed", "perceptualHash": "0xkeep-me"}

    monkeypatch.setattr(server, "protect", fake_protect)

    create = client.post("/protect", json={"imageUri": real_image})
    final = wait_for_terminal_status(client, create.json()["jobId"])
    assert final["status"] == "completed"

    jobs_db.mark_interrupted_jobs_failed(server._jobs_conn)

    still_there = client.get(f"/protect/{create.json()['jobId']}").json()
    assert still_there["status"] == "completed"
    assert still_there["perceptualHash"] == "0xkeep-me"


def test_style_target_uses_request_override(client, real_image, monkeypatch):
    captured = {}

    def fake_protect(**kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(server, "protect", fake_protect)

    create = client.post(
        "/protect", json={"imageUri": real_image, "styleTargetUri": "/custom/target.png"}
    )
    wait_for_terminal_status(client, create.json()["jobId"])

    assert captured["style_target_path"] == "/custom/target.png"
