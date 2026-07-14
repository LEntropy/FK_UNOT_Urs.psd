"""Tests server.py's HTTP contract (INTEGRATION.md's job-based API) against
a mocked orchestrate.protect() -- no torch/GPU actually exercised here,
that's ml-engine's own (manual, GPU-dependent) experiment scripts' job.
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
import server  # noqa: E402


@pytest.fixture
def client():
    server._jobs.clear()
    return TestClient(server.app)


@pytest.fixture
def real_image(tmp_path):
    p = tmp_path / "source.png"
    p.write_bytes(b"not a real png, just needs to exist for the Path.exists() check")
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
