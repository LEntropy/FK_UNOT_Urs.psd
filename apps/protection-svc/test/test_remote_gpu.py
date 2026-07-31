"""remote_gpu.py delegates the heavy style-cloak/metrics/upscale steps to
the GPU PC over SSH -- these tests mock subprocess.run (no real network,
no real GPU PC needed) to check the command construction and JSON parsing,
the same way c2pa_verify.py's own tests mock subprocess for rust-core.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import remote_gpu  # noqa: E402


@pytest.fixture(autouse=True)
def _gpu_env(monkeypatch):
    monkeypatch.setenv("GPU_HOST", "192.168.0.42")
    monkeypatch.setenv("GPU_USER", "mello")
    monkeypatch.setenv("GPU_REMOTE_DIR", "C:/dontai-ml-engine")
    monkeypatch.setenv("GPU_SSH_KEY", "~/.ssh/fake_key_for_tests")


class _FakeCompletedProcess:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_remote_job_paths_reuses_the_same_deterministic_names_remote_cloak_uploads_under():
    gpu_remote_dir, remote, ssh_opts, scp_opts = remote_gpu._connection()
    original, style, cloaked = remote_gpu._remote_job_paths(gpu_remote_dir, "input.jpg", "target.png")

    assert original == "C:/dontai-ml-engine/out/_remote_job_original.jpg"
    assert style == "C:/dontai-ml-engine/out/_remote_job_style.png"
    assert cloaked == "C:/dontai-ml-engine/out/_remote_job_cloaked.png"


def test_remote_compute_metrics_parses_the_json_stdout(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)
        return _FakeCompletedProcess(
            stdout='{"styleDriftScore": 0.142, "styleSimilarityToOriginal": 0.83, '
            '"perceptualPsnrDb": 32.66, "perceptualRmse": 0.02}\n'
        )

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake_run(args))

    result = remote_gpu.remote_compute_metrics("input.jpg", "target.png", size=512)

    assert result == {
        "styleDriftScore": 0.142,
        "styleSimilarityToOriginal": 0.83,
        "perceptualPsnrDb": 32.66,
        "perceptualRmse": 0.02,
    }
    # The one and only remote call should be a single ssh invocation (no
    # scp needed -- remote_cloak() already uploaded/produced these files),
    # running evaluate.py against the same deterministic remote paths in
    # --json mode.
    assert len(calls) == 1
    ssh_call = calls[0]
    assert ssh_call[0] == "ssh"
    remote_cmd = ssh_call[-1]
    assert "src/evaluate.py" in remote_cmd
    assert "_remote_job_original.jpg" in remote_cmd
    assert "_remote_job_style.png" in remote_cmd
    assert "_remote_job_cloaked.png" in remote_cmd
    assert "--size 512" in remote_cmd
    assert "--json" in remote_cmd


def test_remote_compute_metrics_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kw: _FakeCompletedProcess(returncode=1, stderr="GPU PC unreachable")
    )

    with pytest.raises(RuntimeError):
        remote_gpu.remote_compute_metrics("input.jpg", "target.png")
