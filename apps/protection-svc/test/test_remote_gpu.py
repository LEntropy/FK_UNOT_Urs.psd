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


def test_remote_measure_existing_images_uploads_under_a_fresh_unique_tag_and_cleans_up(monkeypatch):
    """Unlike remote_compute_metrics() (reuses remote_cloak's fixed shared
    filenames, only safe right after that same job), this is called
    on-demand, arbitrarily long after the original job -- must not reuse
    those same fixed names, or a concurrent real upload's remote_cloak()
    could race it (or it could read back stale/wrong data)."""
    calls = []

    def fake_run(args):
        calls.append(list(args))
        if args[0] == "ssh" and "evaluate.py" in args[-1]:
            return _FakeCompletedProcess(stdout='{"styleDriftScore": 0.05}')
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake_run(args))

    result = remote_gpu.remote_measure_existing_images("orig.png", "cloaked.png", "target.png", size=256)

    assert result == {"styleDriftScore": 0.05}

    scp_calls = [c for c in calls if c[0] == "scp"]
    assert len(scp_calls) == 3  # original, cloaked, style target -- all freshly uploaded, none reused

    # All three uploads (and the ssh eval command) must share the exact same
    # uuid tag, and that tag must not be remote_cloak()'s own fixed
    # "_remote_job_..." names.
    remote_dests = [c[-1] for c in scp_calls]
    assert all("_retest_" in dest for dest in remote_dests)
    tags = {dest.split("_retest_")[1].split("_")[0] for dest in remote_dests}
    assert len(tags) == 1  # same tag across all three uploads
    assert not any("_remote_job_" in dest for dest in remote_dests)

    ssh_calls = [c for c in calls if c[0] == "ssh"]
    assert len(ssh_calls) == 2  # the evaluate.py run, plus the best-effort cleanup
    assert "evaluate.py" in ssh_calls[0][-1]
    assert "Remove-Item" in ssh_calls[1][-1]


def test_remote_measure_existing_images_still_cleans_up_after_a_failed_ssh_call(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(list(args))
        if args[0] == "ssh" and "evaluate.py" in args[-1]:
            return _FakeCompletedProcess(returncode=1, stderr="GPU PC unreachable")
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake_run(args))

    with pytest.raises(RuntimeError):
        remote_gpu.remote_measure_existing_images("orig.png", "cloaked.png", "target.png")

    cleanup_calls = [c for c in calls if c[0] == "ssh" and "Remove-Item" in c[-1]]
    assert len(cleanup_calls) == 1
