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


@pytest.fixture(autouse=True)
def _multiarch_env(monkeypatch):
    monkeypatch.setenv("MULTIARCH_GPU_HOST", "1.2.3.4")
    monkeypatch.setenv("MULTIARCH_GPU_USER", "root")
    monkeypatch.setenv("MULTIARCH_GPU_PORT", "22222")
    monkeypatch.setenv("MULTIARCH_GPU_SSH_KEY", "~/.ssh/fake_runpod_key_for_tests")


def test_remote_detect_model_leak_uploads_original_and_lora_under_a_fresh_tag_and_cleans_up(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(list(args))
        if args[0] == "ssh" and "model_leak_detect.py" in args[-1]:
            return _FakeCompletedProcess(
                stdout='{"perPrompt": [], "meanDelta": 0.08, "stdevDelta": 0.0, '
                '"verdict": "SUSPECTED_LEAK", "threshold": 0.03}\n'
            )
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake_run(args))

    result = remote_gpu.remote_detect_model_leak(
        "original.png", "suspect.safetensors", ["a painting, oil on canvas"], num_samples=3, resolution=512
    )

    assert result == {
        "perPrompt": [],
        "meanDelta": 0.08,
        "stdevDelta": 0.0,
        "verdict": "SUSPECTED_LEAK",
        "threshold": 0.03,
    }

    scp_calls = [c for c in calls if c[0] == "scp"]
    assert len(scp_calls) == 2  # original image, suspect LoRA
    remote_dests = [c[-1] for c in scp_calls]
    assert all("_leak_" in dest for dest in remote_dests)
    tags = {dest.split("_leak_")[1].split("_")[0] for dest in remote_dests}
    assert len(tags) == 1  # same uuid tag across both uploads

    run_call = next(c for c in calls if c[0] == "ssh" and "model_leak_detect.py" in c[-1])
    assert "--num-samples 3" in run_call[-1]
    assert "--resolution 512" in run_call[-1]
    assert "a painting, oil on canvas" in run_call[-1]

    cleanup_calls = [c for c in calls if c[0] == "ssh" and c[-1].startswith("rm -f")]
    assert len(cleanup_calls) == 1


def test_remote_detect_model_leak_still_cleans_up_after_a_failed_ssh_call(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(list(args))
        if args[0] == "ssh" and "model_leak_detect.py" in args[-1]:
            return _FakeCompletedProcess(returncode=1, stderr="pod unreachable")
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake_run(args))

    with pytest.raises(RuntimeError):
        remote_gpu.remote_detect_model_leak("original.png", "suspect.safetensors", ["a prompt"])

    cleanup_calls = [c for c in calls if c[0] == "ssh" and c[-1].startswith("rm -f")]
    assert len(cleanup_calls) == 1


def test_remote_detect_model_leak_joins_multiple_prompts_with_pipe(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(list(args))
        if args[0] == "ssh" and "model_leak_detect.py" in args[-1]:
            return _FakeCompletedProcess(stdout='{"meanDelta": 0.0, "verdict": "NO_EVIDENCE"}')
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: fake_run(args))

    remote_gpu.remote_detect_model_leak("original.png", "suspect.safetensors", ["prompt one", "prompt two"])

    run_call = next(c for c in calls if c[0] == "ssh" and "model_leak_detect.py" in c[-1])
    assert "--prompts 'prompt one|prompt two'" in run_call[-1]


@pytest.fixture(autouse=True)
def _serverless_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-runpod-key-for-tests")
    monkeypatch.setenv("RUNPOD_STRONGPROTECT_ENDPOINT_ID", "fake-endpoint-id")


class _FakeHttpResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._body


def test_serverless_dual_arch_cloak_submits_then_polls_until_completed(monkeypatch, tmp_path):
    import base64

    import httpx

    original = tmp_path / "original.png"
    original.write_bytes(b"fake image bytes")
    output_path = tmp_path / "output.png"

    posts = []
    gets = []
    statuses = iter(["IN_QUEUE", "IN_PROGRESS", "COMPLETED"])

    def fake_post(url, headers, json, timeout):
        posts.append((url, headers, json))
        return _FakeHttpResponse({"id": "job_abc123", "status": "IN_QUEUE"})

    def fake_get(url, headers, timeout):
        gets.append((url, headers))
        status = next(statuses)
        if status == "COMPLETED":
            output_b64 = base64.b64encode(b"fake protected image bytes").decode()
            return _FakeHttpResponse({"id": "job_abc123", "status": "COMPLETED", "output": {"output_b64": output_b64}})
        return _FakeHttpResponse({"id": "job_abc123", "status": status})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    # poll_interval_seconds=0 -- time.sleep(0) between polls is a real but
    # effectively instant call, no need to mock it out (time is imported
    # locally inside serverless_dual_arch_cloak, matching this module's own
    # convention elsewhere, so there's no module-level remote_gpu.time to
    # monkeypatch anyway).
    remote_gpu.serverless_dual_arch_cloak(str(original), str(output_path), "a painting", poll_interval_seconds=0)

    assert output_path.read_bytes() == b"fake protected image bytes"

    assert len(posts) == 1
    post_url, post_headers, post_body = posts[0]
    assert post_url == "https://api.runpod.ai/v2/fake-endpoint-id/run"
    assert post_headers["Authorization"] == "Bearer fake-runpod-key-for-tests"
    assert post_body["input"]["prompt"] == "a painting"
    assert base64.b64decode(post_body["input"]["image_b64"]) == b"fake image bytes"

    assert len(gets) == 3  # IN_QUEUE, IN_PROGRESS, COMPLETED
    assert all(url == "https://api.runpod.ai/v2/fake-endpoint-id/status/job_abc123" for url, _ in gets)


def test_serverless_dual_arch_cloak_raises_on_a_failed_job(monkeypatch, tmp_path):
    import httpx

    original = tmp_path / "original.png"
    original.write_bytes(b"fake image bytes")

    monkeypatch.setattr(httpx, "post", lambda url, headers, json, timeout: _FakeHttpResponse({"id": "job_x", "status": "IN_QUEUE"}))
    monkeypatch.setattr(
        httpx, "get", lambda url, headers, timeout: _FakeHttpResponse({"id": "job_x", "status": "FAILED", "error": "worker crashed"})
    )

    with pytest.raises(RuntimeError, match="worker crashed"):
        remote_gpu.serverless_dual_arch_cloak(str(original), str(tmp_path / "out.png"), "a painting", poll_interval_seconds=0)


def test_serverless_dual_arch_cloak_raises_on_timeout(monkeypatch, tmp_path):
    import httpx

    original = tmp_path / "original.png"
    original.write_bytes(b"fake image bytes")

    monkeypatch.setattr(httpx, "post", lambda url, headers, json, timeout: _FakeHttpResponse({"id": "job_x", "status": "IN_QUEUE"}))
    monkeypatch.setattr(httpx, "get", lambda url, headers, timeout: _FakeHttpResponse({"id": "job_x", "status": "IN_PROGRESS"}))

    with pytest.raises(RuntimeError, match="did not complete within"):
        remote_gpu.serverless_dual_arch_cloak(
            str(original), str(tmp_path / "out.png"), "a painting", poll_interval_seconds=0, timeout_seconds=0
        )


def test_serverless_score_protection_submits_both_images_then_returns_the_parsed_output(monkeypatch, tmp_path):
    import base64
    import io

    import httpx
    from PIL import Image

    # Real (tiny) PNGs, not placeholder bytes -- serverless_score_protection's
    # own _encode_resized() opens each file with PIL to downscale it before
    # sending (see that function's own comment on the RunPod payload-size
    # limit this was added to avoid), so a non-image placeholder file
    # raises PIL.UnidentifiedImageError before the function ever gets to
    # what this test is actually checking.
    original = tmp_path / "original.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(original)
    protected = tmp_path / "protected.png"
    Image.new("RGB", (8, 8), (40, 50, 60)).save(protected)

    posts = []
    gets = []
    statuses = iter(["IN_QUEUE", "IN_PROGRESS", "COMPLETED"])

    fake_output = {
        "sd15": {"baselineSimilarity": 0.9, "protectedSimilarity": 0.7, "delta": 0.2, "verdict": "PROTECTED",
                  "baselineSamples": [], "protectedSamples": []},
        "sdxl": {"baselineSimilarity": 0.9, "protectedSimilarity": 0.88, "delta": 0.02, "verdict": "NOT_PROTECTED",
                  "baselineSamples": [], "protectedSamples": []},
        "threshold": 0.03,
    }

    def fake_post(url, headers, json, timeout):
        posts.append((url, headers, json))
        return _FakeHttpResponse({"id": "scorejob_abc", "status": "IN_QUEUE"})

    def fake_get(url, headers, timeout):
        gets.append((url, headers))
        status = next(statuses)
        if status == "COMPLETED":
            return _FakeHttpResponse({"id": "scorejob_abc", "status": "COMPLETED", "output": fake_output})
        return _FakeHttpResponse({"id": "scorejob_abc", "status": status})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    result = remote_gpu.serverless_score_protection(
        str(original), str(protected), "a painting", poll_interval_seconds=0
    )

    assert result == fake_output

    assert len(posts) == 1
    post_url, post_headers, post_body = posts[0]
    assert post_url == "https://api.runpod.ai/v2/fake-endpoint-id/run"
    assert post_headers["Authorization"] == "Bearer fake-runpod-key-for-tests"
    assert post_body["input"]["action"] == "score_protection"
    assert post_body["input"]["prompt"] == "a painting"
    # Not an exact-bytes match -- _encode_resized() re-encodes through PIL
    # (open -> optionally downscale -> re-save as PNG), so the bytes sent
    # are never byte-identical to the source file even at the same size.
    # Assert what actually matters: it's valid, openable PNG data of the
    # right (untouched, since 8x8 is well under _encode_resized's 1024px
    # cap) pixel size and color.
    sent_original = Image.open(io.BytesIO(base64.b64decode(post_body["input"]["original_b64"])))
    assert sent_original.size == (8, 8)
    assert sent_original.convert("RGB").getpixel((0, 0)) == (10, 20, 30)
    sent_protected = Image.open(io.BytesIO(base64.b64decode(post_body["input"]["protected_b64"])))
    assert sent_protected.size == (8, 8)
    assert sent_protected.convert("RGB").getpixel((0, 0)) == (40, 50, 60)

    assert len(gets) == 3


def test_serverless_score_protection_raises_on_a_failed_job(monkeypatch, tmp_path):
    import httpx
    from PIL import Image

    # Real PNGs -- see the sibling submit-then-parse test's comment on why
    # a non-image placeholder file breaks _encode_resized()'s PIL open.
    original = tmp_path / "original.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(original)
    protected = tmp_path / "protected.png"
    Image.new("RGB", (8, 8), (40, 50, 60)).save(protected)

    monkeypatch.setattr(httpx, "post", lambda url, headers, json, timeout: _FakeHttpResponse({"id": "job_x", "status": "IN_QUEUE"}))
    monkeypatch.setattr(
        httpx, "get", lambda url, headers, timeout: _FakeHttpResponse({"id": "job_x", "status": "FAILED", "error": "OOM"})
    )

    with pytest.raises(RuntimeError, match="OOM"):
        remote_gpu.serverless_score_protection(
            str(original), str(protected), "a painting", poll_interval_seconds=0
        )
