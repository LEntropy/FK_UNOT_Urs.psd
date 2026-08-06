"""Thin HTTP client for protection-svc's model-leak detection job API
(POST/GET /detect-model-leak -- see apps/protection-svc/server.py and
ml-engine/src/model_leak_detect.py's module doc for the mechanism).

New cross-service dependency for detection-svc, unlike everything else
here (README.md: "No DB access to asset-service... Everything needed
comes from GET {ASSET_SERVICE_URL}/artworks/:id") -- model-leak detection
genuinely needs GPU inference (a real StableDiffusionPipeline generation),
which only protection-svc/ml-engine can do; there's no way to keep this
capability without calling out to it. Same peer-service posture as
asset_client.py otherwise (plain HTTP, no shared DB/storage).
"""

import time

import httpx


class LeakDetectionTimeoutError(Exception):
    pass


def download_suspect_model(url: str, dest_path: str, max_bytes: int, timeout: float = 60.0) -> None:
    """Streams `url` to `dest_path`, aborting if it exceeds `max_bytes` --
    a suspect-model URL is caller-submitted (via POST /model-leak-reports),
    so this is the one place in this service that fetches an arbitrary,
    externally-controlled file onto local disk. safetensors itself is a
    safe tensor-only format (header + raw tensor bytes, no pickle/code
    execution -- exactly why the ecosystem moved to it, see huggingface's
    own format docs), so loading it via diffusers' load_lora_weights isn't
    an RCE risk the way an arbitrary pickle would be. The real risk this
    guards against is a caller submitting a URL to a multi-GB file to
    waste GPU-worker disk/bandwidth/inference time -- a coarse but real
    safeguard, not a full anti-abuse system (no rate limiting here either,
    same honestly-documented gap posture as every other endpoint in this
    project).
    """
    written = 0
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"suspect model at {url!r} exceeds the {max_bytes}-byte limit")
                f.write(chunk)


def submit_leak_detection_job(protection_svc_url: str, original_image_uri: str, suspect_lora_uri: str, prompts: list[str]) -> str:
    resp = httpx.post(
        f"{protection_svc_url}/detect-model-leak",
        json={"originalImageUri": original_image_uri, "suspectLoraUri": suspect_lora_uri, "prompts": prompts},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["jobId"]


def poll_leak_detection_job(protection_svc_url: str, job_id: str, timeout_seconds: float, poll_interval_seconds: float = 3.0) -> dict:
    """Blocking poll loop -- called from detection-svc's own background
    Thread (server.py's _run_model_leak_case), same execution model
    _run_case_for_urls already uses for its own blocking capture() calls,
    not an asyncio task."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        resp = httpx.get(f"{protection_svc_url}/detect-model-leak/{job_id}", timeout=10.0)
        resp.raise_for_status()
        job = resp.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(poll_interval_seconds)
    raise LeakDetectionTimeoutError(f"protection-svc job {job_id!r} did not finish within {timeout_seconds}s")
