"""Delegates the heavy style_cloak step to the GPU PC over SSH, instead of
running it locally. Written for running protection-svc's orchestrator on a
Raspberry Pi (ARM, CPU-only, disk-constrained) -- the Pi has no GPU and
cloak jobs there would be impractically slow (worse than this project's own
CPU baseline). The GPU PC (`apps/protection-svc/ml-engine/remote/README.md`)
already has a working CUDA venv and the latest ml-engine code from earlier
in this project, so this reuses that instead of duplicating a GPU setup on
the Pi.

Requires an SSH keypair from the machine running this (the Pi, in
practice) directly to the GPU PC -- see SETUP_GPU_PC.md's authorized_keys
notes (same admin-vs-regular-account gotcha applies) for how that key gets
registered. Reads GPU_HOST/GPU_USER/GPU_REMOTE_DIR/SSH_KEY from environment
variables (mirrors ml-engine/remote/remote.env's shape) rather than
hardcoding the GPU PC's LAN address here.
"""

import os
import subprocess
import threading
import uuid
from typing import Callable


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"remote_gpu.py: required env var {name} is not set")
    return value


# 2026-08-14: local (this-process) concurrency gate per RunPod Serverless
# endpoint, added on top of protection-svc/server.py's own ThreadPoolExecutor
# caps. Why that alone isn't enough: server.py's _executor (max_workers=1)
# and _cloud_executor (max_workers=4) both fan into the SAME two RunPod
# endpoints below -- e.g. a /protect job (strongProtection), a /score-
# protection job, and a /lora-jobs job can all be in flight at once, all
# three calling functions in *this* module that submit to
# RUNPOD_STRONGPROTECT_ENDPOINT_ID, which itself is only configured for
# workers.max=2 (see the dontai-strongprotect endpoint's own RunPod config).
# Without a gate here, this process could have 5 concurrent submit+poll
# loops racing for 2 real GPU workers -- RunPod's own QUEUE_DELAY
# autoscaler doesn't reject the extra requests, it just queues them
# endpoint-side, so nothing crashes, but a job stuck behind 2 others for
# the endpoint's actual worker capacity can burn most or all of its own
# client-side timeout_seconds just waiting for a worker, and report a
# false "did not complete within Ns" failure for a job that would have
# succeeded given more time -- exactly the failure mode
# serverless_dual_arch_cloak's own timeout_seconds bump (2026-08-13) was
# reacting to from a different cause (server-side executionTimeoutMs).
# Bounding concurrency to each endpoint's real workers.max here means
# extra requests wait in this Python process (still shown as "processing"
# in jobs_db, which is honest -- they genuinely are queued) instead of
# piling an unbounded number of simultaneous jobs onto 2 physical workers.
_ENDPOINT_GATES: dict[str, threading.Semaphore] = {}
_ENDPOINT_GATES_LOCK = threading.Lock()


def _endpoint_gate(endpoint_id: str, max_concurrent_env: str, default_max_concurrent: int = 2) -> threading.Semaphore:
    """Returns the shared semaphore for one RunPod endpoint, sized from
    max_concurrent_env (falls back to default_max_concurrent, which
    matches both the dontai-strongprotect and dontai-stylecloak endpoints'
    current workers.max=2 -- see RUNPOD_STRONGPROTECT_MAX_CONCURRENT /
    RUNPOD_STYLECLOAK_MAX_CONCURRENT below). Keyed by endpoint_id, not the
    env var name, so this still works correctly if RunPod's own workers.max
    ever changes without a matching env var edit landing here at the same
    time -- worst case the gate is looser or tighter than the real cap by
    a bit, never silently pointed at the wrong endpoint.
    """
    with _ENDPOINT_GATES_LOCK:
        gate = _ENDPOINT_GATES.get(endpoint_id)
        if gate is None:
            limit = max(1, int(os.environ.get(max_concurrent_env, str(default_max_concurrent))))
            gate = threading.Semaphore(limit)
            _ENDPOINT_GATES[endpoint_id] = gate
        return gate


def _connection():
    """Shared GPU-PC connection details, used by both remote_cloak and
    remote_upscale."""
    gpu_host = _env("GPU_HOST")
    gpu_user = _env("GPU_USER")
    gpu_remote_dir = _env("GPU_REMOTE_DIR", "C:/dontai-ml-engine")
    ssh_key = os.path.expanduser(_env("GPU_SSH_KEY", "~/.ssh/dontai_pi_to_gpu"))
    remote = f"{gpu_user}@{gpu_host}"
    ssh_opts = ["-i", ssh_key, "-o", "ConnectTimeout=10"]
    # -O forces the legacy SCP protocol instead of modern scp's default
    # SFTP-based transfer. Found for real, live: the GPU PC's Windows
    # OpenSSH sftp-server silently truncates downloads at exactly 204800
    # bytes -- scp reports success (exit 0) but the file is corrupt. Only
    # showed up once real files started exceeding ~200KB, which the old
    # fixed size=256 processing never did. Legacy -O transfers the same
    # file correctly and completely -- verified directly on the Pi against
    # this same GPU PC.
    scp_opts = [*ssh_opts, "-O"]
    return gpu_remote_dir, remote, ssh_opts, scp_opts


def _run(*args: str) -> subprocess.CompletedProcess:
    # encoding="utf-8" (not this machine's default cp949 Windows codepage)
    # -- found live, real: remote_multiarch_cloak's SSH call captured a
    # kohya_ss/diffusers progress bar containing Unicode block characters,
    # which crashed subprocess.run's internal stdout-reader thread trying
    # to decode it as cp949 (UnicodeDecodeError, silently swallowed by
    # Python's default threading.excepthook rather than raised here -- the
    # call still "succeeded" by luck since returncode/wait() don't depend
    # on that reader thread, but result.stdout/stderr would have come back
    # truncated or empty on any call unlucky enough to fail *and* hit this
    # at the same time, hiding the real error message). Same underlying
    # class of bug as gpu-infra-access's documented PowerShell EAP/cp949
    # issue, different process (Python's own subprocess module here, not
    # PowerShell) -- errors="replace" so a genuinely undecodable byte
    # degrades to a replacement character instead of crashing the read.
    result = subprocess.run(list(args), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"remote command failed: {' '.join(args)}\n{result.stderr}")
    return result


def _remote_job_paths(gpu_remote_dir: str, original_path: str, style_target_path: str) -> tuple[str, str, str]:
    """Same deterministic naming remote_cloak() uploads its inputs under --
    shared here so remote_compute_metrics() can reuse those same files
    (original, style target, and remote_cloak's own cloaked output) without
    re-uploading anything, as long as it's called right after remote_cloak()
    for the same original_path/style_target_path (which is how
    orchestrate.py's protect() actually calls it)."""
    remote_input = f"{gpu_remote_dir}/out/_remote_job_original{os.path.splitext(original_path)[1]}"
    remote_style = f"{gpu_remote_dir}/out/_remote_job_style{os.path.splitext(style_target_path)[1]}"
    remote_output = f"{gpu_remote_dir}/out/_remote_job_cloaked.png"
    return remote_input, remote_style, remote_output


def remote_cloak(
    original_path: str,
    style_target_path: str,
    output_path: str,
    preset_name: str,
    eot: bool,
    size: int = 256,
    eot_samples: int = 2,
    perceptual_mask: bool = False,
    use_amp: bool = False,
) -> None:
    """Runs style_cloak.py on the GPU PC and copies the result back to
    `output_path` (a local path on whatever machine calls this).
    """
    gpu_remote_dir, remote, ssh_opts, scp_opts = _connection()
    remote_input, remote_style, remote_output = _remote_job_paths(gpu_remote_dir, original_path, style_target_path)

    # 1. Upload the input images.
    _run("scp", *scp_opts, original_path, f"{remote}:{remote_input}")
    _run("scp", *scp_opts, style_target_path, f"{remote}:{remote_style}")

    # 2. Run style_cloak.py on the GPU PC, in its existing CUDA venv.
    eot_flag = "--eot" if eot else ""
    mask_flag = "--perceptual-mask" if perceptual_mask else ""
    amp_flag = "--amp" if use_amp else ""
    remote_cmd = (
        f"cd '{gpu_remote_dir}'; "
        f".\\.venv\\Scripts\\python.exe src/style_cloak.py "
        f"--original '{remote_input}' --style-target '{remote_style}' "
        f"--output '{remote_output}' --preset {preset_name} --size {size} "
        f"--eot-samples {eot_samples} {eot_flag} {mask_flag} {amp_flag}"
    )
    _run("ssh", *ssh_opts, remote, f'powershell -NoProfile -Command "{remote_cmd}"')

    # 3. Download the result.
    _run("scp", *scp_opts, f"{remote}:{remote_output}", output_path)


def remote_compute_metrics(original_path: str, style_target_path: str, size: int = 256) -> dict:
    """GPU-PC counterpart to evaluate.py's compute_protection_metrics() --
    orchestrate.py's protect() used to just skip this measurement entirely
    under USE_REMOTE_GPU (this machine, in practice the Pi, has no local
    torch worth relying on), leaving styleDriftScore/perceptualPsnrDb/
    styleSimilarityToOriginal permanently null for every real upload. Since
    remote_cloak() already uploaded the original and style-target images and
    downloaded the cloaked result under fixed, well-known remote paths (see
    _remote_job_paths), this just re-runs evaluate.py's own `--json` mode
    against those same three files already sitting on the GPU PC -- no new
    upload needed, only a single lightweight SSH command (three VGG19
    forward passes, "a second or two" per evaluate.py's own docstring) and
    its JSON stdout to parse. Must be called right after remote_cloak() for
    the same original_path/style_target_path, same requirement as that
    function's own doc note on _remote_job_paths.
    """
    gpu_remote_dir, remote, ssh_opts, _ = _connection()
    remote_input, remote_style, remote_output = _remote_job_paths(gpu_remote_dir, original_path, style_target_path)

    remote_cmd = (
        f"cd '{gpu_remote_dir}'; "
        f".\\.venv\\Scripts\\python.exe src/evaluate.py "
        f"--original '{remote_input}' --cloaked '{remote_output}' "
        f"--style-target '{remote_style}' --size {size} --json"
    )
    result = _run("ssh", *ssh_opts, remote, f'powershell -NoProfile -Command "{remote_cmd}"')

    import json

    return json.loads(result.stdout)


def remote_measure_existing_images(
    original_path: str, cloaked_path: str, style_target_path: str, size: int = 256
) -> dict:
    """Live, on-demand counterpart to remote_compute_metrics() -- that
    function only works right after remote_cloak() for the same job (it
    reuses remote_cloak's fixed, shared remote filenames). This is for the
    Test Lab's "재실행" (re-run) button: asset-service already has a
    *finished* artwork's decrypted original and its already-published
    protected image sitting on local disk, arbitrarily long after the
    original protect() job ran -- calling remote_compute_metrics() at that
    point would either race a concurrent real upload's own use of those
    same shared filenames, or read back whatever that job happens to have
    left there. Uploads its own three files under a fresh uuid-tagged name
    instead, so it can never collide with an in-flight protect() job (or
    another concurrent re-test), and best-effort deletes them again
    afterward -- a re-test's temp files have no reason to linger on the
    GPU PC once the measurement is done, unlike remote_cloak()'s output
    (which callers still need to scp back).
    """
    gpu_remote_dir, remote, ssh_opts, scp_opts = _connection()
    tag = uuid.uuid4().hex[:12]
    remote_original = f"{gpu_remote_dir}/out/_retest_{tag}_original{os.path.splitext(original_path)[1]}"
    remote_cloaked = f"{gpu_remote_dir}/out/_retest_{tag}_cloaked{os.path.splitext(cloaked_path)[1]}"
    remote_style = f"{gpu_remote_dir}/out/_retest_{tag}_style{os.path.splitext(style_target_path)[1]}"

    try:
        _run("scp", *scp_opts, original_path, f"{remote}:{remote_original}")
        _run("scp", *scp_opts, cloaked_path, f"{remote}:{remote_cloaked}")
        _run("scp", *scp_opts, style_target_path, f"{remote}:{remote_style}")

        remote_cmd = (
            f"cd '{gpu_remote_dir}'; "
            f".\\.venv\\Scripts\\python.exe src/evaluate.py "
            f"--original '{remote_original}' --cloaked '{remote_cloaked}' "
            f"--style-target '{remote_style}' --size {size} --json"
        )
        result = _run("ssh", *ssh_opts, remote, f'powershell -NoProfile -Command "{remote_cmd}"')

        import json

        return json.loads(result.stdout)
    finally:
        try:
            cleanup_cmd = f"Remove-Item -Force '{remote_original}','{remote_cloaked}','{remote_style}' -ErrorAction SilentlyContinue"
            _run("ssh", *ssh_opts, remote, f'powershell -NoProfile -Command "{cleanup_cmd}"')
        except RuntimeError:
            pass  # best-effort -- a leftover temp file on the GPU PC isn't worth failing the request over


def _multiarch_connection():
    """Connection details for the dedicated A40-class pod running
    ensemble_attack_multiarch.py -- deliberately separate env vars from
    _connection()'s GPU_HOST/GPU_USER/GPU_SSH_KEY (the owned 8GB GPU PC).
    This is a genuinely different machine (different venv layout, different
    checkpoints, a non-standard SSH port since it's a rented cloud pod, not
    a fixed LAN box) -- PHASE4_SCOPING.md §6 has the full reasoning for why
    this needed its own target instead of reusing remote_cloak()'s, and
    why "extend remote_gpu.py with a second host" won out over standing up
    a separate Serverless integration once ongoing A40-class capacity was
    confirmed available.
    """
    gpu_host = _env("MULTIARCH_GPU_HOST")
    gpu_user = _env("MULTIARCH_GPU_USER", "root")
    gpu_port = _env("MULTIARCH_GPU_PORT", "22")
    gpu_remote_dir = _env("MULTIARCH_GPU_REMOTE_DIR", "/workspace/dontai-protection-svc")
    ssh_key = os.path.expanduser(_env("MULTIARCH_GPU_SSH_KEY", "~/.ssh/dontai_runpod"))
    remote = f"{gpu_user}@{gpu_host}"
    ssh_opts = ["-i", ssh_key, "-p", gpu_port, "-o", "ConnectTimeout=15"]
    # scp uses -P (capital) for port, unlike ssh's -p -- same -O legacy-scp
    # note as _connection() does not apply here (this pod's OpenSSH sftp-
    # server doesn't have the GPU PC's Windows sftp-server truncation bug),
    # so no -O needed.
    scp_opts = ["-i", ssh_key, "-P", gpu_port, "-o", "ConnectTimeout=15"]
    return gpu_remote_dir, remote, ssh_opts, scp_opts


def remote_multiarch_cloak(
    original_path: str,
    output_path: str,
    prompt: str,
    preset_name: str = "MULTIARCH_FULL",
) -> None:
    """Runs ensemble_attack_multiarch.py's multiarch_ensemble_attack() on
    the dedicated A40-class pod (PHASE4_SCOPING.md §6) -- the validated
    SD1.5 LoRA-training protection effect, needs more VRAM (two full
    diffusion backbones, fp32) than remote_cloak()'s GPU PC target has.
    SD1.5-validated only; SDXL protection is not a covered case (see
    PHASE4_SCOPING.md §6's replication numbers) -- callers should not
    present this as broader "AI-training protection" than that.

    Several minutes per call (MULTIARCH_FULL's outer_iters=30) -- an order
    of magnitude slower than remote_cloak(). Callers should already be
    treating this as an async job (server.py's /protect is already a job
    queue for exactly this reason with the existing, faster mechanism), not
    something to await synchronously.

    `prompt` has no natural production equivalent to the experiments' own
    trigger-word captions (orchestrate.py's `protect()` has no real
    per-image caption in its data model -- same gap concept_misalign.py's
    module doc already flags for the same reason). Callers pass whatever
    descriptive text they have (e.g. the artwork's title) -- a rough
    proxy, not the literal mechanism the validation experiments used.
    """
    gpu_remote_dir, remote, ssh_opts, scp_opts = _multiarch_connection()
    # "MULTIARCH_KOHYA_PYTHON" is a legacy name from this project's
    # validation pods (which needed kohya_ss/sd-scripts for real LoRA
    # training) -- the production dontai-strongprotect Docker image
    # (docker/strongprotect/Dockerfile) doesn't clone kohya_ss at all,
    # since multiarch_ensemble_attack() itself never imports sd-scripts,
    # just torch/diffusers/peft. Default now points at that image's plain
    # system Python; kept the env var name for backward compat with any
    # already-deployed MULTIARCH_KOHYA_PYTHON override.
    multiarch_python = _env("MULTIARCH_KOHYA_PYTHON", "/usr/bin/python3")
    sd15_checkpoint = _env("MULTIARCH_SD15_CHECKPOINT", "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors")
    sdxl_checkpoint = _env("MULTIARCH_SDXL_CHECKPOINT", "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors")

    remote_input = f"{gpu_remote_dir}/out/_remote_multiarch_original{os.path.splitext(original_path)[1]}"
    remote_output = f"{gpu_remote_dir}/out/_remote_multiarch_cloaked.png"

    _run("ssh", *ssh_opts, remote, f"mkdir -p '{gpu_remote_dir}/out'")

    # 1. Upload the input image.
    _run("scp", *scp_opts, original_path, f"{remote}:{remote_input}")

    # 2. Run ensemble_attack_multiarch.py on the pod (torch/diffusers/peft
    #    installed directly into the dontai-strongprotect image's system
    #    Python -- see docker/strongprotect/Dockerfile).
    escaped_prompt = prompt.replace("'", "'\\''")  # single-quote-safe for the remote shell, not a Windows PowerShell target like remote_cloak's
    remote_cmd = (
        f"{multiarch_python} '{gpu_remote_dir}/ml-engine/src/ensemble_attack_multiarch.py' "
        f"--original '{remote_input}' --sd15-checkpoint '{sd15_checkpoint}' "
        f"--sdxl-checkpoint '{sdxl_checkpoint}' --prompt '{escaped_prompt}' "
        f"--output '{remote_output}' --preset {preset_name}"
    )
    _run("ssh", *ssh_opts, remote, remote_cmd)

    # 3. Download the result.
    _run("scp", *scp_opts, f"{remote}:{remote_output}", output_path)


def remote_dual_arch_cloak(
    original_path: str,
    output_path: str,
    prompt: str,
    sd15_preset: str = "L3_ANTI_TRAIN",
    sdxl_preset: str = "SDXL_FULL",
) -> None:
    """Runs aspl_attack.py (SD1.5) then aspl_attack_sdxl_only.py (SDXL) on
    the same dedicated A40-class pod remote_multiarch_cloak() uses --
    sequential single-architecture attacks, not the joint
    multiarch_ensemble_attack(). PHASE4_SCOPING.md §6's follow-up finding:
    attacking SD1.5 and SDXL *jointly* (sharing one epsilon budget across
    both backbones each PGD step) measurably suppresses SDXL's own effect
    to nothing, while attacking SDXL *alone* with the same mechanism clears
    the same 4-way validation bar (original n=30, two outlier-robustness
    checks, independent n=12 replication) SD1.5 already had. So this
    function replaces the joint call as strong_protection's mechanism:
    each architecture gets its own full attack, chained (SD1.5 attacks the
    original, SDXL then attacks *that* output) so the final image reflects
    both.

    This exact chained composition (SDXL attacking an already-SD1.5-attacked
    image, not a pristine original) has itself been validated at n=30
    (ml-engine/experiments/dual_arch_validation/run_dual_arch_n30.py,
    2026-08-07): SD1.5 mean delta +0.1655, 95% CI [+0.1341, +0.1968] --
    roughly 4-5x the single-stage effect, the compounding perturbation
    acting like a much larger effective epsilon budget rather than the two
    stages fighting each other. SDXL mean delta +0.0446, 95% CI
    [+0.0211, +0.0681], robust to removing 1-2 outlier images -- comparable
    to (if anything slightly better than) its own single-stage validation.
    Neither architecture's effect is suppressed by the chaining; SD1.5's is
    amplified. See [[lora-protection-research]] memory for full numbers.

    Takes roughly 2x remote_multiarch_cloak()'s time (two full ~10min
    single-architecture attacks instead of one ~15min joint one) -- still
    an async-job-queue candidate, same as remote_multiarch_cloak().
    """
    gpu_remote_dir, remote, ssh_opts, scp_opts = _multiarch_connection()
    multiarch_python = _env("MULTIARCH_KOHYA_PYTHON", "/usr/bin/python3")
    sd15_checkpoint = _env("MULTIARCH_SD15_CHECKPOINT", "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors")
    sdxl_checkpoint = _env("MULTIARCH_SDXL_CHECKPOINT", "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors")

    remote_input = f"{gpu_remote_dir}/out/_remote_dual_original{os.path.splitext(original_path)[1]}"
    remote_intermediate = f"{gpu_remote_dir}/out/_remote_dual_sd15.png"
    remote_output = f"{gpu_remote_dir}/out/_remote_dual_final.png"

    _run("ssh", *ssh_opts, remote, f"mkdir -p '{gpu_remote_dir}/out'")

    # 1. Upload the input image.
    _run("scp", *scp_opts, original_path, f"{remote}:{remote_input}")

    escaped_prompt = prompt.replace("'", "'\\''")

    # 2. SD1.5 attack on the original.
    sd15_cmd = (
        f"{multiarch_python} '{gpu_remote_dir}/ml-engine/src/aspl_attack.py' "
        f"--original '{remote_input}' --checkpoint '{sd15_checkpoint}' "
        f"--prompt '{escaped_prompt}' --output '{remote_intermediate}' --preset {sd15_preset}"
    )
    _run("ssh", *ssh_opts, remote, sd15_cmd)

    # 3. SDXL attack chained on the SD1.5-attacked output.
    sdxl_cmd = (
        f"{multiarch_python} '{gpu_remote_dir}/ml-engine/src/aspl_attack_sdxl_only.py' "
        f"--original '{remote_intermediate}' --sdxl-checkpoint '{sdxl_checkpoint}' "
        f"--prompt '{escaped_prompt}' --output '{remote_output}' --preset {sdxl_preset}"
    )
    _run("ssh", *ssh_opts, remote, sdxl_cmd)

    # 4. Download the final result.
    _run("scp", *scp_opts, f"{remote}:{remote_output}", output_path)


def cancel_runpod_job(endpoint_id: str, runpod_job_id: str) -> bool:
    """Best-effort direct RunPod job cancel (2026-08-14, cancel-upload
    feature) -- used by server.py's /protect/{job_id}/cancel route once it
    has a runpod_job_id on file (set via on_submitted below). Returns
    False on any failure (job already finished, network error, etc.)
    rather than raising -- cancellation is inherently racy against a job
    that might complete in the same instant, and the caller (jobs_db's
    request_cancel) has already marked the job cancelled locally either
    way, so a failed RunPod-side cancel just means slightly wasted GPU
    time, not an inconsistent job state."""
    import httpx

    api_key = _env("RUNPOD_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.post(
            f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{runpod_job_id}", headers=headers, timeout=15.0
        )
        return resp.status_code < 400
    except httpx.HTTPError:
        return False


def serverless_dual_arch_cloak(
    original_path: str,
    output_path: str,
    prompt: str,
    hybrid_preset: str = "CLEAN_FULL",
    latent_epsilon: float | None = None,
    pixel_epsilon: float | None = None,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 3300.0,
    on_submitted: Callable[[str, str], None] | None = None,
) -> None:
    """RunPod Serverless counterpart to remote_dual_arch_cloak() -- calls
    the `dontai-strongprotect` Serverless endpoint (docker/
    strongprotect-serverless/handler.py), which as of 2026-08-13 defaults
    to clean_protect.py's four-stage native pixel-space-only chain (no
    VAE-decode latent stage) instead of hybrid_protect.py's latent-then-
    pixel composition -- see clean_protect.py's own module doc for why:
    hybrid_protect.py's latent stage was confirmed causing real color/
    quality damage (oil-painting-style distortion, a night sky reduced to
    magenta/green blotches) on a real user's deployed artwork. Same
    STATUS caveat as before this switch: n=1, single-image validated, not
    this project's usual n=30-plus-replication bar; wired in ahead of
    that (2026-08-13 decision) because a visually honest n=1 mechanism
    beats a known-broken one staying live. `hybrid_preset` is now a
    misnomer kept for API-compatibility -- it's actually a
    clean_protect.CLEAN_PRESETS name ("CLEAN_FULL" or "CALIBRATION"), not
    a hybrid_protect.py preset; renaming the parameter is left for a
    follow-up since orchestrate.py's callers pass it positionally-safe
    keyword args either way. `latent_epsilon`/`pixel_epsilon` are
    currently NOT threaded through to clean_protect() (it has no latent
    stage, so "latent_epsilon" has no meaning, and per-stage pixel
    epsilon overrides aren't wired yet) -- passing them is a silent no-op
    for now; a caller relying on the old "advanced options" epsilon
    override will not get the effect they expect until that's built.
    timeout_seconds raised from the four-stage hybrid chain's 2400s to
    3300s (2026-08-13, first real deployment run): the dontai-strongprotect
    RunPod endpoint's own executionTimeoutMs was 1800000 (30min), shorter
    than clean_protect.py's real four-stage wall-clock time -- a live run
    got through stage 1 (349s) into stage 2 before RunPod's own timeout
    killed the job server-side ("executionTimeout exceeded"), independent
    of this function's client-side poll timeout entirely. Fixed by raising
    the endpoint's own timeout to 3000000ms (50min) via update-endpoint;
    this function's timeout_seconds must stay >= that or the client gives
    up and reports failure before the server would even time out.

    Why Serverless over remote_dual_arch_cloak()'s SSH-to-a-pod path
    (PHASE4_SCOPING.md §6's 2026-08-07 update): pure execution time was a
    wash for the original two-stage chain (Serverless 685.1s vs Pods
    718.8s, one A40 job), but Serverless scales to zero automatically --
    the SSH path assumes a pod that's already running, which in practice
    meant either paying for one sitting idle 24/7 or it simply not being
    up when a real request needed it (this session found and deleted a
    pod idle for 4.5 hours before anyone noticed, for exactly this
    reason). Serverless removes that failure mode by construction -- a
    worker only exists while a job is actually running.

    Two-phase HTTP job protocol (not runsync -- this job runs several
    minutes, well past what a single blocking HTTP call should be relied
    on for): POST .../run submits and returns immediately with a job id;
    this then polls GET .../status/{id} until COMPLETED/FAILED/CANCELLED/
    TIMED_OUT, same shape as every other job-polling loop in this project
    (jobs_db, asset-service's pollProtectJob, detection-svc's
    poll_leak_detection_job).
    """
    import base64
    import time

    import httpx

    api_key = _env("RUNPOD_API_KEY")
    endpoint_id = _env("RUNPOD_STRONGPROTECT_ENDPOINT_ID")
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    with open(original_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    gate = _endpoint_gate(endpoint_id, "RUNPOD_STRONGPROTECT_MAX_CONCURRENT")
    with gate:  # see _endpoint_gate's own doc -- caps concurrent in-flight jobs to this endpoint's real worker capacity
        submit = httpx.post(
            f"{base_url}/run",
            headers=headers,
            json={
                "input": {
                    # "clean_cloak" (2026-08-13) -- see this function's own
                    # doc for why it replaced the implicit "dual_arch_cloak"
                    # default. Explicit here rather than relying on the
                    # handler's own default so this call site's intent is
                    # visible without cross-referencing handler.py.
                    "action": "clean_cloak",
                    "image_b64": image_b64,
                    "prompt": prompt,
                    "clean_preset": hybrid_preset,
                    # NOT currently wired to clean_protect() -- see this
                    # function's own doc. Sent anyway (harmlessly ignored by
                    # the handler) so a future override implementation
                    # doesn't also need an orchestrate.py-side change.
                    "latent_epsilon": latent_epsilon,
                    "pixel_epsilon": pixel_epsilon,
                }
            },
            timeout=30.0,
        )
        submit.raise_for_status()
        job_id = submit.json()["id"]
        # Fires as early as possible (2026-08-14, cancel-upload feature) --
        # server.py's callback writes this into jobs_db so a cancel request
        # arriving seconds later can still reach the real RunPod job, not
        # just this function's own in-memory job_id (which a cancel caller
        # in a different process/thread has no access to otherwise).
        if on_submitted is not None:
            on_submitted(job_id, endpoint_id)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_resp = httpx.get(f"{base_url}/status/{job_id}", headers=headers, timeout=30.0)
            status_resp.raise_for_status()
            body = status_resp.json()
            status = body["status"]

            if status == "COMPLETED":
                output_b64 = body["output"]["output_b64"]
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(output_b64))
                return
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod Serverless job {job_id} ended with status {status}: {body.get('error')}")

            time.sleep(poll_interval_seconds)

        raise RuntimeError(f"RunPod Serverless job {job_id} did not complete within {timeout_seconds}s")


def serverless_score_protection(
    original_path: str,
    protected_path: str,
    prompt: str,
    seed: int = 1,
    train_steps: int = 150,
    num_samples: int = 2,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 1800.0,
) -> dict:
    """Test Lab's on-demand real-LoRA-training protection score (PHASE4_
    SCOPING.md §6 follow-up: give users the same kind of evidence this
    project's own n=30 dual_arch_validation used, for their own artwork,
    not just a proxy VGG19-feature-distance number). Calls the same
    `dontai-strongprotect` Serverless endpoint serverless_dual_arch_cloak()
    uses, with `action: "score_protection"` in the job input so the
    handler (docker/strongprotect-serverless/handler.py) dispatches to
    protection_score.py instead of the attack chain.

    Same submit-then-poll two-phase protocol as serverless_dual_arch_
    cloak() -- this is slower, not faster (four LoRA trainings: SD1.5
    baseline/protected, SDXL baseline/protected), so the default timeout
    matches that function's.

    Unlike serverless_dual_arch_cloak() (which writes its result to a
    local file), this returns the parsed result dict directly -- the
    caller wants the whole JSON (per-architecture delta/verdict/sample
    images as base64), not a single output image.
    """
    import base64
    import io
    import time

    import httpx
    from PIL import Image

    api_key = _env("RUNPOD_API_KEY")
    endpoint_id = _env("RUNPOD_STRONGPROTECT_ENDPOINT_ID")
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    # protection_score.py's own LoRA training resizes everything down to
    # 512px (SD1.5) / 1024px (SDXL) anyway (load_image_tensor) -- sending
    # the real upload's native resolution (a real production image can be
    # several thousand px, several MB as PNG) gains nothing and, sending
    # *two* such images in one JSON body, is exactly what pushed a real
    # request over RunPod's /run payload size limit (400 Bad Request, no
    # further detail in the response body -- found live on a 2400x1800
    # upload, ~9MB PNG each for original+protected, well past whatever
    # the actual cap is). 1024 matches SDXL's own training resolution --
    # not a lossy downgrade relative to what the job would use anyway.
    def _encode_resized(path: str, max_dim: int = 1024) -> str:
        img = Image.open(path).convert("RGB")
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    original_b64 = _encode_resized(original_path)
    protected_b64 = _encode_resized(protected_path)

    gate = _endpoint_gate(endpoint_id, "RUNPOD_STRONGPROTECT_MAX_CONCURRENT")
    with gate:  # see _endpoint_gate's own doc -- this and serverless_dual_arch_cloak/serverless_generate_lora share one endpoint's worker capacity
        submit = httpx.post(
            f"{base_url}/run",
            headers=headers,
            json={
                "input": {
                    "action": "score_protection",
                    "original_b64": original_b64,
                    "protected_b64": protected_b64,
                    "prompt": prompt,
                    "seed": seed,
                    "train_steps": train_steps,
                    "num_samples": num_samples,
                }
            },
            timeout=30.0,
        )
        submit.raise_for_status()
        job_id = submit.json()["id"]

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_resp = httpx.get(f"{base_url}/status/{job_id}", headers=headers, timeout=30.0)
            status_resp.raise_for_status()
            body = status_resp.json()
            status = body["status"]

            if status == "COMPLETED":
                return body["output"]
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod Serverless job {job_id} ended with status {status}: {body.get('error')}")

            time.sleep(poll_interval_seconds)

        raise RuntimeError(f"RunPod Serverless job {job_id} did not complete within {timeout_seconds}s")


def serverless_generate_lora(
    image_path: str,
    output_path: str,
    prompt: str,
    seed: int = 1,
    train_steps: int = 150,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 1800.0,
) -> dict:
    """Coin-system feature (2026-08-10): trains a real, downloadable SD1.5
    LoRA on a single one of the user's own artworks (ml-engine/src/
    lora_generate.py's module doc has the full mechanism -- reuses
    protection_score.py's own `_train_lora_sd15` training loop). Calls the
    same `dontai-strongprotect` Serverless endpoint as serverless_
    dual_arch_cloak()/serverless_score_protection(), with `action:
    "generate_lora"` so the handler dispatches to lora_generate.py instead.

    Single training run (not four like score_protection), so timeout
    defaults to score_protection's own bound rather than something
    tighter -- one SD1.5 LoRA training at these settings has run well
    under that in practice, this just isn't the place to be optimistic
    about wall-clock time.

    Writes the resulting .safetensors file to output_path (like
    serverless_dual_arch_cloak(), unlike serverless_score_protection()
    which returns its result inline) -- the caller wants a file on disk to
    hand off to asset-service's own storage, not a JSON blob.
    """
    import base64
    import time

    import httpx

    api_key = _env("RUNPOD_API_KEY")
    endpoint_id = _env("RUNPOD_STRONGPROTECT_ENDPOINT_ID")
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    gate = _endpoint_gate(endpoint_id, "RUNPOD_STRONGPROTECT_MAX_CONCURRENT")
    with gate:  # see _endpoint_gate's own doc -- shares this endpoint's worker capacity with serverless_dual_arch_cloak/serverless_score_protection
        submit = httpx.post(
            f"{base_url}/run",
            headers=headers,
            json={
                "input": {
                    "action": "generate_lora",
                    "image_b64": image_b64,
                    "prompt": prompt,
                    "seed": seed,
                    "train_steps": train_steps,
                }
            },
            timeout=30.0,
        )
        submit.raise_for_status()
        job_id = submit.json()["id"]

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_resp = httpx.get(f"{base_url}/status/{job_id}", headers=headers, timeout=30.0)
            status_resp.raise_for_status()
            body = status_resp.json()
            status = body["status"]

            if status == "COMPLETED":
                output_b64 = body["output"]["output_b64"]
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(output_b64))
                return {"outputPath": output_path, "contentPrompt": body["output"].get("contentPrompt")}
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod Serverless job {job_id} ended with status {status}: {body.get('error')}")

            time.sleep(poll_interval_seconds)

        raise RuntimeError(f"RunPod Serverless job {job_id} did not complete within {timeout_seconds}s")


def remote_detect_model_leak(
    original_path: str,
    suspect_lora_path: str,
    prompts: list[str],
    num_samples: int = 4,
    resolution: int = 512,
    gen_seed: int = 42,
) -> dict:
    """Runs model_leak_detect.py's detect_model_leak() on the same
    dedicated A40-class pod remote_multiarch_cloak() uses -- it already has
    the exact dependency this needs (diffusers/transformers/peft/torch,
    no kohya_ss/sd-scripts required, since detect_model_leak() only
    generates images and loads LoRA weights, never trains one), just for a
    different script. Reuses _multiarch_connection() rather than
    _connection()'s owned 8GB GPU PC -- that machine's plain ml-engine venv
    (remote_cloak()'s target) has no diffusers at all.

    Uploads the original (registered) image and the suspect LoRA file
    under a fresh uuid tag (same never-collide-with-a-concurrent-call
    reasoning as remote_measure_existing_images), runs the detector, and
    best-effort cleans up both uploaded files afterward -- unlike
    remote_multiarch_cloak's output, nothing here needs to persist on the
    pod once the JSON result is back.
    """
    gpu_remote_dir, remote, ssh_opts, scp_opts = _multiarch_connection()
    multiarch_python = _env("MULTIARCH_KOHYA_PYTHON", "/usr/bin/python3")
    sd15_checkpoint = _env("MULTIARCH_SD15_CHECKPOINT", "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors")

    tag = uuid.uuid4().hex[:12]
    remote_original = f"{gpu_remote_dir}/out/_leak_{tag}_original{os.path.splitext(original_path)[1]}"
    remote_lora = f"{gpu_remote_dir}/out/_leak_{tag}_suspect{os.path.splitext(suspect_lora_path)[1]}"

    _run("ssh", *ssh_opts, remote, f"mkdir -p '{gpu_remote_dir}/out'")

    try:
        _run("scp", *scp_opts, original_path, f"{remote}:{remote_original}")
        _run("scp", *scp_opts, suspect_lora_path, f"{remote}:{remote_lora}")

        prompts_arg = "|".join(prompts).replace("'", "'\\''")
        remote_cmd = (
            f"{multiarch_python} '{gpu_remote_dir}/ml-engine/src/model_leak_detect.py' "
            f"--checkpoint '{sd15_checkpoint}' --suspect-lora '{remote_lora}' "
            f"--original '{remote_original}' --prompts '{prompts_arg}' "
            f"--num-samples {num_samples} --resolution {resolution} --gen-seed {gen_seed}"
        )
        result = _run("ssh", *ssh_opts, remote, remote_cmd)

        import json

        return json.loads(result.stdout)
    finally:
        try:
            _run("ssh", *ssh_opts, remote, f"rm -f '{remote_original}' '{remote_lora}'")
        except RuntimeError:
            pass  # best-effort -- a leftover temp file on the pod isn't worth failing the request over


def remote_upscale(input_path: str, output_path: str, target_width: int, target_height: int) -> None:
    """Runs upscale.py's super-resolution restoration step on the GPU PC
    instead of locally. Found for real, live, in production: loading torch
    + the EDSR CNN and running it on a real near-native-resolution image
    (the resolution fix processes up to 1024px now, vs. the old fixed 256px)
    grew protection-svc's memory footprint to ~7.1GB resident, which the
    Raspberry Pi's kernel OOM-killer then killed outright -- taking down
    every in-flight job on the Pi, not just the one that triggered it. This
    delegates the step to the GPU PC instead, the same pattern remote_cloak
    already uses for the same underlying reason (no GPU / limited resources
    on the Pi -- see this module's own doc).
    """
    gpu_remote_dir, remote, ssh_opts, scp_opts = _connection()

    remote_input = f"{gpu_remote_dir}/out/_remote_job_upscale_input{os.path.splitext(input_path)[1]}"
    remote_output = f"{gpu_remote_dir}/out/_remote_job_upscaled.png"

    _run("scp", *scp_opts, input_path, f"{remote}:{remote_input}")

    remote_cmd = (
        f"cd '{gpu_remote_dir}'; "
        f".\\.venv\\Scripts\\python.exe src/upscale.py "
        f"--input '{remote_input}' --output '{remote_output}' "
        f"--target-width {target_width} --target-height {target_height}"
    )
    _run("ssh", *ssh_opts, remote, f'powershell -NoProfile -Command "{remote_cmd}"')

    _run("scp", *scp_opts, f"{remote}:{remote_output}", output_path)


def _stylecloak_base() -> tuple[str, str, dict]:
    """Shared connection details for the three serverless_* functions
    below -- same RUNPOD_API_KEY as strong_protection's own serverless
    functions, but a separate endpoint (RUNPOD_STYLECLOAK_ENDPOINT_ID):
    a deliberately small, checkpoint-free image (docker/stylecloak/),
    not the ~15GB dual-arch attack image -- see that Dockerfile's own doc
    for why L1-L3 needs its own lightweight endpoint instead of reusing
    strong_protection's. Returns endpoint_id alongside base_url/headers
    (2026-08-14) so callers can key a per-endpoint concurrency gate off
    it -- see _endpoint_gate's own doc."""
    api_key = _env("RUNPOD_API_KEY")
    endpoint_id = _env("RUNPOD_STYLECLOAK_ENDPOINT_ID")
    headers = {"Authorization": f"Bearer {api_key}"}
    return f"https://api.runpod.ai/v2/{endpoint_id}", endpoint_id, headers


def _stylecloak_submit_and_poll(
    job_input: dict, poll_interval_seconds: float, timeout_seconds: float
) -> dict:
    """Same two-phase submit-then-poll protocol serverless_dual_arch_cloak/
    serverless_score_protection already use -- see that function's own doc
    for why (job runs longer than a single blocking HTTP call should be
    trusted for, even though this tier is meant to be fast). Also shares
    that function's per-endpoint concurrency gate (2026-08-14, see
    _endpoint_gate's own doc) -- serverless_cloak/serverless_compute_metrics/
    serverless_upscale all fan into this one helper and this one endpoint
    (RUNPOD_STYLECLOAK_ENDPOINT_ID, workers.max=2), so without a gate here
    the same overload risk serverless_dual_arch_cloak's gate addresses
    would apply to this endpoint too."""
    import time

    import httpx

    base_url, endpoint_id, headers = _stylecloak_base()

    gate = _endpoint_gate(endpoint_id, "RUNPOD_STYLECLOAK_MAX_CONCURRENT")
    with gate:
        submit = httpx.post(f"{base_url}/run", headers=headers, json={"input": job_input}, timeout=30.0)
        submit.raise_for_status()
        job_id = submit.json()["id"]

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_resp = httpx.get(f"{base_url}/status/{job_id}", headers=headers, timeout=30.0)
            status_resp.raise_for_status()
            body = status_resp.json()
            status = body["status"]

            if status == "COMPLETED":
                return body["output"]
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod Serverless job {job_id} ended with status {status}: {body.get('error')}")

            time.sleep(poll_interval_seconds)

        raise RuntimeError(f"RunPod Serverless job {job_id} did not complete within {timeout_seconds}s")


def serverless_cloak(
    original_path: str,
    style_target_path: str,
    output_path: str,
    preset_name: str,
    size: int = 256,
    eot: bool = False,
    eot_samples: int = 2,
    perceptual_mask: bool = False,
    use_amp: bool = False,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 600.0,
) -> None:
    """RunPod Serverless counterpart to remote_cloak() -- calls the
    `dontai-stylecloak` endpoint's "cloak" action (docker/
    stylecloak-serverless/handler.py), which runs style_cloak.py's cloak()
    directly. Replaces the SSH-to-GPU-PC path for L1_PREVIEW/L2_PORTFOLIO/
    L3_ANTI_TRAIN the same way serverless_dual_arch_cloak already replaced
    it for strong_protection (PHASE4_SCOPING.md §6) -- see this module's
    own docker/stylecloak/Dockerfile doc for why this is a separate,
    smaller endpoint rather than reusing strong_protection's.
    """
    import base64

    with open(original_path, "rb") as f:
        original_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(style_target_path, "rb") as f:
        style_target_b64 = base64.b64encode(f.read()).decode("ascii")

    output = _stylecloak_submit_and_poll(
        {
            "action": "cloak",
            "original_b64": original_b64,
            "style_target_b64": style_target_b64,
            "preset_name": preset_name,
            "size": size,
            "eot": eot,
            "eot_samples": eot_samples,
            "perceptual_mask": perceptual_mask,
            "use_amp": use_amp,
        },
        poll_interval_seconds,
        timeout_seconds,
    )

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(output["output_b64"]))


def serverless_compute_metrics(
    original_path: str,
    cloaked_path: str,
    style_target_path: str,
    size: int = 256,
    poll_interval_seconds: float = 3.0,
    timeout_seconds: float = 120.0,
) -> dict:
    """RunPod Serverless counterpart to remote_compute_metrics() -- calls
    the `dontai-stylecloak` endpoint's "compute_metrics" action. Cheap
    relative to serverless_cloak() (three VGG19 forward passes, no
    optimization loop -- same cost class evaluate.py's own
    compute_protection_metrics() doc already describes)."""
    import base64

    with open(original_path, "rb") as f:
        original_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(cloaked_path, "rb") as f:
        cloaked_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(style_target_path, "rb") as f:
        style_target_b64 = base64.b64encode(f.read()).decode("ascii")

    return _stylecloak_submit_and_poll(
        {
            "action": "compute_metrics",
            "original_b64": original_b64,
            "cloaked_b64": cloaked_b64,
            "style_target_b64": style_target_b64,
            "size": size,
        },
        poll_interval_seconds,
        timeout_seconds,
    )


def serverless_upscale(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    poll_interval_seconds: float = 3.0,
    timeout_seconds: float = 180.0,
) -> None:
    """RunPod Serverless counterpart to remote_upscale() -- calls the
    `dontai-stylecloak` endpoint's "upscale" action. Same reasoning as
    remote_upscale's own doc for why this step is delegated at all (a
    resource-constrained host running the EDSR CNN locally can OOM) --
    RunPod Serverless removes the "needs an always-on GPU PC" half of
    that same problem too."""
    import base64

    with open(input_path, "rb") as f:
        input_b64 = base64.b64encode(f.read()).decode("ascii")

    output = _stylecloak_submit_and_poll(
        {
            "action": "upscale",
            "input_b64": input_b64,
            "target_width": target_width,
            "target_height": target_height,
        },
        poll_interval_seconds,
        timeout_seconds,
    )

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(output["output_b64"]))
