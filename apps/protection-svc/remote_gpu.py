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
import uuid


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"remote_gpu.py: required env var {name} is not set")
    return value


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
    result = subprocess.run(list(args), capture_output=True, text=True)
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
    kohya_python = _env("MULTIARCH_KOHYA_PYTHON", "/workspace/kohya_ss/venv/bin/python")
    sd15_checkpoint = _env("MULTIARCH_SD15_CHECKPOINT", "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors")
    sdxl_checkpoint = _env("MULTIARCH_SDXL_CHECKPOINT", "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors")

    remote_input = f"{gpu_remote_dir}/out/_remote_multiarch_original{os.path.splitext(original_path)[1]}"
    remote_output = f"{gpu_remote_dir}/out/_remote_multiarch_cloaked.png"

    _run("ssh", *ssh_opts, remote, f"mkdir -p '{gpu_remote_dir}/out'")

    # 1. Upload the input image.
    _run("scp", *scp_opts, original_path, f"{remote}:{remote_input}")

    # 2. Run ensemble_attack_multiarch.py on the pod, in its kohya_ss venv
    #    (needs diffusers + peft + accelerate, same as remote_cloak's GPU
    #    PC venv needs for style_cloak.py, just a different machine/venv).
    escaped_prompt = prompt.replace("'", "'\\''")  # single-quote-safe for the remote shell, not a Windows PowerShell target like remote_cloak's
    remote_cmd = (
        f"{kohya_python} '{gpu_remote_dir}/ml-engine/src/ensemble_attack_multiarch.py' "
        f"--original '{remote_input}' --sd15-checkpoint '{sd15_checkpoint}' "
        f"--sdxl-checkpoint '{sdxl_checkpoint}' --prompt '{escaped_prompt}' "
        f"--output '{remote_output}' --preset {preset_name}"
    )
    _run("ssh", *ssh_opts, remote, remote_cmd)

    # 3. Download the result.
    _run("scp", *scp_opts, f"{remote}:{remote_output}", output_path)


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
