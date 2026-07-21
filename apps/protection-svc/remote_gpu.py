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


def _run(*args: str) -> None:
    result = subprocess.run(list(args), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"remote command failed: {' '.join(args)}\n{result.stderr}")


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

    remote_input = f"{gpu_remote_dir}/out/_remote_job_original{os.path.splitext(original_path)[1]}"
    remote_style = f"{gpu_remote_dir}/out/_remote_job_style{os.path.splitext(style_target_path)[1]}"
    remote_output = f"{gpu_remote_dir}/out/_remote_job_cloaked.png"

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
