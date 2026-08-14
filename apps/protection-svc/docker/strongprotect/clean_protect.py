"""Native-resolution SD1.5+SDXL sequential-chain protection -- the
production replacement for hybrid_protect.py.

WHY THIS REPLACES hybrid_protect.py (2026-08-13): hybrid_protect.py was
wired into production (remote_gpu.py's serverless_dual_arch_cloak, via the
"dontai-strongprotect" Serverless endpoint) at HYBRID_FULL despite its own
module doc explicitly saying "STATUS: this composition is DESIGNED, not
validated... needs n=30 plus independent replication before it goes
anywhere near production." Its two-latent-stage design (SD1.5 latent then
SDXL latent, both at latent_epsilon=0.15) is EXACTLY the combination this
project's own R&D (see [[lora-protection-research]], "16th experiment
follow-up 12") independently found to produce severe oil-painting-style
distortion and color drift on real illustrations -- confirmed live on a
user's own deployed artwork (a night-sky fireworks scene came back with
the sky reduced to magenta/green blotches, barely recognizable).

WHAT THIS MODULE DOES INSTEAD: pixel-space-only, no VAE-decode latent
stage at all -- sequential chaining of native_lowfreq_attack.py (SD1.5)
and native_aspl_sdxl_attack.py (SDXL), each getting its own full epsilon
budget in its own pass (same "budgets compose instead of competing"
principle hybrid_protect.py itself documents, just without the latent
stage that turned out to be the actual problem).

THE CHECKERBOARD/LEOPARD-PATTERN FIX (2026-08-13): every earlier SDXL
result this project produced (regardless of latent vs pixel space) showed
a persistent checkerboard/mottled-texture artifact. Root cause, found by
elimination after mask-strength, stage-count, and SD1.5-grid-resolution
fixes all failed to remove it: `native_aspl_sdxl_attack.py`'s delta is
optimized on a coarse `param_size` grid (1024, on a 1920px-wide image)
then bicubic-upsampled to native resolution before being added to the
original -- that upsample step itself introduces moire/aliasing. Raising
`param_size` toward native resolution removes it, at a real but bounded
protective-strength cost (SDXL solo delta: param_size=1024 -> 0.0934
[checkerboard], 1536 -> 0.0559 [clean], 1920 -> 0.0355 [clean, weaker]).
1536 is the resolution/strength sweet spot this module uses.

Measured full-chain result (n=1, demo illustration, 1920x1080):
delta_sd15 +0.1097, delta_sdxl +0.1035 -- SDXL delta exceeds its OWN solo
record (+0.0934) because the SD1.5 stage's fixed-hue perturbation gives
the SDXL surrogate a slightly different starting point, not a loss. Both
numbers are somewhat below hybrid_protect.py's HYBRID_FULL n=1 claims, but
those numbers came from a mechanism now confirmed to visibly break real
images -- a working, visually honest 0.10-0.11 beats a broken 0.12-0.13.

STATUS: same validation bar as hybrid_protect.py had -- this is an n=1,
single-image result, not yet the n=30-plus-replication bar this project
requires before trusting a delta claim. Wired into production ahead of
that (mirroring the 2026-08-08 hybrid_protect.py decision) because the
alternative -- leaving the known-broken hybrid_protect.py live -- is worse
than shipping a mechanism that is at minimum visually honest, per the
2026-08-13 user decision to prioritize this fix immediately given a real
deployed artwork was affected.

STAGE ORDER IS LOAD-BEARING, same as hybrid_protect.py: SD1.5 runs first,
SDXL second. Measured directly (2026-08-10) -- reversing the order
(SDXL first, SD1.5 second) produced WORSE deltas on BOTH architectures
(0.0905/0.0879 vs forward order's 0.0992/0.1063 at the pre-param_size-fix
settings), not just a shifted tradeoff. This isn't obviously explainable
and wasn't re-verified after the param_size fix, but there's no reason to
expect the asymmetry disappeared.
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ML_SRC = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CleanPreset:
    sd15_param_size: int
    sd15_epsilon: float
    sd15_steps: int
    sdxl_param_size: int
    sdxl_epsilon: float
    sdxl_outer_iters: int
    sd15_topup_epsilon: float
    sd15_topup_steps: int
    sdxl_topup_epsilon: float
    sdxl_topup_outer_iters: int


CLEAN_PRESETS = {
    # Fast wiring/timing check before spending real GPU-hours on the full
    # thing -- same role CALIBRATION plays in hybrid_protect.py.
    "CALIBRATION": CleanPreset(
        sd15_param_size=512, sd15_epsilon=0.05, sd15_steps=20,
        sdxl_param_size=768, sdxl_epsilon=0.06, sdxl_outer_iters=5,
        sd15_topup_epsilon=0.02, sd15_topup_steps=10,
        sdxl_topup_epsilon=0.02, sdxl_topup_outer_iters=3,
    ),
    # The n=1-measured recipe from this module's own doc above. delta_sd15
    # +0.1097, delta_sdxl +0.1035.
    "CLEAN_FULL": CleanPreset(
        sd15_param_size=1024, sd15_epsilon=0.05, sd15_steps=500,
        sdxl_param_size=1536, sdxl_epsilon=0.06, sdxl_outer_iters=90,
        sd15_topup_epsilon=0.02, sd15_topup_steps=200,
        sdxl_topup_epsilon=0.02, sdxl_topup_outer_iters=30,
    ),
}


def _run_stage(label: str, cmd: list[str]) -> float:
    print(f"=== {label} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(f"  {label} done in {elapsed:.1f}s", flush=True)
    return elapsed


def clean_protect(
    original_path: str,
    sd15_checkpoint: str,
    sdxl_checkpoint: str,
    prompt: str,
    output_path: str,
    preset_name: str = "CLEAN_FULL",
    seed: int = 0,
    work_dir: str | None = None,
) -> dict:
    """Runs the four-stage native pixel-space stack and writes the final
    protected image. Each stage is a subprocess -- same reason
    hybrid_protect.py's stages are: each attack script loads its own full
    pipeline into VRAM and expects to tear it down on exit."""
    preset = CLEAN_PRESETS[preset_name]
    work = Path(work_dir) if work_dir else Path(output_path).parent / "_clean_stages"
    work.mkdir(parents=True, exist_ok=True)

    s1 = work / "stage1_sd15.png"
    s2 = work / "stage2_sdxl.png"
    s3 = work / "stage3_sd15_topup.png"

    py = sys.executable
    timings = {}

    timings["stage1_sd15"] = _run_stage(
        "stage 1/4: SD1.5 pixel (full budget)",
        [
            py, str(ML_SRC / "native_lowfreq_attack.py"),
            "--original", original_path,
            "--checkpoint", sd15_checkpoint,
            "--output", str(s1),
            "--param-size", str(preset.sd15_param_size),
            "--epsilon", str(preset.sd15_epsilon),
            "--steps", str(preset.sd15_steps),
            "--eot-samples", "1",
            "--recon-weight", "0.5",
            "--param-smooth-sigma", "1.0",
            "--seed", str(seed),
        ],
    )

    timings["stage2_sdxl"] = _run_stage(
        "stage 2/4: SDXL pixel (full budget)",
        [
            py, str(ML_SRC / "native_aspl_sdxl_attack.py"),
            "--original", str(s1),
            "--checkpoint", sdxl_checkpoint,
            "--prompt", prompt,
            "--output", str(s2),
            "--param-size", str(preset.sdxl_param_size),
            "--epsilon", str(preset.sdxl_epsilon),
            "--outer-iters", str(preset.sdxl_outer_iters),
            "--param-smooth-sigma", "5.0",
            "--perceptual-mask",
            "--seed", str(seed),
        ],
    )

    timings["stage3_sd15_topup"] = _run_stage(
        "stage 3/4: SD1.5 pixel top-up",
        [
            py, str(ML_SRC / "native_lowfreq_attack.py"),
            "--original", str(s2),
            "--checkpoint", sd15_checkpoint,
            "--output", str(s3),
            "--param-size", str(preset.sd15_param_size),
            "--epsilon", str(preset.sd15_topup_epsilon),
            "--steps", str(preset.sd15_topup_steps),
            "--eot-samples", "1",
            "--recon-weight", "0.5",
            "--param-smooth-sigma", "1.0",
            "--seed", str(seed),
        ],
    )

    timings["stage4_sdxl_topup"] = _run_stage(
        "stage 4/4: SDXL pixel top-up",
        [
            py, str(ML_SRC / "native_aspl_sdxl_attack.py"),
            "--original", str(s3),
            "--checkpoint", sdxl_checkpoint,
            "--prompt", prompt,
            "--output", output_path,
            "--param-size", str(preset.sdxl_param_size),
            "--epsilon", str(preset.sdxl_topup_epsilon),
            "--outer-iters", str(preset.sdxl_topup_outer_iters),
            "--param-smooth-sigma", "5.0",
            "--perceptual-mask",
            "--seed", str(seed),
        ],
    )

    total = sum(timings.values())
    print(f"[clean_protect] wrote {output_path} (4 stages, {total:.1f}s total)", flush=True)
    return {"output_path": output_path, "timings": timings, "stages_run": 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--sd15-checkpoint", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="CLEAN_FULL", choices=list(CLEAN_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    clean_protect(
        original_path=args.original,
        sd15_checkpoint=args.sd15_checkpoint,
        sdxl_checkpoint=args.sdxl_checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        seed=args.seed,
        work_dir=args.work_dir,
    )
