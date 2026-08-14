"""Hybrid protection stack -- composes every mechanism this project has
actually validated, in a way that is structurally incapable of the
budget competition that killed its two previous "combine things" attempts.

WHY THE PREVIOUS HYBRIDS FAILED (both are in [[lora-protection-research]]):

  - hybrid_attack.py summed three objectives (style + concept + denoise)
    into ONE loss optimizing ONE delta inside ONE epsilon ball. Result:
    -0.0054, the single worst mechanism this project ever measured. The
    objectives spent the same scarce budget pulling in different
    directions.
  - ensemble_attack_multiarch.py (joint SD1.5+SDXL) shared ONE delta
    across two architectures. SD1.5 passed; SDXL was crowded out to
    nothing -- and attacking SDXL *alone* later cleared the same
    validation bar, proving it was competition, not incapacity.

WHAT ACTUALLY WORKED, AND THE PRINCIPLE IT REVEALS: sequential chaining
(aspl_attack.py -> aspl_attack_sdxl_only.py, validated n=30: SD1.5
+0.1655, SDXL +0.0446). Each stage got its OWN full epsilon budget in its
OWN optimization pass, applied to the previous stage's output. Budgets
compose instead of competing. That is the design rule this module follows
everywhere: never sum objectives into one constrained delta; always give
each mechanism its own pass and its own budget.

WHAT THIS MODULE ADDS: the pixel-space chain's real problem is that the
budget it needs to be strong is large enough to be visibly ugly (see
[[strong-protection-visual-honesty]] -- it broke this project's own
"looks nearly identical" UI claim). The fix is not a smaller budget
(that just trades away protection) but a *second, orthogonal* budget:

  - LATENT space (aspl_attack_latent.py / aspl_attack_sdxl_only_latent.py,
    new 2026-08-08): perturbation is bounded inside the VAE's latent
    code, so every point in the ball decodes to a *plausible image*.
    IMPORTANT CORRECTION (2026-08-08, real-illustration pilot): the
    original n=1 starry_night.jpg check used latent_epsilon=0.3 and
    looked fine on that image, but running the *same* epsilon on a real
    user illustration produced a heavily distorted result -- the subject
    was barely recognizable, worse than the pixel-space chain this
    module exists to fix. Visual quality and protective strength trade
    off almost linearly inside this mechanism; a large latent budget is
    NOT a free lunch just because it decodes through the VAE. Lowering
    to latent_epsilon=0.15 restored visual quality close to the original
    (still not fully invisible -- some texture bleed near high-detail
    regions) but that alone cut the protective delta roughly in half
    versus epsilon=0.3 (n=1, SD1.5-only: epsilon=0.3 delta=+0.0725,
    epsilon=0.15 delta=+0.0442). latent_epsilon=0.15 is a deliberate,
    visually-driven compromise, not the strongest setting this mechanism
    can produce -- see the pixel top-up below for how the lost strength
    gets recovered.
  - PIXEL space, small epsilon: high-frequency detail that latent space
    structurally cannot represent (the VAE discards it -- that is what
    makes it a compressor). At epsilon=0.02 (this project's own
    L1_PREVIEW value, its documented near-invisible tier) it costs
    essentially nothing visually while adding signal in a band the latent
    stage never touched. CONFIRMED, not just argued (2026-08-08, n=1,
    SD1.5-only, same real illustration, latent_epsilon=0.15 base): adding
    this pixel top-up moved the delta from +0.0442 (latent-only) to
    +0.0943 -- the small pixel budget alone contributed +0.0501, MORE
    than the entire latent stage did on its own. This is the load-bearing
    evidence that the top-up is not "free but useless": it is the
    mechanism that recovers most of the strength given up by lowering
    latent_epsilon for visual quality.

These two cannot compete for the same budget: one is clamped in latent
coordinates, the other in RGB coordinates, in separate passes. That is
the structural guarantee, not a hope. What is NOT structurally
guaranteed -- and had to be measured, not assumed -- is that the latent
stage alone would be both strong and invisible at any single epsilon;
it isn't. The two-stage split exists precisely because no single latent
epsilon gave both, and epsilon=0.15+0.02 is the best n=1 balance found
so far.

STAGE ORDER IS LOAD-BEARING -- latent stages must run BEFORE pixel
stages. A latent stage ends in a VAE decode, which reconstructs pixels
from the latent code and therefore discards fine high-frequency detail
not represented in that code. Any pixel-space perturbation applied first
would be partially erased by that decode. Pixel-last means the
high-frequency signal survives untouched into the final output.

PERCEPTUAL MASK IS RE-USED HERE DESPITE FAILING ITS OWN PILOT, for a
specific reason: it was tested at epsilon=0.08, where the distortion is
structural (the model's own learned visual vocabulary bleeding through)
and no redistribution of a budget that large can hide it. Its actual
documented purpose (style_cloak.py's compute_perceptual_mask: hide *small*
noise by moving it into already-textured regions, measured +1.37dB PSNR
there) matches the small-epsilon pixel stages here exactly. Failing in one
regime is not evidence against the regime it was built for.

STATUS: this composition is DESIGNED, not validated. The latent_epsilon=
0.15 + pixel_epsilon=0.02 combination rests on two n=1, SD1.5-only
datapoints (latent-only vs latent+top-up, same real illustration,
2026-08-08) -- SDXL has not been re-checked at these values, only at the
original epsilon=0.3. Per this project's own rule ([[feedback-financial-
care]], every prior mechanism's history), it needs n=30 plus an
independent replication before any claim is made about it -- and before it
goes anywhere near production. Run experiments/hybrid_validation/ first.

Each stage runs as a SUBPROCESS, not an in-process import: each attack
script loads its own full pipeline into VRAM and expects to tear it down
on exit. A fresh process per stage is the simplest guarantee that no
state (or VRAM) leaks between stages -- the same reasoning
docker/strongprotect-serverless/handler.py already documents, and the
same OOM this project hit for real when run_dual_arch_n30.py held
training models resident while calling attack functions that load their
own.
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ML_SRC = Path(__file__).parent


@dataclass
class HybridPreset:
    """Per-stage budgets. Deliberately NOT one shared number -- the whole
    point is that each stage owns its own, in its own coordinate space."""

    latent_epsilon: float
    pixel_epsilon: float
    latent_preset_sd15: str
    latent_preset_sdxl: str
    pixel_preset_sd15: str
    pixel_preset_sdxl: str
    perceptual_mask: bool
    mask_low: float
    mask_high: float


HYBRID_PRESETS = {
    # Fast wiring/timing check before spending real GPU-hours on the full
    # thing -- same role CALIBRATION plays in every other module here.
    "CALIBRATION": HybridPreset(
        latent_epsilon=0.3,
        pixel_epsilon=0.02,
        latent_preset_sd15="L1_PREVIEW",
        latent_preset_sdxl="CALIBRATION",
        pixel_preset_sd15="L1_PREVIEW",
        pixel_preset_sdxl="CALIBRATION",
        perceptual_mask=True,
        mask_low=0.3,
        mask_high=1.7,
    ),
    # latent_epsilon=0.15 (lowered from an initial 0.3, 2026-08-08): 0.3
    # was calibrated on starry_night.jpg and looked fine there, but broke
    # down on a real user illustration -- heavy structural distortion, the
    # exact visual-honesty problem this module exists to fix. 0.15 is the
    # value that pilot found visually close to the original again. This
    # gives up real protective strength on its own (n=1, SD1.5-only:
    # epsilon=0.3 delta=+0.0725 vs epsilon=0.15 delta=+0.0442) -- the
    # pixel top-up below is what earns most of it back, not a bigger
    # latent budget.
    # pixel_epsilon=0.02: this project's own L1_PREVIEW epsilon, the tier
    # its docs already describe as visually near-invisible. Chosen because
    # it is the largest pixel budget this codebase has ever called
    # invisible -- the top-up should not be the thing that reintroduces
    # the visible damage the latent stages exist to avoid. CONFIRMED
    # (2026-08-08, n=1, SD1.5-only, same illustration, on top of
    # latent_epsilon=0.15): raised the delta from +0.0442 to +0.0943 --
    # +0.0501 from this stage alone, more than the latent stage
    # contributed by itself.
    "HYBRID_FULL": HybridPreset(
        latent_epsilon=0.15,
        pixel_epsilon=0.02,
        latent_preset_sd15="L3_ANTI_TRAIN",
        latent_preset_sdxl="SDXL_FULL",
        pixel_preset_sd15="L3_ANTI_TRAIN",
        pixel_preset_sdxl="SDXL_FULL",
        perceptual_mask=True,
        mask_low=0.3,
        mask_high=1.7,
    ),
}


def _run_stage(label: str, cmd: list[str]) -> float:
    print(f"=== {label} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        # Surface the child's real traceback -- CalledProcessError's str()
        # drops stdout/stderr, which turned a real failure in this
        # project's serverless handler into a bare "exit status 2" once.
        raise RuntimeError(
            f"{label} exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(f"  {label} done in {elapsed:.1f}s", flush=True)
    return elapsed


def hybrid_protect(
    original_path: str,
    sd15_checkpoint: str,
    sdxl_checkpoint: str,
    prompt: str,
    output_path: str,
    preset_name: str = "HYBRID_FULL",
    seed: int = 0,
    work_dir: str | None = None,
    skip_pixel_stages: bool = False,
    latent_epsilon_override: float | None = None,
    pixel_epsilon_override: float | None = None,
) -> dict:
    """Runs the four-stage stack and writes the final protected image.

    skip_pixel_stages=True runs only the latent half -- exactly the
    configuration the n=1 check measured, kept as a first-class option so
    a validation run can A/B the pixel top-up's marginal contribution
    instead of assuming it helps.

    latent_epsilon_override/pixel_epsilon_override (2026-08-08, user-
    facing "advanced options" upload feature): let an opted-in caller pick
    a different point on the same visibility-vs-strength tradeoff this
    module's own doc describes, instead of only ever running at
    HYBRID_FULL's fixed n=1-calibrated values. Same override-a-dataclass-
    field pattern aspl_attack.py/aspl_attack_sdxl_only.py already use for
    their own epsilon_override, not a new mechanism. Does NOT change what
    "HYBRID_FULL" itself means -- only what an explicit override call gets.
    """
    preset = HYBRID_PRESETS[preset_name]
    if latent_epsilon_override is not None or pixel_epsilon_override is not None:
        from dataclasses import replace as _dc_replace
        overrides = {}
        if latent_epsilon_override is not None:
            overrides["latent_epsilon"] = latent_epsilon_override
        if pixel_epsilon_override is not None:
            overrides["pixel_epsilon"] = pixel_epsilon_override
        preset = _dc_replace(preset, **overrides)
    work = Path(work_dir) if work_dir else Path(output_path).parent / "_hybrid_stages"
    work.mkdir(parents=True, exist_ok=True)

    s1 = work / "stage1_sd15_latent.png"
    s2 = work / "stage2_sdxl_latent.png"
    s3 = work / "stage3_sd15_pixel.png"

    py = sys.executable
    timings = {}

    # Stage 1-2: latent space, large budget. Heavy protection, plausible
    # output (the decoder can only emit points on the image manifold).
    timings["stage1_sd15_latent"] = _run_stage(
        "stage 1/4: SD1.5 latent",
        [
            py, str(ML_SRC / "aspl_attack_latent.py"),
            "--original", original_path,
            "--checkpoint", sd15_checkpoint,
            "--prompt", prompt,
            "--output", str(s1),
            "--preset", preset.latent_preset_sd15,
            "--seed", str(seed),
            "--latent-epsilon", str(preset.latent_epsilon),
        ],
    )

    timings["stage2_sdxl_latent"] = _run_stage(
        "stage 2/4: SDXL latent",
        [
            py, str(ML_SRC / "aspl_attack_sdxl_only_latent.py"),
            "--original", str(s1),
            "--sdxl-checkpoint", sdxl_checkpoint,
            "--prompt", prompt,
            "--output", str(s2),
            "--preset", preset.latent_preset_sdxl,
            "--seed", str(seed),
            "--latent-epsilon", str(preset.latent_epsilon),
        ],
    )

    if skip_pixel_stages:
        Path(s2).replace(output_path)
        print(f"[hybrid_protect] latent-only mode -- wrote {output_path}", flush=True)
        return {"output_path": output_path, "timings": timings, "stages_run": 2}

    # Stage 3-4: pixel space, small budget, perceptual-masked. Adds
    # high-frequency signal the VAE structurally discards, in a band the
    # latent stages could not have touched, at a size the mask can hide.
    pixel_mask_args = (
        ["--perceptual-mask", "--mask-low", str(preset.mask_low), "--mask-high", str(preset.mask_high)]
        if preset.perceptual_mask else []
    )

    timings["stage3_sd15_pixel"] = _run_stage(
        "stage 3/4: SD1.5 pixel (small epsilon)",
        [
            py, str(ML_SRC / "aspl_attack.py"),
            "--original", str(s2),
            "--checkpoint", sd15_checkpoint,
            "--prompt", prompt,
            "--output", str(s3),
            "--preset", preset.pixel_preset_sd15,
            "--seed", str(seed),
            "--epsilon-override", str(preset.pixel_epsilon),
            *pixel_mask_args,
        ],
    )

    timings["stage4_sdxl_pixel"] = _run_stage(
        "stage 4/4: SDXL pixel (small epsilon)",
        [
            py, str(ML_SRC / "aspl_attack_sdxl_only.py"),
            "--original", str(s3),
            "--sdxl-checkpoint", sdxl_checkpoint,
            "--prompt", prompt,
            "--output", output_path,
            "--preset", preset.pixel_preset_sdxl,
            "--seed", str(seed),
            "--epsilon-override", str(preset.pixel_epsilon),
            *pixel_mask_args,
        ],
    )

    total = sum(timings.values())
    print(f"[hybrid_protect] wrote {output_path} (4 stages, {total:.1f}s total)", flush=True)
    return {"output_path": output_path, "timings": timings, "stages_run": 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--sd15-checkpoint", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="HYBRID_FULL", choices=list(HYBRID_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--skip-pixel-stages", action="store_true")
    parser.add_argument("--latent-epsilon-override", type=float, default=None)
    parser.add_argument("--pixel-epsilon-override", type=float, default=None)
    args = parser.parse_args()

    hybrid_protect(
        original_path=args.original,
        sd15_checkpoint=args.sd15_checkpoint,
        sdxl_checkpoint=args.sdxl_checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        seed=args.seed,
        work_dir=args.work_dir,
        skip_pixel_stages=args.skip_pixel_stages,
        latent_epsilon_override=args.latent_epsilon_override,
        pixel_epsilon_override=args.pixel_epsilon_override,
    )
