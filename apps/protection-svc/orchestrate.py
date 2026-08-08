"""End-to-end protection-svc pipeline orchestrator, wiring together the two
components that so far only existed as independent CLIs with their own
tests (ml-engine/README.md, rust-core/README.md). Matches the pipeline
diagram and job-result shape in apps/protection-svc/INTEGRATION.md:

    original image
         |
         v
    [ml-engine]  style_cloak (slow: seconds-to-minutes)
         |
         v
    [rust-core]  watermark
         |
         v
    [rust-core]  resolution variants (tagged Safe/Unknown/Unsafe)
         |
         v
    perceptualHash computed on the final published (watermarked) image --
    NOT ml-engine's raw cloak output (INTEGRATION.md is explicit about this)
         |
         v
    metadataHash (keccak256, must match blockchain-svc's computeContentHash)
         |
         v
    result matching GET /protect/{jobId}'s shape

Now wrapped in an HTTP job API too -- see server.py, which imports and
calls protect() directly (this module is the shared implementation; server.py
is a thin async/job-status layer on top, not a reimplementation).

Usage (direct CLI, still useful for local iteration without running the server):
    <ml-engine venv python> orchestrate.py --input path/to/art.jpg \\
        --out-dir out/pipeline_run --preset L3_ANTI_TRAIN --title "My Artwork"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).parent / "ml-engine"

# rust-core's binary name/location differs by platform and build profile --
# .exe + debug/ on the Windows dev machine (fast iteration), no extension +
# release/ on the Pi deployment (built once with `cargo build --release`,
# see rust-core/README.md). RUST_CORE_BIN env var overrides both if set.
_rust_core_candidates = [
    Path(__file__).parent / "rust-core" / "target" / "release" / "rust-core",
    Path(__file__).parent / "rust-core" / "target" / "release" / "rust-core.exe",
    Path(__file__).parent / "rust-core" / "target" / "debug" / "rust-core.exe",
    Path(__file__).parent / "rust-core" / "target" / "debug" / "rust-core",
]
RUST_CORE_BIN = Path(os.environ["RUST_CORE_BIN"]) if "RUST_CORE_BIN" in os.environ else next(
    (p for p in _rust_core_candidates if p.exists()), _rust_core_candidates[0]
)

# Absolute (not rust-core's own relative default) so the signing identity's
# location doesn't depend on this process's working directory matching
# rust-core's own assumption about it -- see c2pa_manifest.rs's
# LocalSigner::load_or_generate doc for why this file needs to actually
# persist across calls to mean anything.
C2PA_SIGNING_KEY_PATH = Path(__file__).parent / "rust-core" / "keys" / "c2pa_signing_key.der"

sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from style_cloak import PRESETS  # noqa: E402
from perceptual_hash import compute_perceptual_hash_from_path  # noqa: E402

# USE_REMOTE_GPU=1: delegate the cloak step to a GPU PC over SSH instead of
# running it in-process (see remote_gpu.py's module doc). Used when this
# orchestrator runs somewhere with no usable GPU -- the Pi deployment, in
# practice. Deliberately does NOT import style_cloak.cloak (which needs
# torch loaded) when in remote mode, though torch happens to be installed
# on the Pi anyway at the time this was written -- the point is not
# depending on that being true.
USE_REMOTE_GPU = os.environ.get("USE_REMOTE_GPU") == "1"

if USE_REMOTE_GPU:
    from remote_gpu import remote_cloak, remote_compute_metrics, remote_upscale
else:
    from style_cloak import cloak

# strong_protection (PHASE4_SCOPING.md §6) always dispatches to a
# dedicated A40-class worker regardless of USE_REMOTE_GPU -- it needs more
# VRAM than any local machine this project runs on has, GPU PC included --
# so this import is unconditional; serverless_dual_arch_cloak() itself
# only touches RUNPOD_API_KEY/RUNPOD_STRONGPROTECT_ENDPOINT_ID (raises if
# unset) when a caller actually opts in (see protect()'s strong_protection
# branch below), not at import time. Uses the sequential SD1.5-then-SDXL
# chain (each architecture attacked independently), not a joint attack --
# PHASE4_SCOPING.md §6's follow-up finding was that the joint design
# suppresses SDXL's own effect to nothing, while attacking it separately
# clears the same validation bar SD1.5 already had.
#
# RunPod Serverless (serverless_dual_arch_cloak), not the SSH-to-an-
# always-on-pod path (remote_dual_arch_cloak, still in remote_gpu.py for
# manual/debugging use against a hand-started pod) -- PHASE4_SCOPING.md
# §6's 2026-08-07 empirical comparison found essentially identical
# execution time (685s vs 719s, one A40 job) but Serverless scales to
# zero automatically, removing the "someone has to remember an idle pod
# is running" failure mode a real production deployment can't tolerate
# (this session found and deleted a pod that had sat idle for 4.5 hours).
from remote_gpu import serverless_dual_arch_cloak

from Crypto.Hash import keccak  # noqa: E402

# Real, measured result on the GPU PC comparing this project's own two
# strategies for a real high-resolution upload (2835x4289): processing at
# a fixed size=256 then EDSR-upscaling back up, vs. processing directly
# closer to the real resolution -- the direct approach won on BOTH axes at
# once (not a trade-off): PSNR 27.74dB -> 32.66dB (+4.9dB, crosses into
# "visually near-identical" territory) AND styleDriftScore 0.084 -> 0.142
# (+69% -- *stronger* protection, not weaker). Downsampling to 256 first
# throws away real detail the optimizer needs to work against, and then
# EDSR's job (produce natural-looking output) partially smooths the
# adversarial signal right back out on the way up. 1024 matches this
# project's own prior "1024px re-validation" precedent (see
# ml-engine/README.md) for a resolution genuinely exercised before, not a
# new unvalidated guess.
MAX_PROCESSING_SIZE = 1024


def choose_processing_size(image_path: str, max_size: int = MAX_PROCESSING_SIZE) -> int:
    """The real image's own long-edge resolution, capped at max_size --
    replaces always forcing size=256 regardless of what was actually
    uploaded (see the note above this function for why that was a real,
    measured problem, not just a hunch). A modest upload (long edge below
    the cap) processes at its own native size, needing no upscale step at
    all afterward; only uploads bigger than the cap still go through
    cloak() at a smaller size and get restored via upscale.py.
    """
    from PIL import Image as _Image

    width, height = _Image.open(image_path).size
    return min(max(width, height), max_size)


def choose_eot_samples(size: int) -> int:
    """Hit this for real, live, while measuring the resolution fix above:
    size=1024 at the usual eot_samples=2 pushed this project's GPU PC to
    ~96% VRAM and got dramatically slower than linear scaling would
    predict -- 2+ hours without finishing (killed), the *exact* VRAM-
    pressure-induced slowdown ml-engine/README.md's "1024px re-validation"
    section already documented once before at eot_samples=3. Re-ran at
    eot_samples=1 and it finished in ~2 minutes with no quality regression
    that mattered (still a clear win over the old size=256 pipeline on
    both PSNR and styleDriftScore -- see MAX_PROCESSING_SIZE's comment).
    Only sizes still inside the originally-validated 256px envelope keep
    the fuller eot_samples=2 default; everything above it -- which is
    every real upload big enough to actually need this size fix in the
    first place -- drops to 1, matching the project's own established
    fix for this exact configuration rather than re-discovering it badly
    in production on a real user's upload.
    """
    return 2 if size <= 256 else 1


def choose_use_amp(size: int) -> bool:
    """Real GPU measurement (post-fixing a real fp16 overflow bug in
    model.py's Gram-matrix computation -- large reductions need forced
    fp32 accumulation even under autocast): at size=1024, mixed precision
    matched fp32's styleDriftScore/PSNR almost exactly (0.1608 vs 0.1606,
    29.38dB vs 29.41dB) while running 2.2x faster (90.5s vs 159.3s) and
    using 29% less peak VRAM (3042MB vs 4294MB) -- a clean win at the
    resolution this project actually runs at.

    Investigated specifically to see whether the VRAM headroom would let
    MAX_PROCESSING_SIZE go higher than 1024 -- it doesn't: size=1536 with
    AMP still hit the same VRAM-pressure/allocator-thrashing wall
    choose_eot_samples's doc already documents once (this time confirmed
    stuck for real: 55+ minutes of accumulated CPU time with no
    progress, killed). So this stays a speed/headroom win at the existing
    1024 cap, not a lever for raising it further on this project's 8GB
    GPU PC.

    Only sizes above the originally fp32-validated 256px envelope get
    AMP -- same reasoning as choose_eot_samples: small enough jobs
    finish quickly in fp32 anyway, no reason to introduce fp16 into a
    path that was never measured with it.
    """
    return size > 256


def choose_perceptual_mask(preset_name: str) -> bool:
    """Real GPU measurement on top of the now-fixed native-resolution
    pipeline (size=1024, eot_samples=1): redistributing the epsilon clamp
    toward already-textured regions (JND-style) instead of a uniform clamp
    is a real quality win at a cost well inside the "same or negligible
    difference" bar this project holds protection strength to, for both
    presets it's been measured against:
      L3_ANTI_TRAIN: +1.37dB PSNR (27.53 -> 28.90), -1.9% styleDriftScore
                     (0.1645 -> 0.1614)
      L2_PORTFOLIO:  +1.73dB PSNR (31.15 -> 32.88), -2.6% styleDriftScore
                     (0.1567 -> 0.1526)
    L1_PREVIEW has not been measured -- it's already the cheap/low-epsilon
    tier (no EOT either), and the noise-visibility complaint this was
    responding to was never about L1, so it stays off there rather than
    assumed to generalize.
    """
    return preset_name in ("L2_PORTFOLIO", "L3_ANTI_TRAIN")


def compute_metadata_hash(metadata: dict) -> str:
    """keccak256 of canonical JSON. Must byte-match blockchain-svc's
    computeContentHash expectations (apps/blockchain-svc/src/hash.ts) --
    cross-checked against ethers.js's keccak256("test") during development
    to confirm pycryptodome's Keccak matches (NOT the same as NIST SHA3-256,
    a common mix-up). Stable key order (sort_keys) is what makes this
    deterministic across calls.
    """
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    h = keccak.new(digest_bits=256)
    h.update(canonical.encode("utf-8"))
    return "0x" + h.hexdigest()


def run_rust_core(*args: str) -> str:
    if not RUST_CORE_BIN.exists():
        raise FileNotFoundError(
            f"rust-core binary not found at {RUST_CORE_BIN} -- run `cargo build` in rust-core/ first"
        )
    result = subprocess.run([str(RUST_CORE_BIN), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rust-core {args[0]} failed:\n{result.stderr}")
    return result.stdout


def parse_variants_output(output: str) -> list[dict]:
    """Parses rust-core's `variants` text-table output into structured
    records. rust-core doesn't emit JSON yet -- text parsing is a stopgap
    specific to this being a script, not a real IPC boundary; if this
    orchestration graduates into an actual service, rust-core should grow a
    `--json` output mode instead of this.
    """
    variants = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        name, width, height, scale = parts[0], parts[1], parts[2], parts[3]
        if not (width.isdigit() and height.isdigit() and scale.rstrip("x").replace(".", "", 1).isdigit()):
            continue  # not a data row (header, or rust-core's summary line)
        status = " ".join(parts[4:])
        variants.append(
            {
                "name": name,
                "width": int(width),
                "height": int(height),
                "scaleVsSource": float(scale.rstrip("x")),
                "protectionStatus": status,
            }
        )
    return variants


def _maybe_auto_select_style_target(input_path: str, style_target_path: str, size: int) -> str:
    """Overrides the caller-given style_target_path with the candidate from
    STYLE_TARGET_CANDIDATES_DIR that ai-engine's LoRA validation experiment
    found gives the biggest real degradation effect (ml-engine/src/
    select_style_target.py's module doc has the full finding: pre-cloak
    Gram-matrix dissimilarity between original and target correlates with
    real CLIP-measured effect, r=-0.516 in a controlled follow-up).

    Off by default (env var unset) rather than silently changing every
    upload's behavior -- this needs a real curated candidate pool to be
    worth turning on, and this repo doesn't ship one (ai-engine's pool is
    10 famous paintings assembled for that experiment, not a production
    asset). Also a no-op under USE_REMOTE_GPU: selection needs a local
    torch/VGG19 forward pass per candidate, which is exactly what remote-GPU
    mode exists to avoid needing on this machine -- extending remote_gpu.py
    to run selection remotely too is future work, not silently done wrong
    here.
    """
    candidates_dir = os.environ.get("STYLE_TARGET_CANDIDATES_DIR")
    if not candidates_dir:
        return style_target_path
    if USE_REMOTE_GPU:
        print(
            "[orchestrate] STYLE_TARGET_CANDIDATES_DIR is set but USE_REMOTE_GPU=1 -- "
            "auto-selection needs a local torch pass, skipping and using the given style_target_path",
            flush=True,
        )
        return style_target_path

    candidates = [
        str(p) for p in Path(candidates_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file()
    ]
    if not candidates:
        print(f"[orchestrate] STYLE_TARGET_CANDIDATES_DIR={candidates_dir!r} has no images, using given style_target_path", flush=True)
        return style_target_path

    from select_style_target import select_most_dissimilar_target

    selected_path, similarity = select_most_dissimilar_target(input_path, candidates, size=size)
    print(f"[orchestrate] auto-selected style target {selected_path} (pre-cloak similarity={similarity:.4f})", flush=True)
    return selected_path


def protect(
    input_path: str,
    out_dir: str,
    preset_name: str,
    style_target_path: str,
    title: str,
    creator_id: str,
    allow_ai_training: bool,
    watermark_payload_hex: str,
    size: int = 256,
    eot: bool | None = None,
    concept_misalign_target_path: str | None = None,
    strong_protection: bool = False,
) -> dict:
    start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    style_target_path = _maybe_auto_select_style_target(input_path, style_target_path, size)

    # Feed/gallery thumbnail, generated from the real ORIGINAL image (not
    # the protected one) at rust-core variants.rs's FEED_THUMBNAIL_MAX_
    # DIMENSION -- see that module's own doc for why sourcing from the
    # original is safe here (information loss, not the cloak, is the
    # defense). Runs first and unconditionally: doesn't depend on which
    # cloak path below succeeds, or even on the upload finishing protection
    # at all, so a slow/failed cloak shouldn't block this. Best-effort like
    # every other enrichment step in this function -- a missing thumbnail
    # variant just means the feed falls back to a larger one, not a failed
    # upload.
    feed_thumbnail_path = out / "feed_thumbnail.png"
    feed_thumbnail_ready = False
    print("[orchestrate] 0/4 feed thumbnail (from original, pre-protection) ...", flush=True)
    try:
        run_rust_core("feed-thumbnail", "--input", input_path, "--output", str(feed_thumbnail_path))
        feed_thumbnail_ready = True
    except Exception as exc:  # noqa: BLE001 -- a real upload succeeding matters more than this
        print(f"[orchestrate] feed thumbnail generation failed, continuing without it: {exc}", flush=True)

    # Matches apps/protection-svc/INTEGRATION.md's preset->params table:
    # L1_PREVIEW skips EOT (cheap tier, not worth the ~4x compute cost);
    # L2/L3 use it. `eot=None` (the CLI/default case) applies that rule;
    # an explicit True/False (server.py's request body) overrides it --
    # INTEGRATION.md's job contract documents `eot` as caller-settable.
    if eot is None:
        eot = preset_name != "L1_PREVIEW"

    # `size` itself is now caller-controlled (server.py's ProtectRequest.size
    # defaults to choose_processing_size(), not a fixed 256 -- see that
    # function's doc for the real GPU measurement that changed this: a
    # fixed 256 measurably lost on both perceptual quality AND protection
    # strength compared to processing closer to the real upload's own
    # resolution). This function itself keeps size: int = 256 as its own
    # parameter default only for direct callers (the CLI below, tests,
    # ml-engine's manual experiment scripts) that don't go through
    # server.py's request layer at all.
    cloaked_path = out / "cloaked.png"

    # strong_protection (PHASE4_SCOPING.md §6, opt-in only, same shape as
    # concept_misalign_target_path below): replaces style-cloak entirely
    # rather than stacking on top of it -- this project's own hybrid_attack
    # experiment already found combining multiple attack objectives makes
    # things worse, not better. Runs hybrid_protect.py's four-stage
    # latent-then-pixel composition (serverless_dual_arch_cloak, via
    # RunPod Serverless -- see this module's own import-time comment for
    # why Serverless over the SSH-to-a-pod path). CAVEAT (2026-08-08):
    # unlike the two-stage pixel-only chain it replaced (which cleared
    # this project's usual n=30-plus-replication bar), this composition
    # is validated at n=1, SD1.5-only -- see hybrid_protect.py's own
    # module doc. Wired in ahead of full validation at the user's
    # explicit, informed decision. Falls back to style_cloak on any
    # failure (endpoint unreachable, RUNPOD_API_KEY unset, job failed,
    # etc.) rather than publishing an unprotected image -- a real upload
    # succeeding with the proven mechanism beats a failed upload.
    used_strong_protection = False
    if strong_protection:
        print(
            "[orchestrate] 1/4 style-cloak (hybrid latent+pixel, SD1.5+SDXL, "
            "RunPod Serverless, see hybrid_protect.py) ...",
            flush=True,
        )
        try:
            serverless_dual_arch_cloak(
                original_path=input_path,
                output_path=str(cloaked_path),
                prompt=title,
            )
            used_strong_protection = True
        except Exception as exc:  # noqa: BLE001 -- fall back to the proven mechanism rather than publish unprotected
            print(f"[orchestrate] strong_protection requested but dual-arch cloak failed ({exc}) -- falling back to style_cloak", flush=True)

    if not used_strong_protection:
        mode = "remote GPU" if USE_REMOTE_GPU else "local"
        eot_samples = choose_eot_samples(size)
        perceptual_mask = choose_perceptual_mask(preset_name)
        use_amp = choose_use_amp(size)
        print(f"[orchestrate] 1/4 style-cloak ({mode}) preset={preset_name} eot={eot} size={size} eot_samples={eot_samples} perceptual_mask={perceptual_mask} use_amp={use_amp} ...", flush=True)
        if USE_REMOTE_GPU:
            remote_cloak(
                original_path=input_path,
                style_target_path=style_target_path,
                output_path=str(cloaked_path),
                preset_name=preset_name,
                eot=eot,
                size=size,
                eot_samples=eot_samples,
                perceptual_mask=perceptual_mask,
                use_amp=use_amp,
            )
        else:
            cloak(
                original_path=input_path,
                style_target_path=style_target_path,
                output_path=str(cloaked_path),
                preset_name=preset_name,
                eot=eot,
                size=size,
                eot_samples=eot_samples,
                perceptual_mask=perceptual_mask,
                use_amp=use_amp,
            )

    # Real, per-upload protection metrics (asked for: something non-technical
    # users can be shown, not just "trust us it worked"). Measured right
    # after style-cloak, before any concept-misalign step further perturbs
    # cloaked_path -- this is specifically the style-drift number
    # style_cloak.py's own optimization target maps to, not a mix of two
    # different mechanisms' effects. Needs a VGG19 forward pass, so under
    # USE_REMOTE_GPU this delegates to the GPU PC too (remote_compute_metrics
    # reuses the exact files remote_cloak() already uploaded/downloaded
    # there -- no extra transfer) instead of the old behavior of silently
    # skipping this measurement entirely on a Pi with no local torch worth
    # relying on, which left styleDriftScore/perceptualPsnrDb/
    # styleSimilarityToOriginal permanently null for every real upload on
    # that deployment. Non-fatal either way -- a real upload succeeding is
    # more important than this nice-to-have number, so any failure here
    # (GPU PC unreachable, missing torch, OOM, whatever) is logged and the
    # pipeline continues without it.
    protection_metrics: dict = {}
    if used_strong_protection:
        # style_cloak's VGG19-Gram-matrix style-drift-vs-target metric
        # doesn't apply here -- aspl_attack/aspl_attack_sdxl_only have no
        # style_target_path input and optimize a genuinely different
        # (denoising-loss-based) objective, so this metric would compare
        # against a target the cloak step never actually used. Left empty
        # rather than computed-and-mislabeled; a real CLIP-similarity-based
        # metric for this mechanism is real future work, not a quick swap.
        print("[orchestrate] 1c/4 skipping style-drift metric (not meaningful for dual-arch strong_protection)", flush=True)
    else:
        try:
            print("[orchestrate] 1c/4 measuring protection effect (style drift vs. target, perceptual similarity to original) ...", flush=True)
            if USE_REMOTE_GPU:
                protection_metrics = remote_compute_metrics(
                    original_path=input_path,
                    style_target_path=style_target_path,
                    size=size,
                )
            else:
                from evaluate import compute_protection_metrics

                protection_metrics = compute_protection_metrics(
                    original_path=input_path,
                    cloaked_path=str(cloaked_path),
                    style_target_path=style_target_path,
                    size=size,
                )
        except Exception as exc:  # noqa: BLE001 -- a missing metric shouldn't fail a real upload
            print(f"[orchestrate] protection-metrics measurement failed, continuing without it: {exc}", flush=True)

    # Concept Misalignment Layer (PHASE4_SCOPING.md §1, PROJECT_DESIGN.md
    # §3-3 layer [3]) -- opt-in only, off unless a caller explicitly passes
    # concept_misalign_target_path, for the same reason it's not on by
    # default in any preset: concept_misalign.py's own module doc is
    # explicit that PHASE4_SCOPING.md §1's recommended LoRA-training
    # validation experiment has not been run against it, so this is an
    # unvalidated mechanism, not a proven protection effect -- a caller
    # opting in is accepting that, not getting a silently-upgraded default.
    # Also a no-op under USE_REMOTE_GPU, same reasoning as
    # _maybe_auto_select_style_target above: needs a local torch/CLIP
    # forward pass, and extending remote_gpu.py to cover this too is
    # future work, not silently done wrong here.
    if concept_misalign_target_path:
        if USE_REMOTE_GPU:
            print(
                "[orchestrate] concept_misalign_target_path is set but USE_REMOTE_GPU=1 -- "
                "concept misalignment needs a local CLIP pass, skipping",
                flush=True,
            )
        else:
            from concept_misalign import CONCEPT_PRESETS, misalign

            misalign_preset = preset_name if preset_name in CONCEPT_PRESETS else "L3_ANTI_TRAIN"
            print(
                f"[orchestrate] 1b/4 concept-misalign (local, EXPERIMENTAL/unvalidated -- "
                f"see concept_misalign.py's module doc) preset={misalign_preset} ...",
                flush=True,
            )
            misalign(
                original_path=str(cloaked_path),
                concept_target_path=concept_misalign_target_path,
                output_path=str(cloaked_path),
                preset_name=misalign_preset,
                eot=eot,
                size=size,
            )

    # Restore the real resolution/aspect ratio. cloak() (and, if it ran,
    # concept-misalign) only ever process a letterboxed size x size square
    # (see style_cloak.py's letterbox_resize doc) -- without undoing that,
    # every published image would stay locked at that small square
    # regardless of what was actually uploaded, which is exactly the
    # reported problem (asset-service's larger delivery-gateway variants,
    # public_preview_1280/2048, structurally can never be generated from a
    # source that's never bigger than `size` on its long edge). Crop the
    # letterbox padding back out using the *original* upload's real
    # dimensions, then use a real super-resolution model (not a naive
    # resize) to restore something close to that original resolution.
    try:
        from PIL import Image as _Image
        from style_cloak import letterbox_content_box

        orig_w, orig_h = _Image.open(input_path).size
        box = letterbox_content_box(orig_w, orig_h, size)
        # .convert("RGB") forces PIL to eagerly load pixel data now, before
        # the save() below opens (and truncates) this same path for writing.
        # Without it, Image.open() is lazy and .save(cloaked_path) truncates
        # the file before .crop() ever reads from it -- hit for real on a
        # production upload: a valid-looking PNG header but a truncated body
        # (rust-core's embed step failed with IoError(UnexpectedEof)).
        cropped = _Image.open(cloaked_path).convert("RGB").crop(box)
        cropped.save(cloaked_path)

        print(f"[orchestrate] 1d/4 restoring resolution to {orig_w}x{orig_h} via super-resolution ...", flush=True)
        if USE_REMOTE_GPU:
            # Loading torch + the EDSR CNN and running it locally on a real
            # near-native-resolution image (the resolution fix processes up
            # to 1024px now, vs. the old fixed 256px) OOM-killed protection-
            # svc's whole process for real in production on the Pi (~7.1GB
            # resident on an ~8GB machine, no GPU) -- taking down every
            # in-flight job, not just the one that triggered it. Delegate to
            # the GPU PC instead, same reasoning as remote_cloak.
            remote_upscale(str(cloaked_path), str(cloaked_path), orig_w, orig_h)
        else:
            from upscale import upscale_to_size

            used_sr = upscale_to_size(str(cloaked_path), str(cloaked_path), orig_w, orig_h)
            if not used_sr:
                print("[orchestrate] (SR model unavailable or unnecessary -- used a plain resize instead)", flush=True)
    except Exception as exc:  # noqa: BLE001 -- a small-but-real image beats a crashed upload
        print(f"[orchestrate] resolution restoration failed, publishing at the smaller processing size instead: {exc}", flush=True)

    watermarked_path = out / "watermarked.png"
    print("[orchestrate] 2/4 watermark ...", flush=True)
    run_rust_core(
        "embed",
        "--input", str(cloaked_path),
        "--output", str(watermarked_path),
        "--payload-hex", watermark_payload_hex,
        "--strength", "24.0",
    )

    # Computed here (not at step 4/4's original spot) so the real value can
    # go into the C2PA assertion below instead of an empty placeholder --
    # deterministic from watermarked_path either way, so computing it once
    # here and reusing it later isn't a behavior change, just an ordering one.
    perceptual_hash = compute_perceptual_hash_from_path(str(watermarked_path))

    # C2PA manifest embedding (rust-core/src/c2pa_manifest.rs) -- previously
    # built, tested, and reachable only via the standalone `c2pa-sign` CLI
    # subcommand, never actually called from this pipeline (see
    # rust-core/README.md's C2PA section for that history). Real content
    # available at this point: doNotTrain, title, creatorId, and the
    # watermarked image's own perceptualHash. NOT blockchain-svc's on-chain
    # contentHash/txHash -- despite what an earlier draft of this module's
    # doc comment implied, that data structurally can't exist yet here:
    # on-chain registration is a separate, later step asset-service's own
    # job state machine triggers after protect() already returned, not
    # something protect() itself has any access to.
    #
    # Best-effort like protection_metrics/concept-misalign above: a missing
    # C2PA manifest shouldn't fail a real upload. Signs in place
    # (watermarked_path -> watermarked_path) -- safe because rust-core's
    # c2pa-sign reads the whole input into memory before writing any output
    # bytes, so there's no read/write race with itself.
    c2pa_applied = False
    print("[orchestrate] 2b/4 C2PA manifest ...", flush=True)
    try:
        ownership = {
            "doNotTrain": not allow_ai_training,
            "title": title,
            "creatorId": creator_id,
            "perceptualHash": perceptual_hash,
        }
        run_rust_core(
            "c2pa-sign",
            "--input", str(watermarked_path),
            "--output", str(watermarked_path),
            "--format", "png",
            "--title", title,
            "--ownership-json", json.dumps(ownership),
            "--signing-key-path", str(C2PA_SIGNING_KEY_PATH),
        )
        c2pa_applied = True
    except Exception as exc:  # noqa: BLE001 -- a real upload succeeding matters more than this enrichment
        print(f"[orchestrate] C2PA signing failed, continuing without it: {exc}", flush=True)

    variants_dir = out / "variants"
    print("[orchestrate] 3/4 resolution variants ...", flush=True)
    variants_output = run_rust_core("variants", "--input", str(watermarked_path), "--out-dir", str(variants_dir))
    variants = parse_variants_output(variants_output)
    # rust-core's `variants` subcommand already wrote each of these to its
    # own file under variants_dir (main.rs: `result.image.save(&out_path)`)
    # -- report that real path so callers (asset-service's orchestration.ts)
    # can store a distinct storageUri per variant instead of every tier
    # pointing at the same full-size watermarked_path (a previously-known
    # stub, see orchestration.ts's old comment "rust-core variants aren't
    # uploaded anywhere separate yet").
    for v in variants:
        v["path"] = str(variants_dir / f"{v['name']}.png")

    if feed_thumbnail_ready:
        from PIL import Image as _Image3

        fw, fh = _Image3.open(feed_thumbnail_path).size
        variants.append(
            {
                "name": "feed_thumbnail_original",
                "width": fw,
                "height": fh,
                # Not comparable to DELIVERY_VARIANTS' scale-vs-protected-
                # source semantics (this variant is sourced from the
                # original, not watermarked_path) -- 0.0 is a placeholder,
                # not a real measurement.
                "scaleVsSource": 0.0,
                "protectionStatus": "INFO_LOSS_ONLY (from original, unvalidated resolution floor -- see rust-core variants.rs)",
                "path": str(feed_thumbnail_path),
            }
        )

    print("[orchestrate] 4/4 metadataHash ...", flush=True)

    metadata = {
        "title": title,
        "creatorId": creator_id,
        "allowAiTraining": allow_ai_training,
    }
    metadata_hash = compute_metadata_hash(metadata)

    result = {
        "status": "completed",
        "protectedImageUri": str(watermarked_path),
        "perceptualHash": perceptual_hash,
        "metadataHash": metadata_hash,
        "appliedPreset": preset_name,
        # Distinct from "strong_protection was requested" -- serverless_dual_arch_cloak()
        # can fail (endpoint unreachable, job failed) and fall back to
        # style_cloak silently rather than fail the whole upload (see
        # this function's own strong_protection branch above). Callers
        # (asset-service, gating the Test Lab's real-LoRA-effect-test
        # button) need to know what actually ran, not just what was asked
        # for -- only a strong_protection artwork has cleared this
        # project's own n=30-plus-replication real-effect validation bar.
        "usedStrongProtection": used_strong_protection,
        "eotUsed": eot,
        "size": size,
        "sizeValidated": size == 256,  # see the `size` param's doc comment above
        "doNotTrain": not allow_ai_training,
        "watermarkPayloadHex": watermark_payload_hex,
        "conceptMisalignApplied": bool(concept_misalign_target_path) and not USE_REMOTE_GPU,
        "c2paApplied": c2pa_applied,
        "processingTimeMs": round((time.time() - start) * 1000),
        "variants": variants,
        # None (not 0) when compute_protection_metrics() above didn't run
        # or failed -- a real "we didn't measure this" is not the same
        # value as a real measured drift of zero, and callers (asset-
        # service, the web UI) need to tell those apart rather than
        # silently treating a missing measurement as "no protection".
        "styleDriftScore": protection_metrics.get("styleDriftScore"),
        "styleSimilarityToOriginal": protection_metrics.get("styleSimilarityToOriginal"),
        "perceptualPsnrDb": protection_metrics.get("perceptualPsnrDb"),
    }

    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preset", choices=list(PRESETS), default="L3_ANTI_TRAIN")
    parser.add_argument("--style-target", default=str(ML_ENGINE_DIR / "out" / "style_target.png"))
    parser.add_argument("--title", default="Untitled artwork")
    parser.add_argument("--creator-id", default="creator_unknown")
    parser.add_argument("--allow-ai-training", action="store_true")
    parser.add_argument("--watermark-payload-hex", default="deadbeefcafef00d")
    parser.add_argument(
        "--concept-misalign-target",
        default=None,
        help="EXPERIMENTAL/opt-in (PHASE4_SCOPING.md §1, unvalidated -- see concept_misalign.py's "
        "module doc): path to a decoy-concept image. If set, runs concept_misalign.py on the "
        "style-cloaked output before watermarking. Omit (default) to skip entirely.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="cloak processing resolution (square). 256 is the only value this project has validated "
        "presets/EOT/robustness numbers at -- see the `size` param's doc comment in protect().",
    )
    args = parser.parse_args()

    result = protect(
        input_path=args.input,
        out_dir=args.out_dir,
        preset_name=args.preset,
        style_target_path=args.style_target,
        title=args.title,
        creator_id=args.creator_id,
        allow_ai_training=args.allow_ai_training,
        watermark_payload_hex=args.watermark_payload_hex,
        size=args.size,
        concept_misalign_target_path=args.concept_misalign_target,
    )

    print(json.dumps(result, indent=2))
