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
    from remote_gpu import remote_cloak
else:
    from style_cloak import cloak

from Crypto.Hash import keccak


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
) -> dict:
    start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Matches apps/protection-svc/INTEGRATION.md's preset->params table:
    # L1_PREVIEW skips EOT (cheap tier, not worth the ~4x compute cost);
    # L2/L3 use it. `eot=None` (the CLI/default case) applies that rule;
    # an explicit True/False (server.py's request body) overrides it --
    # INTEGRATION.md's job contract documents `eot` as caller-settable.
    if eot is None:
        eot = preset_name != "L1_PREVIEW"

    # `size` defaults to cloak()'s own default (256) rather than something
    # larger -- NOT because larger is unsupported (cloak() has always taken
    # a `size` param), but because every measured number in this project
    # (preset epsilon values, EOT scale ranges, the 0.5x/0.25x robustness
    # breakpoints in ml-engine/README.md and rust-core/README.md) was
    # validated at 256x256 specifically. Passing --size 1024 runs outside
    # that validated envelope -- it will very likely still work mechanically
    # (more VGG19 compute, same algorithm) but the epsilon/robustness numbers
    # this project can currently stand behind don't automatically transfer.
    # See apps/protection-svc/INTEGRATION.md for the concrete consequence:
    # this default is *why* the public_preview_1280/2048 Delivery Gateway
    # tiers are unreachable unless a caller explicitly opts into a larger,
    # unvalidated size.
    cloaked_path = out / "cloaked.png"
    mode = "remote GPU" if USE_REMOTE_GPU else "local"
    print(f"[orchestrate] 1/4 style-cloak ({mode}) preset={preset_name} eot={eot} size={size} ...", flush=True)
    if USE_REMOTE_GPU:
        remote_cloak(
            original_path=input_path,
            style_target_path=style_target_path,
            output_path=str(cloaked_path),
            preset_name=preset_name,
            eot=eot,
            size=size,
        )
    else:
        cloak(
            original_path=input_path,
            style_target_path=style_target_path,
            output_path=str(cloaked_path),
            preset_name=preset_name,
            eot=eot,
            size=size,
        )

    watermarked_path = out / "watermarked.png"
    print("[orchestrate] 2/4 watermark ...", flush=True)
    run_rust_core(
        "embed",
        "--input", str(cloaked_path),
        "--output", str(watermarked_path),
        "--payload-hex", watermark_payload_hex,
        "--strength", "24.0",
    )

    variants_dir = out / "variants"
    print("[orchestrate] 3/4 resolution variants ...", flush=True)
    variants_output = run_rust_core("variants", "--input", str(watermarked_path), "--out-dir", str(variants_dir))
    variants = parse_variants_output(variants_output)

    print("[orchestrate] 4/4 perceptualHash + metadataHash ...", flush=True)
    perceptual_hash = compute_perceptual_hash_from_path(str(watermarked_path))

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
        "eotUsed": eot,
        "size": size,
        "sizeValidated": size == 256,  # see the `size` param's doc comment above
        "doNotTrain": not allow_ai_training,
        "watermarkPayloadHex": watermark_payload_hex,
        "processingTimeMs": round((time.time() - start) * 1000),
        "variants": variants,
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
    )

    print(json.dumps(result, indent=2))
