"""n=1 real-training check for hybrid_protect.py, with a built-in A/B on
the one thing that module's design actually leaves open.

hybrid_protect.py's structural argument (latent and pixel budgets live in
different coordinate spaces, so they cannot compete) is sound as far as it
goes, but it does not by itself prove the pixel top-up *contributes*
anything measurable on top of the latent stages. It could be free-but-
useless as easily as free-and-helpful. So this runs BOTH configurations on
the same image with the same seed:

  A. latent-only  (stages 1-2, exactly what the n=1 latent check measured)
  B. full hybrid  (stages 1-4, latent + small-epsilon masked pixel top-up)

and reports both deltas per architecture. Interpretation set in advance,
so the result cannot be read selectively after the fact:

  - B > A on both architectures  -> the top-up earns its ~2x runtime cost
  - B ~= A                       -> drop stages 3-4, ship latent-only
  - B < A                        -> the stages interfere after all; the
                                    "different spaces cannot compete"
                                    argument is wrong somewhere and needs
                                    re-examining before anything ships

n=1 decides nothing on its own -- this is the cheap gate that says whether
the full n=30 (plus independent replication, same as every validated
mechanism in this project) is worth funding, and which configuration that
n=30 should even test.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, "/workspace/dontai-protection-svc/ml-engine/src")
sys.path.insert(0, "/workspace")
from aspl_attack import ASPLAttacker  # noqa: E402
from ensemble_attack_multiarch import SDXLBranch  # noqa: E402
from hybrid_protect import hybrid_protect  # noqa: E402
from run_dual_arch_n30 import generate_and_score_sd15, train_lora_sd15  # noqa: E402
from run_n1_check import generate_and_score as generate_and_score_sdxl  # noqa: E402
from run_n1_check import train_lora as train_lora_sdxl  # noqa: E402

SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
TRAIN_STEPS = 200


def score_condition(
    label: str,
    protected_path: str,
    original_path: str,
    prompt: str,
    seed: int,
    sd15_attacker,
    sd15_text_embeddings,
    sdxl_branch,
    true_image,
    clip_model,
    clip_processor,
) -> dict:
    """Train+score both architectures against one protected image.

    Baselines are recomputed per condition rather than shared: they are
    cheap relative to the attack stages, and reusing a baseline across
    conditions would quietly couple the two numbers being compared.
    """
    print(f"=== [{label}] SD1.5: baseline vs protected ===", flush=True)
    t0 = time.time()
    lora_b15 = train_lora_sd15(sd15_attacker, original_path, sd15_text_embeddings, seed, TRAIN_STEPS)
    b15 = generate_and_score_sd15(sd15_attacker, lora_b15, sd15_text_embeddings, true_image, clip_model, clip_processor)
    lora_b15.unload()
    lora_a15 = train_lora_sd15(sd15_attacker, protected_path, sd15_text_embeddings, seed, TRAIN_STEPS)
    a15 = generate_and_score_sd15(sd15_attacker, lora_a15, sd15_text_embeddings, true_image, clip_model, clip_processor)
    lora_a15.unload()
    print(f"  [{label}] sd15_delta={b15 - a15:+.4f} (done in {time.time() - t0:.1f}s)", flush=True)

    print(f"=== [{label}] SDXL: baseline vs protected ===", flush=True)
    t0 = time.time()
    lora_bxl = train_lora_sdxl(sdxl_branch, original_path, prompt, 1024, seed, TRAIN_STEPS)
    bxl = generate_and_score_sdxl(sdxl_branch, lora_bxl, prompt, 1024, true_image, clip_model, clip_processor)
    lora_bxl.unload()
    lora_axl = train_lora_sdxl(sdxl_branch, protected_path, prompt, 1024, seed, TRAIN_STEPS)
    axl = generate_and_score_sdxl(sdxl_branch, lora_axl, prompt, 1024, true_image, clip_model, clip_processor)
    lora_axl.unload()
    print(f"  [{label}] sdxl_delta={bxl - axl:+.4f} (done in {time.time() - t0:.1f}s)", flush=True)

    return {
        "sd15_baseline": b15, "sd15_protected": a15, "sd15_delta": b15 - a15,
        "sdxl_baseline": bxl, "sdxl_protected": axl, "sdxl_delta": bxl - axl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preset", default="HYBRID_FULL")
    parser.add_argument("--out-dir", default="/workspace/hybrid_n1_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_only_path = out_dir / "protected_latent_only.png"
    full_hybrid_path = out_dir / "protected_full_hybrid.png"

    # Phase 1: build both protected images. Subprocess-per-stage inside
    # hybrid_protect() means no training model of ours is resident yet --
    # same VRAM separation run_dual_arch_n30.py learned the hard way.
    print("### Building condition A: latent-only (stages 1-2) ###", flush=True)
    info_a = hybrid_protect(
        original_path=args.original, sd15_checkpoint=SD15_CKPT, sdxl_checkpoint=SDXL_CKPT,
        prompt=args.prompt, output_path=str(latent_only_path), preset_name=args.preset,
        seed=0, work_dir=str(out_dir / "_stages_a"), skip_pixel_stages=True,
    )

    print("### Building condition B: full hybrid (stages 1-4) ###", flush=True)
    info_b = hybrid_protect(
        original_path=args.original, sd15_checkpoint=SD15_CKPT, sdxl_checkpoint=SDXL_CKPT,
        prompt=args.prompt, output_path=str(full_hybrid_path), preset_name=args.preset,
        seed=0, work_dir=str(out_dir / "_stages_b"), skip_pixel_stages=False,
    )

    # Phase 2: load training models once, score both conditions.
    device = torch.device("cuda")
    sd15_attacker = ASPLAttacker(SD15_CKPT, device)
    sdxl_branch = SDXLBranch(SDXL_CKPT, device, torch.float32)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    true_image = Image.open(args.original).convert("RGB")
    sd15_text_embeddings = sd15_attacker.encode_prompt(args.prompt)

    common = dict(
        original_path=args.original, prompt=args.prompt, seed=args.seed,
        sd15_attacker=sd15_attacker, sd15_text_embeddings=sd15_text_embeddings,
        sdxl_branch=sdxl_branch, true_image=true_image,
        clip_model=clip_model, clip_processor=clip_processor,
    )
    result_a = score_condition("A: latent-only", str(latent_only_path), **common)
    result_b = score_condition("B: full hybrid", str(full_hybrid_path), **common)

    result = {
        "preset": args.preset,
        "seed": args.seed,
        "A_latent_only": {**result_a, "timings": info_a["timings"], "stages_run": info_a["stages_run"]},
        "B_full_hybrid": {**result_b, "timings": info_b["timings"], "stages_run": info_b["stages_run"]},
        "pixel_topup_marginal": {
            "sd15": result_b["sd15_delta"] - result_a["sd15_delta"],
            "sdxl": result_b["sdxl_delta"] - result_a["sdxl_delta"],
        },
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"=== DONE, wrote {result_path} ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
