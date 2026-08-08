"""n=1 real-training check for the CHAINED latent-space dual-arch attack
(aspl_attack_latent.py -> aspl_attack_sdxl_only_latent.py), mirroring
run_dual_arch_n30.py's per-image-per-seed body (train+score an SD1.5 LoRA
and an SDXL LoRA, baseline vs attacked) but at n=1: today's pixel-space
chained attack visibly distorts the image enough to contradict this
project's own "looks nearly identical" UI copy ([[strong-protection-visual-
honesty]]); the latent-space attack looked qualitatively more natural in a
same-day visual pilot (see PHASE4_SCOPING.md SS6). This checks whether that
visual improvement survived as a real training-defeating effect, or traded
it away -- run before investing in a full n=30 validation of this new
mechanism.
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, "/workspace/dontai-protection-svc/ml-engine/src")
from aspl_attack import ASPLAttacker  # noqa: E402
from aspl_attack_latent import aspl_attack_latent  # noqa: E402
from aspl_attack_sdxl_only_latent import aspl_attack_sdxl_only_latent  # noqa: E402
from ensemble_attack_multiarch import SDXLBranch  # noqa: E402
from run_dual_arch_n30 import generate_and_score_sd15, train_lora_sd15  # noqa: E402
from run_n1_check import generate_and_score as generate_and_score_sdxl  # noqa: E402
from run_n1_check import train_lora as train_lora_sdxl  # noqa: E402

SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
TRAIN_STEPS = 200


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--latent-epsilon", type=float, default=0.3)
    parser.add_argument("--out-dir", default="/workspace/latent_n1_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = out_dir / "sd15_latent_attacked.png"
    final_path = out_dir / "chained_latent_final.png"

    # Phase 1: chained attack, no training models resident (same OOM lesson
    # as run_dual_arch_n30.py -- see that script's own comment).
    print("=== SD1.5 latent attack ===")
    t0 = time.time()
    aspl_attack_latent(
        original_path=args.original, checkpoint_path=SD15_CKPT, prompt=args.prompt,
        output_path=str(intermediate_path), preset_name="L3_ANTI_TRAIN", seed=0,
        latent_epsilon=args.latent_epsilon,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("=== SDXL latent attack (on SD1.5-latent-attacked output) ===")
    t0 = time.time()
    aspl_attack_sdxl_only_latent(
        original_path=str(intermediate_path), sdxl_checkpoint=SDXL_CKPT, prompt=args.prompt,
        output_path=str(final_path), preset_name="SDXL_FULL", seed=0,
        latent_epsilon=args.latent_epsilon,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # Phase 2: load training models, train+score both architectures.
    device = torch.device("cuda")
    sd15_attacker = ASPLAttacker(SD15_CKPT, device)
    sdxl_branch = SDXLBranch(SDXL_CKPT, device, torch.float32)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    true_image = Image.open(args.original).convert("RGB")
    sd15_text_embeddings = sd15_attacker.encode_prompt(args.prompt)

    print("=== SD1.5: training baseline vs latent-attacked LoRA ===")
    t0 = time.time()
    lora_b15 = train_lora_sd15(sd15_attacker, args.original, sd15_text_embeddings, args.seed, TRAIN_STEPS)
    b15 = generate_and_score_sd15(sd15_attacker, lora_b15, sd15_text_embeddings, true_image, clip_model, clip_processor)
    lora_b15.unload()
    lora_a15 = train_lora_sd15(sd15_attacker, str(final_path), sd15_text_embeddings, args.seed, TRAIN_STEPS)
    a15 = generate_and_score_sd15(sd15_attacker, lora_a15, sd15_text_embeddings, true_image, clip_model, clip_processor)
    lora_a15.unload()
    print(f"  sd15_baseline={b15:.4f} sd15_attacked={a15:.4f} sd15_delta={b15 - a15:+.4f} (done in {time.time() - t0:.1f}s)")

    print("=== SDXL: training baseline vs latent-attacked LoRA ===")
    t0 = time.time()
    lora_bxl = train_lora_sdxl(sdxl_branch, args.original, args.prompt, 1024, args.seed, TRAIN_STEPS)
    bxl = generate_and_score_sdxl(sdxl_branch, lora_bxl, args.prompt, 1024, true_image, clip_model, clip_processor)
    lora_bxl.unload()
    lora_axl = train_lora_sdxl(sdxl_branch, str(final_path), args.prompt, 1024, args.seed, TRAIN_STEPS)
    axl = generate_and_score_sdxl(sdxl_branch, lora_axl, args.prompt, 1024, true_image, clip_model, clip_processor)
    lora_axl.unload()
    print(f"  sdxl_baseline={bxl:.4f} sdxl_attacked={axl:.4f} sdxl_delta={bxl - axl:+.4f} (done in {time.time() - t0:.1f}s)")

    result = {
        "sd15_baseline": b15, "sd15_attacked": a15, "sd15_delta": b15 - a15,
        "sdxl_baseline": bxl, "sdxl_attacked": axl, "sdxl_delta": bxl - axl,
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"=== DONE, wrote {result_path} ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
