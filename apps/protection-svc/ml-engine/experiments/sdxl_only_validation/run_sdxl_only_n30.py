"""n=30 SDXL-only ASPL validation, run per-pod against a subset of the
10-image manifest (image assignment passed via --images). For each image:
attack once (SDXL_FULL preset), then for each seed in SEEDS train a
baseline LoRA and an attacked LoRA and score both against the true image
via CLIP similarity -- same methodology as the n=1/n=3 pilot in
run_n1_check.py (this script is that pilot's train_lora/generate_and_score
functions, reused directly, looped over the full image set with resume
support).

Deliberately does not use kohya_ss (not installed on this pod, and
installing it just to re-derive what run_n1_check.py already proved works
would add setup risk for no methodological necessity) -- LoRA training is
done directly via diffusers+peft, matching run_n1_check.py's config
(TRAIN_STEPS=200 ~= kohya's 20 repeats x 10 epochs, LoRA r=32/alpha=16).

Must run under a venv with diffusers + peft + transformers + torch/CUDA.
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
from aspl_attack_sdxl_only import aspl_attack_sdxl_only  # noqa: E402
from ensemble_attack_multiarch import SDXLBranch  # noqa: E402
from generate_subculture_images import PROMPTS as _ORIGINAL_PROMPTS  # noqa: E402
from generate_subculture_images import REPLICATION_PROMPTS  # noqa: E402
from run_n1_check import generate_and_score, train_lora  # noqa: E402

SEEDS = [1, 2, 3]
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
IMAGE_DIR = "/workspace/subculture_images"
OUT_DIR = Path("/workspace/sdxl_only_n30_out")

PROMPTS = {**_ORIGINAL_PROMPTS, **REPLICATION_PROMPTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="comma-separated image names (no extension) assigned to this pod")
    parser.add_argument("--pod-tag", required=True)
    args = parser.parse_args()

    image_names = [n.strip() for n in args.images.split(",") if n.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    branch = SDXLBranch(SDXL_CKPT, device, torch.float32)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    results = []
    for idx, name in enumerate(image_names, start=1):
        image_path = f"{IMAGE_DIR}/{name}.png"
        prompt = PROMPTS[name]
        true_image = Image.open(image_path).convert("RGB")

        attacked_path = OUT_DIR / f"attacked_{name}.png"
        if attacked_path.exists():
            print(f"=== [{idx}/{len(image_names)}] [{name}] attack already done, skipping (resume) ===")
        else:
            print(f"=== [{idx}/{len(image_names)}] [{name}] attacking (SDXL_FULL) ===")
            t0 = time.time()
            aspl_attack_sdxl_only(
                original_path=image_path, sdxl_checkpoint=SDXL_CKPT, prompt=prompt,
                output_path=str(attacked_path), preset_name="SDXL_FULL", seed=0,
            )
            print(f"  attack done in {time.time() - t0:.1f}s")

        for seed in SEEDS:
            result_path = OUT_DIR / f"result_{name}_{seed}.json"
            if result_path.exists():
                print(f"  [{name}/{seed}] already scored, skipping (resume)")
                results.append(json.loads(result_path.read_text()))
                continue

            t0 = time.time()
            lora_baseline = train_lora(branch, image_path, prompt, 1024, seed, 200)
            baseline_score = generate_and_score(branch, lora_baseline, prompt, 1024, true_image, clip_model, clip_processor)
            lora_baseline.unload()

            lora_attacked = train_lora(branch, str(attacked_path), prompt, 1024, seed, 200)
            attacked_score = generate_and_score(branch, lora_attacked, prompt, 1024, true_image, clip_model, clip_processor)
            lora_attacked.unload()

            row = {
                "name": name, "seed": seed,
                "baseline_score": baseline_score, "attacked_score": attacked_score,
                "delta": baseline_score - attacked_score,
            }
            result_path.write_text(json.dumps(row, indent=2))
            results.append(row)
            print(f"  [{name}/{seed}] delta={row['delta']:+.4f} (done in {time.time() - t0:.1f}s)")

    manifest_path = OUT_DIR / f"results_{args.pod_tag}.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"=== DONE, wrote {manifest_path} ===")


if __name__ == "__main__":
    main()
