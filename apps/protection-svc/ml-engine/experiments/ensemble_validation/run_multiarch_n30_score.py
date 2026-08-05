"""Scoring pass for run_multiarch_n30.py's output: generates samples from
each (image, seed, arch, condition) LoRA, scores CLIP similarity to the true
image, writes per-pod scores JSON for later cross-pod aggregation.

Must run under kohya_ss's venv (diffusers + peft + transformers + CUDA).
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

SEEDS = [1, 2, 3]
NUM_SAMPLES = 4
GEN_SEED = 42
OUT_DIR = Path("/workspace/n30_out")
SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"


def clip_similarity(model, processor, img_a: Image.Image, img_b: Image.Image) -> float:
    inputs = processor(images=[img_a, img_b], return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1]).item()


def score_lora(pipe, model, processor, lora_path: Path, true_image: Image.Image, resolution: int, prompt: str) -> float:
    pipe.load_lora_weights(str(lora_path))
    scores = []
    for i in range(NUM_SAMPLES):
        gen = torch.Generator(device="cuda").manual_seed(GEN_SEED + i)
        image = pipe(
            prompt=prompt, num_inference_steps=20, guidance_scale=7.0,
            height=resolution, width=resolution, generator=gen,
        ).images[0]
        scores.append(clip_similarity(model, processor, true_image, image))
    pipe.unload_lora_weights()
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-tag", required=True)
    args = parser.parse_args()

    manifest = json.loads((OUT_DIR / f"manifest_{args.pod_tag}.json").read_text())

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    pipe15 = StableDiffusionPipeline.from_single_file(SD15_CKPT, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipexl = StableDiffusionXLPipeline.from_single_file(SDXL_CKPT, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    results = []
    for entry in manifest:
        name, prompt, true_image_path = entry["name"], entry["prompt"], entry["true_image"]
        true_image = Image.open(true_image_path).convert("RGB")
        for seed in SEEDS:
            b15 = score_lora(pipe15, model, processor, OUT_DIR / f"lora_{name}_{seed}_sd15_baseline" / "baseline_v1.safetensors", true_image, 512, prompt)
            a15 = score_lora(pipe15, model, processor, OUT_DIR / f"lora_{name}_{seed}_sd15_attacked" / "attacked_v1.safetensors", true_image, 512, prompt)
            bxl = score_lora(pipexl, model, processor, OUT_DIR / f"lora_{name}_{seed}_sdxl_baseline" / "baseline_v1.safetensors", true_image, 1024, prompt)
            axl = score_lora(pipexl, model, processor, OUT_DIR / f"lora_{name}_{seed}_sdxl_attacked" / "attacked_v1.safetensors", true_image, 1024, prompt)
            row = {
                "name": name, "seed": seed,
                "sd15_baseline": b15, "sd15_attacked": a15, "sd15_delta": b15 - a15,
                "sdxl_baseline": bxl, "sdxl_attacked": axl, "sdxl_delta": bxl - axl,
            }
            results.append(row)
            print(f"[{name}/{seed}] sd15_delta={b15 - a15:+.4f} sdxl_delta={bxl - axl:+.4f}")

    out_path = OUT_DIR / f"scores_{args.pod_tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
