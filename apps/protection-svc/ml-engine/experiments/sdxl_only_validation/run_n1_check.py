"""Quick n=1 real-training check for the SDXL-only ASPL attack
(aspl_attack_sdxl_only.py), isolating whether SDXL responds to this attack
class on its own -- as opposed to ensemble_attack_multiarch.py, where SDXL
was always attacked jointly with SD1.5 and showed no measured effect.

This is deliberately NOT the full n=30 kohya_ss pipeline used for the SD1.5
validation (run_multiarch_n30.py) -- this pod's slim image doesn't have
kohya_ss/bitsandbytes/xformers installed, and installing them just to check
one image before deciding whether SDXL is even worth pursuing further would
waste real GPU-hours. Instead this trains LoRAs directly via diffusers+peft
(the same mechanism the attack's own surrogate uses), targeting a comparable
training exposure (repeats x epochs ~= 200 steps, LoRA r=32/alpha=16, matching
run_multiarch_n30.py's kohya config as closely as a from-scratch loop
reasonably can) and scores with the same CLIP-similarity methodology as
run_multiarch_n30_score.py (openai/clip-vit-base-patch32, 4 samples, seed 42+i).

If this n=1 check shows a real delta (attacked LoRA generates measurably
less similar to the true image than baseline), that's the trigger to invest
in the full kohya_ss n=12-30 statistically-powered pipeline. If it shows
~zero delta like the joint multiarch run did, that's evidence SDXL itself
(not just the joint optimization) is the reason -- also a real, useful result.

Run directly with python3 (no accelerate needed -- single-process, single-GPU).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, "/workspace/dontai-protection-svc/ml-engine/src")
from ensemble_attack_multiarch import SDXLBranch  # noqa: E402
from style_cloak import load_image_tensor  # noqa: E402

TRAIN_STEPS = 200  # ~= kohya's 20 repeats x 10 epochs for a 1-image dataset
LORA_R = 32
LORA_ALPHA = 16
LR = 5e-5
GEN_SEED = 42
NUM_SAMPLES = 4


def train_lora(branch: SDXLBranch, image_path: str, prompt: str, size: int, seed: int, steps: int):
    from peft import LoraConfig, get_peft_model

    # branch.unet is mutated in place by get_peft_model (LoRA layers are attached
    # directly to its submodules, not copied) -- calling this twice on the same
    # branch without unload()-ing in between stacks a second adapter on top of
    # the first, so the two "independent" LoRAs actually share weights. Callers
    # MUST call .unload() on the returned model before training another LoRA on
    # the same branch.
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    lora_unet = get_peft_model(branch.unet, cfg)
    lora_unet.train()

    params = [p for n, p in lora_unet.named_parameters() if "lora_A" in n or "lora_B" in n]
    for p in params:
        p.requires_grad_(True)
    opt = optim.AdamW(params, lr=LR)

    generator = torch.Generator(device=branch.device).manual_seed(seed)
    x = load_image_tensor(image_path, size, branch.device).to(branch.dtype)
    cond = branch.encode_prompt(prompt, size)

    for step in range(steps):
        opt.zero_grad()
        loss = branch.denoising_loss(lora_unet, x, cond, generator)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"    step {step:3d}  loss={loss.item():.6f}")

    for p in params:
        p.requires_grad_(False)
    lora_unet.eval()
    return lora_unet


def clip_similarity(model, processor, img_a: Image.Image, img_b: Image.Image) -> float:
    inputs = processor(images=[img_a, img_b], return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1]).item()


@torch.no_grad()
def generate_and_score(branch: SDXLBranch, lora_unet, prompt: str, size: int, true_image: Image.Image, model, processor) -> float:
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_config(branch.scheduler.config)
    scheduler.set_timesteps(20)
    cond = branch.encode_prompt(prompt, size)
    scores = []

    for i in range(NUM_SAMPLES):
        gen = torch.Generator(device=branch.device).manual_seed(GEN_SEED + i)
        latents = torch.randn((1, 4, size // 8, size // 8), generator=gen, device=branch.device, dtype=branch.dtype)
        latents = latents * scheduler.init_noise_sigma
        for t in scheduler.timesteps:
            noise_pred = lora_unet(
                scheduler.scale_model_input(latents, t), t,
                encoder_hidden_states=cond["encoder_hidden_states"], added_cond_kwargs=cond["added_cond_kwargs"],
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        image = branch.vae.decode(latents / branch.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(image)
        scores.append(clip_similarity(model, processor, true_image, pil_image))

    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--attacked", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    branch = SDXLBranch(args.sdxl_checkpoint, device, torch.float32)
    true_image = Image.open(args.original).convert("RGB")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Train -> score -> unload for each condition in turn, never leaving two
    # adapters live on branch.unet at once (see train_lora's docstring note).
    print("[n1_check] training baseline LoRA...")
    t0 = time.time()
    lora_baseline = train_lora(branch, args.original, args.prompt, args.size, args.seed, TRAIN_STEPS)
    print(f"  done in {time.time() - t0:.1f}s")
    print("[n1_check] generating + scoring baseline...")
    baseline_score = generate_and_score(branch, lora_baseline, args.prompt, args.size, true_image, clip_model, clip_processor)
    print(f"  baseline_score={baseline_score:.4f}")
    lora_baseline.unload()

    print("[n1_check] training attacked LoRA...")
    t0 = time.time()
    lora_attacked = train_lora(branch, args.attacked, args.prompt, args.size, args.seed, TRAIN_STEPS)
    print(f"  done in {time.time() - t0:.1f}s")
    print("[n1_check] generating + scoring attacked...")
    attacked_score = generate_and_score(branch, lora_attacked, args.prompt, args.size, true_image, clip_model, clip_processor)
    print(f"  attacked_score={attacked_score:.4f}")
    lora_attacked.unload()

    delta = baseline_score - attacked_score
    result = {"baseline_score": baseline_score, "attacked_score": attacked_score, "delta": delta}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"[n1_check] delta={delta:+.4f} (positive = attack degraded similarity) -- wrote {args.output}")


if __name__ == "__main__":
    main()
