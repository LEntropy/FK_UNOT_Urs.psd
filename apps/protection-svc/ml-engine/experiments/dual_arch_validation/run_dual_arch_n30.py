"""n=30 validation of the SEQUENTIAL dual-arch composition remote_dual_arch_
cloak() (apps/protection-svc/remote_gpu.py) uses in production: for each
image, run aspl_attack.py (SD1.5) on the original to get an intermediate,
then aspl_attack_sdxl_only.py (SDXL) on *that* intermediate to get the final
chained-attacked image.

Each stage's own protective effect was independently validated (attacking a
pristine original) -- SD1.5 via multiarch_ensemble_attack's n=30/n=12
replication, SDXL via aspl_attack_sdxl_only's own n=30/n=12 replication (see
[[lora-protection-research]] memory). What was NOT validated is whether the
*final chained image* still protects against training on BOTH
architectures, or whether the SDXL stage (attacking an already-perturbed
image instead of a pristine one) degrades the SD1.5 protection that's
already baked in, or vice versa. This script measures exactly that: for
each (image, seed), trains an SD1.5 LoRA AND an SDXL LoRA on baseline vs
the final chained image (4 trainings total, mirroring the original joint
n=30 experiment's structure -- run_multiarch_n30.py), CLIP-scores each
against the true image the same way every prior experiment in this project
has.

No kohya_ss on this pod -- LoRA training done directly via diffusers+peft
(TRAIN_STEPS=200 ~= kohya's 20 repeats x 10 epochs, LoRA r=32/alpha=16),
same as run_n1_check.py/run_sdxl_only_n30.py used for the SDXL-only
validation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, "/workspace/dontai-protection-svc/ml-engine/src")
sys.path.insert(0, "/workspace")
from aspl_attack import ASPLAttacker, aspl_attack  # noqa: E402
from aspl_attack_sdxl_only import aspl_attack_sdxl_only  # noqa: E402
from ensemble_attack_multiarch import SDXLBranch  # noqa: E402
from generate_subculture_images import PROMPTS as _ORIGINAL_PROMPTS  # noqa: E402
from generate_subculture_images import REPLICATION_PROMPTS  # noqa: E402
from run_n1_check import clip_similarity  # noqa: E402
from run_n1_check import generate_and_score as generate_and_score_sdxl  # noqa: E402
from run_n1_check import train_lora as train_lora_sdxl  # noqa: E402
from style_cloak import load_image_tensor  # noqa: E402

PROMPTS = {**_ORIGINAL_PROMPTS, **REPLICATION_PROMPTS}

SEEDS = [1, 2, 3]
SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
IMAGE_DIR = "/workspace/subculture_images"
OUT_DIR = Path("/workspace/dual_arch_n30_out")

TRAIN_STEPS = 200
LORA_R = 32
LORA_ALPHA = 16
LR = 5e-5
GEN_SEED = 42
NUM_SAMPLES = 4


def train_lora_sd15(attacker: ASPLAttacker, image_path: str, text_embeddings, seed: int, steps: int):
    from peft import LoraConfig, get_peft_model

    # Same in-place-mutation caveat as run_n1_check.py's train_lora: callers
    # MUST call .unload() before training another LoRA on the same attacker.
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    lora_unet = get_peft_model(attacker.base_unet, cfg)
    lora_unet.train()

    params = [p for n, p in lora_unet.named_parameters() if "lora_A" in n or "lora_B" in n]
    for p in params:
        p.requires_grad_(True)
    opt = optim.AdamW(params, lr=LR)

    generator = torch.Generator(device=attacker.device).manual_seed(seed)
    x = load_image_tensor(image_path, 512, attacker.device).to(attacker.dtype)

    for step in range(steps):
        opt.zero_grad()
        loss = attacker.denoising_loss(lora_unet, x, text_embeddings, generator)
        loss.backward()
        opt.step()

    for p in params:
        p.requires_grad_(False)
    lora_unet.eval()
    return lora_unet


@torch.no_grad()
def generate_and_score_sd15(attacker: ASPLAttacker, lora_unet, text_embeddings, true_image: Image.Image, model, processor) -> float:
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_config(attacker.scheduler.config)
    scheduler.set_timesteps(20)
    scores = []

    for i in range(NUM_SAMPLES):
        gen = torch.Generator(device=attacker.device).manual_seed(GEN_SEED + i)
        latents = torch.randn((1, 4, 64, 64), generator=gen, device=attacker.device, dtype=attacker.dtype)
        latents = latents * scheduler.init_noise_sigma
        for t in scheduler.timesteps:
            noise_pred = lora_unet(scheduler.scale_model_input(latents, t), t, encoder_hidden_states=text_embeddings).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        image = attacker.vae.decode(latents / attacker.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(image)
        scores.append(clip_similarity(model, processor, true_image, pil_image))

    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="comma-separated image names (no extension) assigned to this pod")
    parser.add_argument("--pod-tag", required=True)
    args = parser.parse_args()

    image_names = [n.strip() for n in args.images.split(",") if n.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: run every image's chained attack first, with NO training models
    # of our own resident in VRAM. aspl_attack()/aspl_attack_sdxl_only() each
    # instantiate their own ASPLAttacker/SDXLBranch internally -- loading our
    # own copies of those same models *before* calling these functions (as an
    # earlier version of this script did) put two copies of each architecture
    # in VRAM at once during the attack phase and reliably OOM'd on all 3 pods
    # (44.39/44.43 GiB used, crashed before finishing even one image). Keeping
    # phases separate means at most one instance of each model exists at a time.
    for idx, name in enumerate(image_names, start=1):
        image_path = f"{IMAGE_DIR}/{name}.png"
        prompt = PROMPTS[name]

        intermediate_path = OUT_DIR / f"sd15_attacked_{name}.png"
        final_path = OUT_DIR / f"chained_attacked_{name}.png"

        if final_path.exists():
            print(f"=== [{idx}/{len(image_names)}] [{name}] chained attack already done, skipping (resume) ===")
            continue

        if intermediate_path.exists():
            print(f"=== [{idx}/{len(image_names)}] [{name}] SD1.5 stage already done, skipping (resume) ===")
        else:
            print(f"=== [{idx}/{len(image_names)}] [{name}] SD1.5 attack (L3_ANTI_TRAIN) ===")
            t0 = time.time()
            aspl_attack(
                original_path=image_path, checkpoint_path=SD15_CKPT, prompt=prompt,
                output_path=str(intermediate_path), preset_name="L3_ANTI_TRAIN", size=512, seed=0,
            )
            print(f"  SD1.5 stage done in {time.time() - t0:.1f}s")

        print(f"  [{name}] SDXL stage on SD1.5-attacked output (SDXL_FULL) ===")
        t0 = time.time()
        aspl_attack_sdxl_only(
            original_path=str(intermediate_path), sdxl_checkpoint=SDXL_CKPT, prompt=prompt,
            output_path=str(final_path), preset_name="SDXL_FULL", seed=0,
        )
        print(f"  SDXL stage done in {time.time() - t0:.1f}s")

    # Phase 2: all attacks are done -- now load the training models once and
    # reuse them across every image/seed's training+scoring.
    device = torch.device("cuda")
    sd15_attacker = ASPLAttacker(SD15_CKPT, device)
    sdxl_branch = SDXLBranch(SDXL_CKPT, device, torch.float32)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    results = []
    for idx, name in enumerate(image_names, start=1):
        image_path = f"{IMAGE_DIR}/{name}.png"
        prompt = PROMPTS[name]
        true_image = Image.open(image_path).convert("RGB")
        final_path = OUT_DIR / f"chained_attacked_{name}.png"

        sd15_text_embeddings = sd15_attacker.encode_prompt(prompt)

        for seed in SEEDS:
            result_path = OUT_DIR / f"result_{name}_{seed}.json"
            if result_path.exists():
                print(f"  [{name}/{seed}] already scored, skipping (resume)")
                results.append(json.loads(result_path.read_text()))
                continue

            t0 = time.time()

            lora_b15 = train_lora_sd15(sd15_attacker, image_path, sd15_text_embeddings, seed, TRAIN_STEPS)
            b15 = generate_and_score_sd15(sd15_attacker, lora_b15, sd15_text_embeddings, true_image, clip_model, clip_processor)
            lora_b15.unload()

            lora_a15 = train_lora_sd15(sd15_attacker, str(final_path), sd15_text_embeddings, seed, TRAIN_STEPS)
            a15 = generate_and_score_sd15(sd15_attacker, lora_a15, sd15_text_embeddings, true_image, clip_model, clip_processor)
            lora_a15.unload()

            lora_bxl = train_lora_sdxl(sdxl_branch, image_path, prompt, 1024, seed, TRAIN_STEPS)
            bxl = generate_and_score_sdxl(sdxl_branch, lora_bxl, prompt, 1024, true_image, clip_model, clip_processor)
            lora_bxl.unload()

            lora_axl = train_lora_sdxl(sdxl_branch, str(final_path), prompt, 1024, seed, TRAIN_STEPS)
            axl = generate_and_score_sdxl(sdxl_branch, lora_axl, prompt, 1024, true_image, clip_model, clip_processor)
            lora_axl.unload()

            row = {
                "name": name, "seed": seed,
                "sd15_baseline": b15, "sd15_attacked": a15, "sd15_delta": b15 - a15,
                "sdxl_baseline": bxl, "sdxl_attacked": axl, "sdxl_delta": bxl - axl,
            }
            result_path.write_text(json.dumps(row, indent=2))
            results.append(row)
            print(f"  [{name}/{seed}] sd15_delta={row['sd15_delta']:+.4f} sdxl_delta={row['sdxl_delta']:+.4f} (done in {time.time() - t0:.1f}s)")

    manifest_path = OUT_DIR / f"results_{args.pod_tag}.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"=== DONE, wrote {manifest_path} ===")


if __name__ == "__main__":
    main()
