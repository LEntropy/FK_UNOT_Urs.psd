"""Empirical calibration: at what latent_drift (from the protected image)
does the real training-based protection delta actually start collapsing?
Answers the question quality_restore.py's DRIFT_GATE guessed wrong at.

Sweeps t in a linear blend from x_prot (t=0, drift=0 by definition) toward
x_orig (t=1), measuring latent_drift and real SD1.5 train+score delta at
each point. This directly maps drift -> delta, which is the only honest
way to pick a safe drift budget.
"""

import sys
sys.path.insert(0, "/workspace/dontai-protection-svc/ml-engine/src")

import json
import torch

import caption_image
caption_image.caption_image = lambda path: "anime original character, illustration"

import protection_score as ps
from style_cloak import load_image_tensor, save_tensor_image

SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
ORIGINAL = "/workspace/user_illust.jpg"
PROTECTED = "/workspace/vae_unc_eps03.png"
PROMPT = "anime original character, illustration"

device = torch.device("cuda")

x_orig = load_image_tensor(ORIGINAL, 512, device)
x_prot = load_image_tensor(PROTECTED, 512, device)

from aspl_attack import ASPLAttacker

attacker = ASPLAttacker(SD15_CKPT, device)
# Reuse attacker.vae for drift measurement instead of loading a second
# standalone VAE -- same weights, avoids holding two VAE copies in VRAM
# alongside the full UNet/text-encoder ASPLAttacker also loads.
vae = attacker.vae

with torch.no_grad():
    target = vae.encode(x_prot.to(torch.float32) * 2 - 1).latent_dist
    z_target = target.mean.clone()
    logvar_target = target.logvar.clone()

def drift_of(x):
    with torch.no_grad():
        post = vae.encode(x.to(torch.float32) * 2 - 1).latent_dist
        return float((torch.nn.functional.mse_loss(post.mean, z_target) +
                      torch.nn.functional.mse_loss(post.logvar, logvar_target)).item())

text_embeddings = attacker.encode_prompt(PROMPT)
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
true_image = ps.Image.open(ORIGINAL).convert("RGB")

TRAIN_STEPS = 150
SEED = 1

results = []

# baseline (t=1.0 conceptually, but computed directly on the real original)
print("=== baseline (original) ===", flush=True)
lora_b = ps._train_lora_sd15(attacker, ORIGINAL, text_embeddings, SEED, TRAIN_STEPS)
baseline_score, _ = ps._generate_and_score_sd15(attacker, lora_b, text_embeddings, true_image, clip_model, clip_processor, 1, 42)
lora_b.unload()
print(f"baseline_score={baseline_score:.4f}", flush=True)

for t in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
    x_blend = (x_prot + t * (x_orig - x_prot)).clamp(0, 1)
    path = f"/workspace/calib_t{t}.png"
    save_tensor_image(x_blend, path)
    d = drift_of(x_blend)

    lora = ps._train_lora_sd15(attacker, path, text_embeddings, SEED, TRAIN_STEPS)
    score, _ = ps._generate_and_score_sd15(attacker, lora, text_embeddings, true_image, clip_model, clip_processor, 1, 42)
    lora.unload()

    delta = baseline_score - score
    print(f"t={t:.2f}  drift={d:.4e}  score={score:.4f}  delta={delta:+.4f}", flush=True)
    results.append({"t": t, "drift": d, "score": score, "delta": delta})

out = {"baseline_score": baseline_score, "points": results}
with open("/workspace/drift_calibration_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("CALIBRATION_DONE", flush=True)
print(json.dumps(out, indent=2), flush=True)
