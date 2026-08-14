"""Test Lab's real-LoRA-training protection-strength score: the exact
methodology this project used to validate strong_protection for real
(experiments/dual_arch_validation/run_dual_arch_n30.py, n=30 statistical
result -- see PHASE4_SCOPING.md SS6 and the lora-protection-research
memory), run once against a single user's own image on demand instead of
n=30 images for a paper-grade statistical claim. A single run is a real
measurement, not noise-free -- surfaced to the user as one concrete
before/after result with its own sample images, not as a reproduction of
the full statistical claim (that lives in PHASE4_SCOPING.md, not
per-artwork).

For each architecture (SD1.5, SDXL): train a LoRA on the ORIGINAL image
and a separate LoRA on the PROTECTED image (same prompt/seed/step count
for both, so training conditions differ only in which image was used),
generate sample images from each trained LoRA, and CLIP-score each
sample against the true original. A working protection shows the
protected LoRA's average similarity meaningfully lower than the
baseline LoRA's -- the same baseline-vs-condition delta every mechanism
in this project has been judged by (model_leak_detect.py's mirror-image
version of this same idea has the fuller explanation of that
methodology).

Runs inside the RunPod Serverless strong_protection worker (see the
sibling docker/strongprotect-serverless/handler.py) -- reuses the same
ASPLAttacker (SD1.5) / SDXLBranch (SDXL) classes the attack scripts
already load checkpoints through, so no separate model-loading code path
is needed here.

Progress goes to stderr, not stdout -- the CLI entrypoint's final
`print(json.dumps(result))` on stdout is the only thing a caller
(handler.py, via subprocess) should try to parse.
"""

import argparse
import base64
import io
import json
import sys

import torch
import torch.optim as optim
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Same real-effect threshold reused project-wide (model_leak_detect.py's
# THRESHOLD, evaluate.py's styleDriftScore pass bar) -- same magnitude,
# not independently calibrated per mechanism, see model_leak_detect.py's
# own honesty note about why.
THRESHOLD = 0.03

LORA_R_SD15 = 32
LORA_ALPHA_SD15 = 16
LORA_R_SDXL = 4
LR = 5e-5


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def clip_similarity(model: CLIPModel, processor: CLIPProcessor, image_a: Image.Image, image_b: Image.Image) -> float:
    inputs = processor(images=[image_a, image_b], return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return float((features[0] @ features[1]).item())


def _train_lora_sd15(attacker, image_path: str, text_embeddings, seed: int, steps: int):
    from peft import LoraConfig, get_peft_model
    from style_cloak import load_image_tensor

    # Same in-place-mutation caveat as run_dual_arch_n30.py's train_lora_sd15:
    # callers MUST call .unload() before training another LoRA on the same
    # attacker's base_unet.
    cfg = LoraConfig(r=LORA_R_SD15, lora_alpha=LORA_ALPHA_SD15, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    lora_unet = get_peft_model(attacker.base_unet, cfg)
    lora_unet.train()

    params = [p for n, p in lora_unet.named_parameters() if "lora_A" in n or "lora_B" in n]
    for p in params:
        p.requires_grad_(True)
    opt = optim.AdamW(params, lr=LR)

    generator = torch.Generator(device=attacker.device).manual_seed(seed)
    x = load_image_tensor(image_path, 512, attacker.device).to(attacker.dtype)

    for _ in range(steps):
        opt.zero_grad()
        loss = attacker.denoising_loss(lora_unet, x, text_embeddings, generator)
        loss.backward()
        opt.step()

    for p in params:
        p.requires_grad_(False)
    lora_unet.eval()
    return lora_unet


@torch.no_grad()
def _generate_and_score_sd15(
    attacker, lora_unet, text_embeddings, true_image, model, processor, num_samples, gen_seed,
    controlnet=None, control_image_tensor=None, controlnet_scale=0.8,
):
    from diffusers import DDIMScheduler

    from controlnet_condition import controlnet_residuals_sd15

    scheduler = DDIMScheduler.from_config(attacker.scheduler.config)
    scheduler.set_timesteps(20)
    scores = []
    samples = []

    for i in range(num_samples):
        gen = torch.Generator(device=attacker.device).manual_seed(gen_seed + i)
        latents = torch.randn((1, 4, 64, 64), generator=gen, device=attacker.device, dtype=attacker.dtype)
        latents = latents * scheduler.init_noise_sigma
        for t in scheduler.timesteps:
            latents_input = scheduler.scale_model_input(latents, t)
            unet_kwargs = {}
            if controlnet is not None:
                down_res, mid_res = controlnet_residuals_sd15(
                    controlnet, latents_input, t, text_embeddings, control_image_tensor, controlnet_scale
                )
                unet_kwargs = {"down_block_additional_residuals": down_res, "mid_block_additional_residual": mid_res}
            noise_pred = lora_unet(latents_input, t, encoder_hidden_states=text_embeddings, **unet_kwargs).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        image = attacker.vae.decode(latents / attacker.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(image)
        samples.append(pil_image)
        scores.append(clip_similarity(model, processor, true_image, pil_image))

    return sum(scores) / len(scores), samples


def _train_lora_sdxl(branch, image_path: str, cond, seed: int, steps: int, size: int):
    from style_cloak import load_image_tensor

    surrogate = branch.build_surrogate(LORA_R_SDXL)
    surrogate.train()
    params = [p for n, p in surrogate.named_parameters() if "lora_A" in n or "lora_B" in n]
    for p in params:
        p.requires_grad_(True)
    opt = optim.AdamW(params, lr=LR)

    generator = torch.Generator(device=branch.device).manual_seed(seed)
    x = load_image_tensor(image_path, size, branch.device).to(branch.dtype)

    for _ in range(steps):
        opt.zero_grad()
        loss = branch.denoising_loss(surrogate, x, cond, generator)
        loss.backward()
        opt.step()

    for p in params:
        p.requires_grad_(False)
    surrogate.eval()
    return surrogate


@torch.no_grad()
def _generate_and_score_sdxl(
    branch, lora_unet, cond, true_image, model, processor, size, num_samples, gen_seed,
    controlnet=None, control_image_tensor=None, controlnet_scale=0.8,
):
    from diffusers import DDIMScheduler

    from controlnet_condition import controlnet_residuals_sdxl

    scheduler = DDIMScheduler.from_config(branch.scheduler.config)
    scheduler.set_timesteps(20)
    latent_size = size // 8
    scores = []
    samples = []

    for i in range(num_samples):
        gen = torch.Generator(device=branch.device).manual_seed(gen_seed + i)
        latents = torch.randn((1, 4, latent_size, latent_size), generator=gen, device=branch.device, dtype=branch.dtype)
        latents = latents * scheduler.init_noise_sigma
        for t in scheduler.timesteps:
            latents_input = scheduler.scale_model_input(latents, t)
            unet_kwargs = {}
            if controlnet is not None:
                down_res, mid_res = controlnet_residuals_sdxl(
                    controlnet, latents_input, t, cond["encoder_hidden_states"],
                    cond["added_cond_kwargs"], control_image_tensor, controlnet_scale,
                )
                unet_kwargs = {"down_block_additional_residuals": down_res, "mid_block_additional_residual": mid_res}
            noise_pred = lora_unet(
                latents_input,
                t,
                encoder_hidden_states=cond["encoder_hidden_states"],
                added_cond_kwargs=cond["added_cond_kwargs"],
                **unet_kwargs,
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        image = branch.vae.decode(latents / branch.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(image)
        samples.append(pil_image)
        scores.append(clip_similarity(model, processor, true_image, pil_image))

    return sum(scores) / len(scores), samples


def _samples_to_b64(samples: list) -> list:
    out = []
    for img in samples:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _verdict(delta: float) -> str:
    if delta > THRESHOLD:
        return "PROTECTED"
    if delta > THRESHOLD / 2:
        return "WEAK"
    return "NOT_PROTECTED"


def score_protection(
    original_path: str,
    protected_path: str,
    prompt: str,
    sd15_checkpoint: str,
    sdxl_checkpoint: str,
    seed: int = 1,
    train_steps: int = 150,
    num_samples: int = 2,
    gen_seed: int = 42,
    use_controlnet: bool = True,
    controlnet_scale: float = 0.8,
) -> dict:
    from aspl_attack import ASPLAttacker
    from caption_image import caption_image
    from controlnet_condition import (
        compute_edge_control_image,
        control_image_to_tensor,
        load_controlnet_sd15,
        load_controlnet_sdxl,
    )
    from ensemble_attack_multiarch import SDXLBranch

    # `prompt` is now the artwork's own upload-time tags (asset-service's
    # routes/artworks.ts joins them), which describe the image's real
    # content the way a free-text title often doesn't -- a title like
    # "무제" or a pun collapses the baseline LoRA's similarity score toward
    # the CLIP floor regardless of protection, which was caught live giving
    # a nonsensical negative delta on a real deployed artwork. Auto-caption
    # is kept only as the fallback for the rare artwork with zero tags, not
    # as the default path anymore.
    if not prompt or not prompt.strip():
        prompt = caption_image(original_path)
        _log(f"[protection_score] no usable caller prompt -- using content caption instead: {prompt!r}")
    else:
        _log(f"[protection_score] using caller-supplied prompt (artwork tags): {prompt!r}")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    true_image = Image.open(original_path).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ControlNet conditioning (2026-08-13, Test Lab "does the user understand
    # what changed" fix): pins BOTH baseline and protected previews to the
    # true original's own edge/composition, via `controlnet_condition.py`.
    # Best-effort -- a download/load failure here should degrade to the
    # pre-existing unconditioned t2i behavior, not break scoring entirely.
    controlnet_sd15 = None
    controlnet_sdxl = None
    control_tensor_sd15 = None
    control_tensor_sdxl = None
    if use_controlnet:
        try:
            controlnet_sd15 = load_controlnet_sd15(device, torch.float32)
            control_image_sd15 = compute_edge_control_image(original_path, 512)
            control_tensor_sd15 = control_image_to_tensor(control_image_sd15, device, torch.float32)
            controlnet_sdxl = load_controlnet_sdxl(device, torch.float32)
            control_image_sdxl = compute_edge_control_image(original_path, 1024)
            control_tensor_sdxl = control_image_to_tensor(control_image_sdxl, device, torch.float32)
            _log("[protection_score] ControlNet loaded -- baseline/protected previews will share the original's composition")
        except Exception as exc:  # noqa: BLE001 -- best-effort, see comment above
            _log(f"[protection_score] ControlNet unavailable, falling back to unconditioned t2i: {exc!r}")
            controlnet_sd15 = None
            controlnet_sdxl = None

    _log("[protection_score] SD1.5: training baseline LoRA")
    sd15 = ASPLAttacker(sd15_checkpoint, device)
    sd15_text = sd15.encode_prompt(prompt)

    lora_baseline_15 = _train_lora_sd15(sd15, original_path, sd15_text, seed, train_steps)
    baseline_score_15, baseline_samples_15 = _generate_and_score_sd15(
        sd15, lora_baseline_15, sd15_text, true_image, clip_model, clip_processor, num_samples, gen_seed,
        controlnet_sd15, control_tensor_sd15, controlnet_scale,
    )
    lora_baseline_15.unload()

    _log("[protection_score] SD1.5: training protected LoRA")
    lora_protected_15 = _train_lora_sd15(sd15, protected_path, sd15_text, seed, train_steps)
    protected_score_15, protected_samples_15 = _generate_and_score_sd15(
        sd15, lora_protected_15, sd15_text, true_image, clip_model, clip_processor, num_samples, gen_seed,
        controlnet_sd15, control_tensor_sd15, controlnet_scale,
    )
    lora_protected_15.unload()

    del sd15
    if controlnet_sd15 is not None:
        del controlnet_sd15
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _log("[protection_score] SDXL: training baseline LoRA")
    sdxl = SDXLBranch(sdxl_checkpoint, device, torch.float32)
    sdxl_cond = sdxl.encode_prompt(prompt, 1024)

    lora_baseline_xl = _train_lora_sdxl(sdxl, original_path, sdxl_cond, seed, train_steps, 1024)
    baseline_score_xl, baseline_samples_xl = _generate_and_score_sdxl(
        sdxl, lora_baseline_xl, sdxl_cond, true_image, clip_model, clip_processor, 1024, num_samples, gen_seed,
        controlnet_sdxl, control_tensor_sdxl, controlnet_scale,
    )
    lora_baseline_xl.unload()

    _log("[protection_score] SDXL: training protected LoRA")
    lora_protected_xl = _train_lora_sdxl(sdxl, protected_path, sdxl_cond, seed, train_steps, 1024)
    protected_score_xl, protected_samples_xl = _generate_and_score_sdxl(
        sdxl, lora_protected_xl, sdxl_cond, true_image, clip_model, clip_processor, 1024, num_samples, gen_seed,
        controlnet_sdxl, control_tensor_sdxl, controlnet_scale,
    )
    lora_protected_xl.unload()

    del sdxl
    if controlnet_sdxl is not None:
        del controlnet_sdxl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    delta_15 = baseline_score_15 - protected_score_15
    delta_xl = baseline_score_xl - protected_score_xl

    return {
        "sd15": {
            "baselineSimilarity": baseline_score_15,
            "protectedSimilarity": protected_score_15,
            "delta": delta_15,
            "verdict": _verdict(delta_15),
            "baselineSamples": _samples_to_b64(baseline_samples_15),
            "protectedSamples": _samples_to_b64(protected_samples_15),
        },
        "sdxl": {
            "baselineSimilarity": baseline_score_xl,
            "protectedSimilarity": protected_score_xl,
            "delta": delta_xl,
            "verdict": _verdict(delta_xl),
            "baselineSamples": _samples_to_b64(baseline_samples_xl),
            "protectedSamples": _samples_to_b64(protected_samples_xl),
        },
        "threshold": THRESHOLD,
        # The actual training/generation prompt used (see caption_image.py's
        # module doc for why this is a content caption, not the caller's
        # title) -- surfaced so a caller can tell a bad-baseline artifact
        # apart from a real result (e.g. baselineSimilarity well below this
        # project's healthy ~0.73-0.75 range is a signal to distrust delta,
        # not evidence of a real effect either direction).
        "contentPrompt": prompt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--protected", required=True)
    parser.add_argument("--sd15-checkpoint", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", default="artwork")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--no-controlnet", action="store_true", help="disable ControlNet composition-pinning, fall back to unconditioned t2i")
    parser.add_argument("--controlnet-scale", type=float, default=0.8)
    args = parser.parse_args()

    result = score_protection(
        original_path=args.original,
        protected_path=args.protected,
        prompt=args.prompt,
        sd15_checkpoint=args.sd15_checkpoint,
        sdxl_checkpoint=args.sdxl_checkpoint,
        seed=args.seed,
        train_steps=args.train_steps,
        num_samples=args.num_samples,
        gen_seed=args.gen_seed,
        use_controlnet=not args.no_controlnet,
        controlnet_scale=args.controlnet_scale,
    )
    print(json.dumps(result))
