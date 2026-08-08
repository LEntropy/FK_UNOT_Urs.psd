"""Full Anti-DreamBooth (ASPL -- Alternating Surrogate and Perturbation
Learning) attack. Van Le et al., "Anti-DreamBooth: Protecting Users from
Personalized Text-to-Image Synthesis", ICCV 2023.

diffusion_attack.py's attack() is a *simplified* variant that just PGD-attacks
the frozen base checkpoint's own denoising loss directly -- one gradient per
step, same cost class as one training step. Real ASPL instead alternates,
many times over:

  (a) briefly fine-tune a surrogate LoRA on the *current* perturbed image
      (minimize denoising loss -- simulates "someone actually trains on this")
  (b) PGD-update the perturbation against that just-trained surrogate
      (maximize denoising loss)

so the perturbation is optimized against a model that has genuinely seen
perturbed data before, rather than against a checkpoint that never adapts.
The surrogate LoRA is periodically reset (fresh random init) so delta doesn't
just overfit to one surrogate's particular trajectory.

Validated in experiments/diffusion_attack_validation/ that the simplified
attack (mean delta +0.0071, CI includes zero) doesn't reliably degrade LoRA
fidelity -- this is the real, more expensive mechanism the literature
actually uses, tried next per the user's own "느려도 상관없어" scoping.

Must run under kohya_ss's venv (diffusers + peft + CUDA).
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from style_cloak import compute_perceptual_mask, load_image_tensor, save_tensor_image


@dataclass
class ASPLPreset:
    epsilon: float
    outer_iters: int
    surrogate_steps: int
    pgd_steps: int
    reset_every: int
    surrogate_lr: float
    pgd_lr: float


ASPL_PRESETS = {
    "L1_PREVIEW": ASPLPreset(
        epsilon=0.02, outer_iters=10, surrogate_steps=2, pgd_steps=3, reset_every=5, surrogate_lr=1e-4, pgd_lr=0.01
    ),
    "L2_PORTFOLIO": ASPLPreset(
        epsilon=0.04, outer_iters=20, surrogate_steps=3, pgd_steps=5, reset_every=8, surrogate_lr=1e-4, pgd_lr=0.01
    ),
    "L3_ANTI_TRAIN": ASPLPreset(
        epsilon=0.08, outer_iters=30, surrogate_steps=3, pgd_steps=6, reset_every=10, surrogate_lr=1e-4, pgd_lr=0.01
    ),
}


class ASPLAttacker:
    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, torch_dtype=dtype, safety_checker=None)
        self.vae = pipe.vae.to(device).eval()
        self.base_unet = pipe.unet.to(device).eval()
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.tokenizer = pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.device = device
        self.dtype = dtype

        for module in (self.vae, self.base_unet, self.text_encoder):
            for param in module.parameters():
                param.requires_grad_(False)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.text_encoder(tokens)[0]

    def build_surrogate(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        surrogate = get_peft_model(self.base_unet, lora_config)
        return surrogate

    def denoising_loss(self, unet, x: torch.Tensor, text_embeddings: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        latents = self.vae.encode(x.to(self.dtype) * 2 - 1).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        timestep = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device, generator=generator
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample
        return F.mse_loss(noise_pred.float(), noise.float())


def _reinit_lora_params(surrogate) -> None:
    """Fresh random re-init of the LoRA adapter weights in place, instead of
    rebuilding the whole peft wrapper (which would mean re-wrapping the
    shared base UNet module tree again) -- cheaper and avoids accumulating
    peft wrapper layers across resets."""
    import math

    for name, param in surrogate.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name:
            torch.nn.init.zeros_(param)


def aspl_attack(
    original_path: str,
    checkpoint_path: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    size: int = 512,
    seed: int = 0,
    perceptual_mask: bool = False,
    mask_low: float = 0.3,
    mask_high: float = 1.7,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = ASPL_PRESETS[preset_name]
    attacker = ASPLAttacker(checkpoint_path, device)
    text_embeddings = attacker.encode_prompt(prompt)
    generator = torch.Generator(device=device).manual_seed(seed)

    original = load_image_tensor(original_path, size, device).to(attacker.dtype)
    delta = torch.zeros_like(original, requires_grad=True)

    # Same redistribution style_cloak.py's cloak() already uses (see
    # compute_perceptual_mask's doc): spend the *same total* epsilon budget
    # unevenly -- more in already-textured regions a human doesn't scrutinize,
    # less in flat regions (sky, skin, background) where noise is most
    # visible -- instead of a flat clamp everywhere. Not tried on the ASPL
    # attacks before this (only style_cloak's own Gram-matrix attack used
    # it); the goal here is to see whether the SD1.5-then-SDXL chained
    # attack's real distortion (PHASE4_SCOPING.md SS6, [[strong-protection-
    # visual-honesty]]) can look less destructive at the same nominal
    # epsilon, not just by shrinking epsilon (which trades away signal
    # directly, and hasn't been shown to be *necessary* the way redistributing
    # the same budget might not be).
    epsilon_budget = compute_perceptual_mask(original, mask_low, mask_high) * preset.epsilon if perceptual_mask else preset.epsilon

    surrogate = attacker.build_surrogate()
    lora_params = [p for p in surrogate.parameters() if p.requires_grad]

    print(
        f"[aspl_attack] preset={preset_name} epsilon={preset.epsilon} outer_iters={preset.outer_iters} "
        f"surrogate_steps={preset.surrogate_steps} pgd_steps={preset.pgd_steps} reset_every={preset.reset_every}"
        f"{' perceptual_mask=on mask_low=' + str(mask_low) + ' mask_high=' + str(mask_high) if perceptual_mask else ''}"
    )

    surrogate_loss_val = float("nan")
    pgd_loss_val = float("nan")
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            _reinit_lora_params(surrogate)

        # (a) briefly fine-tune the surrogate on the current perturbed image
        surrogate.train()
        for p in lora_params:
            p.requires_grad_(True)
        surrogate_opt = torch.optim.Adam(lora_params, lr=preset.surrogate_lr)
        x_adv = (original + delta).clamp(0, 1).detach()
        for _ in range(preset.surrogate_steps):
            surrogate_opt.zero_grad()
            loss = attacker.denoising_loss(surrogate, x_adv, text_embeddings, generator)
            loss.backward()
            surrogate_opt.step()
            surrogate_loss_val = loss.item()

        # (b) PGD-update delta against the just-trained (now frozen) surrogate
        surrogate.eval()
        for p in lora_params:
            p.requires_grad_(False)
        pgd_opt = torch.optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()
            x_adv = (original + delta).clamp(0, 1)
            loss = attacker.denoising_loss(surrogate, x_adv, text_embeddings, generator)
            (-loss).backward()
            pgd_opt.step()
            pgd_loss_val = loss.item()

            with torch.no_grad():
                delta.clamp_(-epsilon_budget, epsilon_budget)
                delta.copy_(((original + delta).clamp(0, 1) - original))

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            print(f"  outer {outer:3d}  surrogate_loss(after ft)={surrogate_loss_val:.6f}  pgd_loss(after atk)={pgd_loss_val:.6f}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv.float(), output_path)
    print(f"[aspl_attack] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=list(ASPL_PRESETS.keys()))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perceptual-mask", action="store_true")
    parser.add_argument("--mask-low", type=float, default=0.3)
    parser.add_argument("--mask-high", type=float, default=1.7)
    args = parser.parse_args()

    aspl_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        size=args.size,
        seed=args.seed,
        perceptual_mask=args.perceptual_mask,
        mask_low=args.mask_low,
        mask_high=args.mask_high,
    )
