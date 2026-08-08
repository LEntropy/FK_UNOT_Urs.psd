"""ASPL attack with the perturbation defined in VAE LATENT space instead of
pixel space -- a genuinely different attack surface from every other
mechanism in this project's history (all 10 prior mechanisms, see
[[lora-protection-research]], bound delta as an L-infinity ball on raw RGB
pixels; this project has never attacked in latent space before).

Why this might matter: a pixel-space epsilon ball has no notion of "stays
on the manifold of images that look like real art" -- pushed hard enough
to survive real training (this project's own finding: small pixel epsilon
doesn't validate), the perturbation reads as structured-but-wrong texture
(the swirl/paisley pattern seen in aspl_attack.py's chained output is very
plausibly the base model's own learned visual vocabulary bleeding through,
since the gradient IS computed through that model). A latent-space epsilon
ball is different: nearby points in a trained VAE's latent space decode to
*other plausible images* (this is the same reason a low-strength img2img
pass changes fine detail while staying recognizably close to the input --
it's walking a short distance through the same latent manifold). The
hypothesis: the same nominal perturbation "budget," spent in latent space
and decoded back to pixels, should read as far more natural/subtle than
the same budget spent directly on RGB, for a comparable protective effect.

Untested until this pilot (see PHASE4_SCOPING.md SS6 / [[strong-protection-
visual-honesty]]'s follow-up) -- purely exploratory, not yet validated at
any scale. Reuses ASPLAttacker from aspl_attack.py (same checkpoint
loading, encode_prompt, build_surrogate) rather than reimplementing it;
only the denoising-loss/perturbation plumbing is new (works on latents
directly, skipping the redundant per-step VAE encode aspl_attack.py's own
denoising_loss does -- this variant only decodes once, at the very end).
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aspl_attack import ASPL_PRESETS, ASPLAttacker, _reinit_lora_params
from style_cloak import load_image_tensor, save_tensor_image


def denoising_loss_from_latent(attacker: ASPLAttacker, unet, latents: torch.Tensor, text_embeddings: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Same denoising-loss objective as ASPLAttacker.denoising_loss, but
    takes an already-encoded latent directly instead of a pixel tensor --
    the perturbation being optimized lives here, not in pixel space, so
    there's no VAE encode to redo every call (only decode, once, at the
    very end of the attack loop)."""
    noise = torch.randn(latents.shape, generator=generator, device=attacker.device, dtype=attacker.dtype)
    timestep = torch.randint(
        0, attacker.scheduler.config.num_train_timesteps, (latents.shape[0],), device=attacker.device, generator=generator
    ).long()
    noisy_latents = attacker.scheduler.add_noise(latents, noise, timestep)
    noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample
    return F.mse_loss(noise_pred.float(), noise.float())


def aspl_attack_latent(
    original_path: str,
    checkpoint_path: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    size: int = 512,
    seed: int = 0,
    latent_epsilon: float = 0.3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = ASPL_PRESETS[preset_name]
    attacker = ASPLAttacker(checkpoint_path, device)
    text_embeddings = attacker.encode_prompt(prompt)
    generator = torch.Generator(device=device).manual_seed(seed)

    original = load_image_tensor(original_path, size, device).to(attacker.dtype)
    with torch.no_grad():
        z0 = attacker.vae.encode(original * 2 - 1).latent_dist.sample() * attacker.vae.config.scaling_factor
    delta = torch.zeros_like(z0, requires_grad=True)

    surrogate = attacker.build_surrogate()
    lora_params = [p for p in surrogate.parameters() if p.requires_grad]

    print(
        f"[aspl_attack_latent] preset={preset_name} latent_epsilon={latent_epsilon} outer_iters={preset.outer_iters} "
        f"surrogate_steps={preset.surrogate_steps} pgd_steps={preset.pgd_steps} reset_every={preset.reset_every} "
        f"(z0 stats: mean={z0.mean().item():.4f} std={z0.std().item():.4f})"
    )

    surrogate_loss_val = float("nan")
    pgd_loss_val = float("nan")
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            _reinit_lora_params(surrogate)

        # (a) briefly fine-tune the surrogate on the current perturbed latent
        surrogate.train()
        for p in lora_params:
            p.requires_grad_(True)
        surrogate_opt = torch.optim.Adam(lora_params, lr=preset.surrogate_lr)
        z_adv = (z0 + delta).detach()
        for _ in range(preset.surrogate_steps):
            surrogate_opt.zero_grad()
            loss = denoising_loss_from_latent(attacker, surrogate, z_adv, text_embeddings, generator)
            loss.backward()
            surrogate_opt.step()
            surrogate_loss_val = loss.item()

        # (b) PGD-update delta (in latent space) against the just-trained surrogate
        surrogate.eval()
        for p in lora_params:
            p.requires_grad_(False)
        pgd_opt = torch.optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()
            z_adv = z0 + delta
            loss = denoising_loss_from_latent(attacker, surrogate, z_adv, text_embeddings, generator)
            (-loss).backward()
            pgd_opt.step()
            pgd_loss_val = loss.item()

            with torch.no_grad():
                delta.clamp_(-latent_epsilon, latent_epsilon)

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            print(f"  outer {outer:3d}  surrogate_loss(after ft)={surrogate_loss_val:.6f}  pgd_loss(after atk)={pgd_loss_val:.6f}")

    with torch.no_grad():
        z_final = z0 + delta
        x_final = attacker.vae.decode(z_final / attacker.vae.config.scaling_factor).sample
        x_final = (x_final / 2 + 0.5).clamp(0, 1)
    save_tensor_image(x_final.float(), output_path)
    print(f"[aspl_attack_latent] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=list(ASPL_PRESETS.keys()))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-epsilon", type=float, default=0.3)
    args = parser.parse_args()

    aspl_attack_latent(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        size=args.size,
        seed=args.seed,
        latent_epsilon=args.latent_epsilon,
    )
