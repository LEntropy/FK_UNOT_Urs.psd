"""SDXL counterpart to aspl_attack_latent.py -- ASPL attack with the
perturbation defined in VAE latent space instead of pixel space. See
aspl_attack_latent.py's module doc for the full rationale (pixel-space
epsilon doesn't respect the "stays on the manifold of real images"
constraint a latent-space epsilon naturally gets from the VAE decoder).

Reuses SDXLBranch from ensemble_attack_multiarch.py (same checkpoint
loading / dual-text-encoder prompt encoding / added_cond_kwargs plumbing
aspl_attack_sdxl_only.py already reuses) -- only the denoising-loss/
perturbation code is new, working on latents directly.
"""

import argparse

import torch
import torch.nn.functional as F

from aspl_attack_sdxl_only import SDXL_ONLY_PRESETS, _lora_params, _reinit_lora
from ensemble_attack_multiarch import SDXLBranch
from style_cloak import load_image_tensor, save_tensor_image


def denoising_loss_from_latent(branch: SDXLBranch, unet, latents: torch.Tensor, cond: dict, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(latents.shape, generator=generator, device=branch.device, dtype=branch.dtype)
    timestep = torch.randint(
        0, branch.scheduler.config.num_train_timesteps, (latents.shape[0],), device=branch.device, generator=generator
    ).long()
    noisy_latents = branch.scheduler.add_noise(latents, noise, timestep)
    noise_pred = unet(
        noisy_latents, timestep,
        encoder_hidden_states=cond["encoder_hidden_states"],
        added_cond_kwargs=cond["added_cond_kwargs"],
    ).sample
    return F.mse_loss(noise_pred.float(), noise.float())


def aspl_attack_sdxl_only_latent(
    original_path: str,
    sdxl_checkpoint: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    seed: int = 0,
    latent_epsilon: float = 0.3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = SDXL_ONLY_PRESETS[preset_name]
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[aspl_attack_sdxl_only_latent] loading SDXL branch ({sdxl_checkpoint})")
    branch = SDXLBranch(sdxl_checkpoint, device, dtype)
    cond = branch.encode_prompt(prompt, preset.size)
    surrogate = branch.build_surrogate(preset.lora_r)

    original = load_image_tensor(original_path, preset.size, device).to(dtype)
    with torch.no_grad():
        z0 = branch.vae.encode(original * 2 - 1).latent_dist.sample() * branch.vae.config.scaling_factor
    delta = torch.zeros_like(z0, requires_grad=True)

    print(
        f"[aspl_attack_sdxl_only_latent] preset={preset_name} latent_epsilon={latent_epsilon} "
        f"outer_iters={preset.outer_iters} surrogate_steps={preset.surrogate_steps} "
        f"pgd_steps={preset.pgd_steps} reset_every={preset.reset_every} "
        f"(z0 stats: mean={z0.mean().item():.4f} std={z0.std().item():.4f})"
    )

    loss_val = float("nan")
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            _reinit_lora(surrogate)

        surrogate.train()
        params = _lora_params(surrogate)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=preset.surrogate_lr)
        z_adv_detached = (z0 + delta).detach()
        for _ in range(preset.surrogate_steps):
            opt.zero_grad()
            loss = denoising_loss_from_latent(branch, surrogate, z_adv_detached, cond, generator)
            loss.backward()
            opt.step()
        for p in params:
            p.requires_grad_(False)
        surrogate.eval()

        pgd_opt = torch.optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()
            z_adv = z0 + delta
            loss = denoising_loss_from_latent(branch, surrogate, z_adv, cond, generator)
            (-loss).backward()
            pgd_opt.step()
            loss_val = loss.item()

            with torch.no_grad():
                delta.clamp_(-latent_epsilon, latent_epsilon)

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            print(f"  outer {outer:3d}  sdxl_loss={loss_val:.6f}")

    with torch.no_grad():
        z_final = z0 + delta
        x_final = branch.vae.decode(z_final / branch.vae.config.scaling_factor).sample
        x_final = (x_final / 2 + 0.5).clamp(0, 1)
    save_tensor_image(x_final, output_path)
    print(f"[aspl_attack_sdxl_only_latent] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="SDXL_FULL", choices=list(SDXL_ONLY_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-epsilon", type=float, default=0.3)
    args = parser.parse_args()

    aspl_attack_sdxl_only_latent(
        original_path=args.original,
        sdxl_checkpoint=args.sdxl_checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        seed=args.seed,
        latent_epsilon=args.latent_epsilon,
    )
