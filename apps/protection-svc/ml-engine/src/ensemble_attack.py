"""Multi-surrogate ensemble ASPL attack with a relaxed epsilon budget.

Diagnosis from five prior negative results (style_cloak, concept_misalign,
diffusion_attack, aspl_attack, hybrid_attack -- see each module's own doc):
every one of them optimized against a SINGLE proxy (VGG19, CLIP, or one
rank-4 LoRA surrogate on one checkpoint) inside a tight epsilon<=0.08
budget chosen to stay visually invisible. Two things were never tried:

1. Ensemble surrogates. Attacking several different surrogate models at
   once is well documented in the classifier-adversarial-attack literature
   to transfer much better to an unseen target model than any single-
   surrogate attack -- the perturbation has to fool all of them
   simultaneously, which is a much harder (and more transferable)
   direction than fooling just one. This module runs N independently-
   initialized LoRA surrogates (different ranks / target-module subsets,
   so their random init and effective capacity differ) on the SAME frozen
   SD1.5 backbone via peft's multi-adapter support (loading N separate
   full checkpoints doesn't fit this GPU's 8GB VRAM budget -- LoRA-config
   diversity on a shared backbone is the feasible approximation of
   architecture diversity), retrains all of them every outer iteration
   (ASPL-style alternation, see aspl_attack.py), and PGD-attacks the SUM
   of their denoising losses jointly every step.

2. A relaxed epsilon. Every experiment so far stayed in the 0.02-0.08
   range specifically to keep the perturbation invisible. This preset
   raises it well past that (default 0.25) -- an explicitly non-invisible
   perturbation, testing whether a bigger, visible-cost budget produces a
   real effect at all, something no prior experiment in this project
   isolated as its own variable.

Runs in fp16 (not fp32 like aspl_attack.py/hybrid_attack.py) -- N
surrogates' combined optimizer state pushes this 8GB GPU's headroom past
where fp32 has previously proven safe (aspl_attack.py's single-surrogate
fp32 run measured as low as ~70MB free at points; adding more surrogates
without switching precision risks an OOM crash on a long unattended run).

Must run under kohya_ss's venv (diffusers + peft + CUDA).
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.optim as optim

from style_cloak import load_image_tensor, save_tensor_image


@dataclass
class EnsemblePreset:
    epsilon: float
    outer_iters: int
    surrogate_steps: int
    pgd_steps: int
    reset_every: int
    surrogate_lr: float
    pgd_lr: float
    lora_configs: list[tuple[int, list[str]]]


ENSEMBLE_PRESETS = {
    "L1_PREVIEW": EnsemblePreset(
        epsilon=0.1, outer_iters=5, surrogate_steps=2, pgd_steps=2, reset_every=3,
        surrogate_lr=1e-4, pgd_lr=0.02,
        lora_configs=[
            (4, ["to_q", "to_v"]),
            (8, ["to_q", "to_k", "to_v"]),
        ],
    ),
    "L4_ENSEMBLE_WIDE": EnsemblePreset(
        epsilon=0.25, outer_iters=30, surrogate_steps=3, pgd_steps=6, reset_every=10,
        surrogate_lr=1e-4, pgd_lr=0.02,
        lora_configs=[
            (4, ["to_q", "to_k", "to_v", "to_out.0"]),
            (8, ["to_q", "to_v"]),
            (16, ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"]),
        ],
    ),
}


class EnsembleAttacker:
    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype = torch.float16):
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

    def build_ensemble(self, lora_configs: list[tuple[int, list[str]]]):
        """One peft-wrapped model, N named LoRA adapters sharing the same
        frozen base UNet weights -- memory-cheap (only the small adapter
        matrices are duplicated per surrogate, not the whole UNet)."""
        from peft import LoraConfig, get_peft_model

        r0, tm0 = lora_configs[0]
        model = get_peft_model(self.base_unet, LoraConfig(r=r0, lora_alpha=r0, target_modules=tm0), adapter_name="s0")
        for i, (r, tm) in enumerate(lora_configs[1:], start=1):
            model.add_adapter(f"s{i}", LoraConfig(r=r, lora_alpha=r, target_modules=tm))
        return model

    def denoising_loss(self, unet, x: torch.Tensor, text_embeddings: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        latents = self.vae.encode(x.to(self.dtype) * 2 - 1).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        timestep = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device, generator=generator
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample
        return torch.nn.functional.mse_loss(noise_pred.float(), noise.float())


def _adapter_params(model, adapter_name: str) -> list[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if adapter_name in n and ("lora_A" in n or "lora_B" in n)]


def _reinit_adapter(model, adapter_name: str) -> None:
    for n, p in model.named_parameters():
        if adapter_name not in n:
            continue
        if "lora_A" in n:
            torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in n:
            torch.nn.init.zeros_(p)


def ensemble_attack(
    original_path: str,
    checkpoint_path: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    size: int = 512,
    seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = ENSEMBLE_PRESETS[preset_name]
    attacker = EnsembleAttacker(checkpoint_path, device)
    text_embeddings = attacker.encode_prompt(prompt)
    generator = torch.Generator(device=device).manual_seed(seed)

    original = load_image_tensor(original_path, size, device)
    delta = torch.zeros_like(original, requires_grad=True)

    ensemble = attacker.build_ensemble(preset.lora_configs)
    adapter_names = [f"s{i}" for i in range(len(preset.lora_configs))]

    print(
        f"[ensemble_attack] preset={preset_name} epsilon={preset.epsilon} outer_iters={preset.outer_iters} "
        f"n_surrogates={len(adapter_names)} configs={preset.lora_configs}"
    )

    losses_val = {name: float("nan") for name in adapter_names}
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            for name in adapter_names:
                _reinit_adapter(ensemble, name)

        # (a) briefly fine-tune every surrogate on the current perturbed image
        x_adv_detached = (original + delta).clamp(0, 1).detach()
        for name in adapter_names:
            ensemble.set_adapter(name)
            ensemble.train()
            params = _adapter_params(ensemble, name)
            for p in params:
                p.requires_grad_(True)
            opt = optim.Adam(params, lr=preset.surrogate_lr)
            for _ in range(preset.surrogate_steps):
                opt.zero_grad()
                loss = attacker.denoising_loss(ensemble, x_adv_detached, text_embeddings, generator)
                loss.backward()
                opt.step()
            for p in params:
                p.requires_grad_(False)

        # (b) PGD-minimize the SUM of all surrogates' (negated) denoising
        # loss jointly -- one perturbation has to fool every surrogate at once
        ensemble.eval()
        pgd_opt = optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()
            x_adv = (original + delta).clamp(0, 1)

            total = torch.zeros((), device=device, dtype=torch.float32)
            for name in adapter_names:
                ensemble.set_adapter(name)
                d_loss = attacker.denoising_loss(ensemble, x_adv, text_embeddings, generator)
                losses_val[name] = d_loss.item()
                total = total - d_loss  # ascent on every surrogate's loss

            total.backward()
            pgd_opt.step()

            with torch.no_grad():
                delta.clamp_(-preset.epsilon, preset.epsilon)
                delta.copy_(((original + delta).clamp(0, 1) - original))

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            losses_str = " ".join(f"{n}={v:.4f}" for n, v in losses_val.items())
            print(f"  outer {outer:3d}  {losses_str}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[ensemble_attack] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="L4_ENSEMBLE_WIDE", choices=list(ENSEMBLE_PRESETS.keys()))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensemble_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        size=args.size,
        seed=args.seed,
    )
