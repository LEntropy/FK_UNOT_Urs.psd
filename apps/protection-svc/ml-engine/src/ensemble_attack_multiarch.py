"""Real multi-architecture ensemble ASPL attack: SD1.5 + SDXL as genuine
separate surrogates, not the LoRA-config diversity on one shared SD1.5
backbone that ensemble_attack.py used as an 8GB-VRAM-forced approximation
of architecture diversity (see that module's own docstring: "loading N
separate full checkpoints doesn't fit this GPU's 8GB VRAM budget"). Holding
both full backbones plus both surrogates' optimizer state simultaneously
needs far more headroom than the project's original 8GB GPU had -- this is
the thing the RunPod A40 (48GB) migration was specifically for.

Mechanically this is the same ASPL alternation as aspl_attack.py/
ensemble_attack.py -- (a) briefly fine-tune each architecture's surrogate
LoRA on the current perturbed image, (b) PGD-update one shared perturbation
against the SUM of both architectures' denoising losses -- generalized to
two backbones with genuinely different forward-pass signatures:

- SD1.5: single CLIP text encoder, unet(latents, t, encoder_hidden_states=...)
- SDXL: two text encoders (concatenated hidden states + text_encoder_2's
  pooled output), unet(..., added_cond_kwargs={"text_embeds", "time_ids"}),
  and normally operates at 1024px vs SD1.5's 512px.

The perturbation `delta` is parameterized at SDXL's native resolution (the
larger of the two); the SD1.5 branch bilinearly downsamples (original+delta)
to 512px before its own VAE/UNet forward pass -- differentiable, so PGD
gradients flow back through the resize into the shared delta.

Runs in fp32 (unlike ensemble_attack.py's fp16, which was a similar 8GB
VRAM compromise for N surrogates' combined optimizer state) since the A40
has enough headroom for both full backbones at full precision.

Must run under kohya_ss's venv (diffusers + peft + CUDA) with both a SD1.5
and an SDXL checkpoint available locally (StableDiffusionPipeline.
from_single_file / StableDiffusionXLPipeline.from_single_file).
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

from style_cloak import load_image_tensor, save_tensor_image


@dataclass
class MultiArchPreset:
    epsilon: float
    outer_iters: int
    surrogate_steps: int
    pgd_steps: int
    reset_every: int
    surrogate_lr: float
    pgd_lr: float
    sd15_size: int
    sdxl_size: int
    lora_r: int


MULTIARCH_PRESETS = {
    # Calibration-scale: same iteration counts as aspl_attack.py's
    # L1_PREVIEW, just with two real backbones instead of one, to time a
    # single (image, seed) unit before committing to the full-scale preset.
    "CALIBRATION": MultiArchPreset(
        epsilon=0.08, outer_iters=10, surrogate_steps=2, pgd_steps=3, reset_every=5,
        surrogate_lr=1e-4, pgd_lr=0.01, sd15_size=512, sdxl_size=1024, lora_r=4,
    ),
    # Full spec: matches aspl_attack.py's L3_ANTI_TRAIN iteration counts
    # exactly (outer_iters=30, surrogate_steps=3, pgd_steps=6, reset_every=10)
    # -- those counts had no documented GPU-driven reduction, only the
    # architecture-diversity + fp32 precision did, so this preset changes
    # only what actually was constrained.
    "MULTIARCH_FULL": MultiArchPreset(
        epsilon=0.08, outer_iters=30, surrogate_steps=3, pgd_steps=6, reset_every=10,
        surrogate_lr=1e-4, pgd_lr=0.01, sd15_size=512, sdxl_size=1024, lora_r=4,
    ),
}


class SD15Branch:
    arch = "sd15"

    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, torch_dtype=dtype, safety_checker=None)
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.unet.enable_gradient_checkpointing()
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.tokenizer = pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.device = device
        self.dtype = dtype
        for module in (self.vae, self.unet, self.text_encoder):
            for p in module.parameters():
                p.requires_grad_(False)

    def encode_prompt(self, prompt: str):
        tokens = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            return {"encoder_hidden_states": self.text_encoder(tokens)[0]}

    def build_surrogate(self, lora_r: int):
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        return get_peft_model(self.unet, cfg)

    def denoising_loss(self, unet, x: torch.Tensor, cond: dict, generator: torch.Generator) -> torch.Tensor:
        latents = self.vae.encode(x.to(self.dtype) * 2 - 1).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        timestep = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device, generator=generator
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=cond["encoder_hidden_states"]).sample
        return F.mse_loss(noise_pred.float(), noise.float())


class SDXLBranch:
    arch = "sdxl"

    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype):
        from diffusers import DDPMScheduler, StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_path, torch_dtype=dtype, safety_checker=None)
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.unet.enable_gradient_checkpointing()
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.text_encoder_2 = pipe.text_encoder_2.to(device).eval()
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.device = device
        self.dtype = dtype
        for module in (self.vae, self.unet, self.text_encoder, self.text_encoder_2):
            for p in module.parameters():
                p.requires_grad_(False)

    def _encode_one(self, tokenizer, text_encoder, prompt: str):
        tokens = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            out = text_encoder(tokens, output_hidden_states=True)
            # SDXL uses the penultimate hidden state, not the final layer output.
            hidden = out.hidden_states[-2]
            pooled = out[0]
        return hidden, pooled

    def encode_prompt(self, prompt: str, size: int):
        hidden_1, _ = self._encode_one(self.tokenizer, self.text_encoder, prompt)
        hidden_2, pooled_2 = self._encode_one(self.tokenizer_2, self.text_encoder_2, prompt)
        encoder_hidden_states = torch.cat([hidden_1, hidden_2], dim=-1)
        # No cropping/upscaling in this pipeline -- original size == target size, crop at (0, 0).
        time_ids = torch.tensor([[size, size, 0, 0, size, size]], device=self.device, dtype=self.dtype)
        return {
            "encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": {"text_embeds": pooled_2, "time_ids": time_ids},
        }

    def build_surrogate(self, lora_r: int):
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        return get_peft_model(self.unet, cfg)

    def denoising_loss(self, unet, x: torch.Tensor, cond: dict, generator: torch.Generator) -> torch.Tensor:
        latents = self.vae.encode(x.to(self.dtype) * 2 - 1).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        timestep = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device, generator=generator
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        noise_pred = unet(
            noisy_latents, timestep,
            encoder_hidden_states=cond["encoder_hidden_states"],
            added_cond_kwargs=cond["added_cond_kwargs"],
        ).sample
        return F.mse_loss(noise_pred.float(), noise.float())


def _lora_params(surrogate) -> list:
    # Filter by parameter name only, not current requires_grad state -- the
    # PGD phase deliberately sets these params' requires_grad to False each
    # outer iteration, so filtering on requires_grad here would find nothing
    # on every iteration after the first (the actual bug hit in calibration
    # run #2: "optimizer got an empty parameter list" on outer=1).
    return [p for n, p in surrogate.named_parameters() if "lora_A" in n or "lora_B" in n]


def _reinit_lora(surrogate) -> None:
    for name, param in surrogate.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name:
            torch.nn.init.zeros_(param)


def multiarch_ensemble_attack(
    original_path: str,
    sd15_checkpoint: str,
    sdxl_checkpoint: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = MULTIARCH_PRESETS[preset_name]
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[multiarch_ensemble_attack] loading SD1.5 branch ({sd15_checkpoint})")
    sd15 = SD15Branch(sd15_checkpoint, device, dtype)
    print(f"[multiarch_ensemble_attack] loading SDXL branch ({sdxl_checkpoint})")
    sdxl = SDXLBranch(sdxl_checkpoint, device, dtype)

    branches = {"sd15": sd15, "sdxl": sdxl}
    sizes = {"sd15": preset.sd15_size, "sdxl": preset.sdxl_size}
    cond = {
        "sd15": sd15.encode_prompt(prompt),
        "sdxl": sdxl.encode_prompt(prompt, preset.sdxl_size),
    }
    surrogates = {name: b.build_surrogate(preset.lora_r) for name, b in branches.items()}

    # delta is parameterized at SDXL's (larger) native resolution; the SD1.5
    # branch bilinearly downsamples for its own forward pass -- differentiable.
    original = load_image_tensor(original_path, preset.sdxl_size, device).to(dtype)
    delta = torch.zeros_like(original, requires_grad=True)

    print(
        f"[multiarch_ensemble_attack] preset={preset_name} epsilon={preset.epsilon} "
        f"outer_iters={preset.outer_iters} surrogate_steps={preset.surrogate_steps} "
        f"pgd_steps={preset.pgd_steps} reset_every={preset.reset_every} archs={list(branches)}"
    )

    losses_val = {name: float("nan") for name in branches}
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            for name in branches:
                _reinit_lora(surrogates[name])

        x_adv_detached = (original + delta).clamp(0, 1).detach()
        for name, branch in branches.items():
            surrogate = surrogates[name]
            surrogate.train()
            params = _lora_params(surrogate)
            for p in params:
                p.requires_grad_(True)
            opt = optim.Adam(params, lr=preset.surrogate_lr)
            x_in = x_adv_detached if name == "sdxl" else F.interpolate(
                x_adv_detached, size=(sizes[name], sizes[name]), mode="bilinear", align_corners=False
            )
            for _ in range(preset.surrogate_steps):
                opt.zero_grad()
                loss = branch.denoising_loss(surrogate, x_in, cond[name], generator)
                loss.backward()
                opt.step()
            for p in params:
                p.requires_grad_(False)
            surrogate.eval()

        pgd_opt = optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()

            # Backward per-branch immediately (accumulating into delta.grad)
            # instead of summing both losses before one combined backward --
            # keeping both branches' forward-pass activation graphs (512px
            # SD1.5 + 1024px SDXL, fp32) alive simultaneously was the actual
            # OOM cause on the A40's 44GiB (measured: 44.33/44.34 GiB used
            # right before the crash). Sequential backward frees each
            # branch's graph before the next branch's forward pass runs.
            # x_adv is recomputed per branch (not shared) so each branch's
            # backward call has its own independent path back to delta --
            # sharing one x_adv node across two backward() calls would hit
            # "Trying to backward through the graph a second time" once the
            # first call frees that shared node's buffers.
            for name, branch in branches.items():
                x_adv = (original + delta).clamp(0, 1)
                x_in = x_adv if name == "sdxl" else F.interpolate(
                    x_adv, size=(sizes[name], sizes[name]), mode="bilinear", align_corners=False
                )
                d_loss = branch.denoising_loss(surrogates[name], x_in, cond[name], generator)
                losses_val[name] = d_loss.item()
                (-d_loss).backward()  # ascent on this branch's loss; grad accumulates into delta.grad

            pgd_opt.step()

            with torch.no_grad():
                delta.clamp_(-preset.epsilon, preset.epsilon)
                delta.copy_(((original + delta).clamp(0, 1) - original))

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            losses_str = " ".join(f"{n}={v:.4f}" for n, v in losses_val.items())
            print(f"  outer {outer:3d}  {losses_str}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[multiarch_ensemble_attack] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--sd15-checkpoint", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="MULTIARCH_FULL", choices=list(MULTIARCH_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    multiarch_ensemble_attack(
        original_path=args.original,
        sd15_checkpoint=args.sd15_checkpoint,
        sdxl_checkpoint=args.sdxl_checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        seed=args.seed,
    )
