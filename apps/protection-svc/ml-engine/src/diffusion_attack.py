"""Anti-DreamBooth-style diffusion training-loss attack (PROJECT_DESIGN.md
§3-3's protection layers don't name this one explicitly -- it's a fourth,
new mechanism, not layer [2] Style Confusion or layer [3] Concept
Misalignment, both already validated against real LoRA training and found
weak/negligible -- see PHASE4_SCOPING.md §1 and experiments/lora_validation/
and experiments/concept_misalignment_validation/'s own READMEs).

Same optimization *shape* as style_cloak.py/concept_misalign.py --

    maximize   the diffusion model's own denoising training loss
               (predicted noise vs actual noise, at a random timestep)
    subject to Perceptual_Distance < epsilon (bounded pixel-space
               perturbation, image looks unchanged to a human)

-- but attacks the actual thing a LoRA/DreamBooth trainer minimizes,
instead of a proxy feature space (VGG19 Gram matrices for style_cloak,
CLIP image embeddings for concept_misalign). This is the mechanism real
academic anti-personalization work (Anti-DreamBooth, Van Le et al. 2023;
PhotoGuard's "diffusion attack" variant) uses for exactly this threat
model -- an attacker fine-tuning a LoRA/DreamBooth model on scraped
images -- rather than a metric merely correlated with it.

**Honest scope, matching this project's practice of not overclaiming**:
this is a *simplified* single-surrogate variant, not full Anti-DreamBooth.
The real ASPL (Alternating Surrogate and Perturbation Learning) method
alternates between (a) briefly fine-tuning a small surrogate DreamBooth/
LoRA model on the current perturbed image and (b) using THAT just-
fine-tuned surrogate's gradient to update the perturbation -- repeated
many times, so the attack tracks how the model's weights would actually
shift during real training. This module skips the alternating surrogate
retraining and instead attacks the FROZEN base checkpoint's own
denoising loss directly (no per-image surrogate training loop) -- much
cheaper (one forward/backward through VAE+UNet per step, the same cost
class as a single real training step, not N training runs per PGD step),
but a real, meaningfully different claim than the untrained-surrogate
case: whether attacking the base model's own loss landscape (which any
LoRA starts training from) transfers into actually degrading a
subsequently-trained LoRA is exactly what this module's own validation
experiment (experiments/diffusion_attack_validation/) needs to answer --
same "mechanism ≠ proven effect" caveat concept_misalign.py's own
module doc carries until its real LoRA-training validation ran.

Usage:
    python src/diffusion_attack.py --original out/original.png \\
        --checkpoint path/to/v1-5-pruned-emaonly-fp16.safetensors \\
        --prompt "a photo of sks person" --preset L3_ANTI_TRAIN
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from style_cloak import load_image_tensor, random_resize_round_trip, save_tensor_image


@dataclass
class DiffusionAttackPreset:
    """Same shape as style_cloak.Preset/concept_misalign.ConceptPreset --
    epsilon is the L-infinity pixel-space perturbation budget, steps is the
    PGD iteration count. Kept as its own table (not shared with the other
    two) since the denoising-loss landscape has a different scale/gradient
    behavior than VGG19 Gram-matrix or CLIP-embedding loss and was never
    jointly tuned with either -- these starting points mirror the other
    two presets' own epsilon/step shape as an untuned starting guess, the
    same honest position concept_misalign.py's own CONCEPT_PRESETS started
    from before its real validation run.
    """

    epsilon: float
    steps: int
    lr: float


DIFFUSION_ATTACK_PRESETS = {
    "L1_PREVIEW": DiffusionAttackPreset(epsilon=0.02, steps=150, lr=0.01),
    "L2_PORTFOLIO": DiffusionAttackPreset(epsilon=0.04, steps=300, lr=0.01),
    "L3_ANTI_TRAIN": DiffusionAttackPreset(epsilon=0.08, steps=500, lr=0.01),
}


class DiffusionLossAttacker:
    """Wraps a frozen SD1.x pipeline's VAE + UNet + text encoder for
    computing the real per-step training loss a DreamBooth/LoRA trainer
    would minimize, and exposes just enough for style_cloak.py-style PGD
    against it. Loads once per process (mirrors StyleFeatureExtractor's own
    module-level-cache pattern in model.py) -- these are the same three
    frozen sub-models every LoRA training run starts from, not a new
    concept.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, torch_dtype=dtype, safety_checker=None)
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.tokenizer = pipe.tokenizer
        # DDPMScheduler explicitly, not whatever inference-oriented
        # scheduler from_single_file happened to attach (often PNDM/DPM-
        # Solver, tuned for fast sampling, not the training forward-noising
        # process) -- this is the actual scheduler class kohya_ss's own
        # train_network.py uses for add_noise() during real training, so
        # this attack targets the same noising process a real trainer's
        # loss actually sees.
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.device = device
        self.dtype = dtype

        for module in (self.vae, self.unet, self.text_encoder):
            for param in module.parameters():
                param.requires_grad_(False)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.text_encoder(tokens)[0]

    def denoising_loss(self, x_adv: torch.Tensor, text_embeddings: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """The exact loss train_network.py/sdxl_train_network.py minimize
        for a real training step: encode to the VAE latent, add noise at a
        random timestep, ask the UNet to predict that noise, MSE against
        the real noise. x_adv must be a [0,1]-range tensor with
        requires_grad already set on whatever leaf it derives from (the
        `delta` PGD is optimizing) -- gradients flow back through the VAE
        encoder into pixel space the same way they'd flow through VGG19/
        CLIP in the other two attacks.
        """
        latents = self.vae.encode(x_adv.to(self.dtype) * 2 - 1).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        timestep = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device, generator=generator
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        noise_pred = self.unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample
        return F.mse_loss(noise_pred.float(), noise.float())


def attack(
    original_path: str,
    checkpoint_path: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    size: int = 512,
    eot: bool = False,
    eot_samples: int = 2,
    eot_min_scale: float = 0.3,
    eot_max_scale: float = 1.0,
    seed: int = 0,
) -> None:
    """Optimizes `original`'s pixels (within an epsilon ball) to MAXIMIZE
    the frozen base checkpoint's own denoising loss for `prompt` -- the
    opposite direction a real LoRA/DreamBooth trainer's gradient descent
    would push, so a model that starts training from this image is
    fighting a worse initial loss landscape for it than the unperturbed
    original. See module doc for the "simplified, no alternating surrogate
    retraining" scope caveat.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = DIFFUSION_ATTACK_PRESETS[preset_name]
    attacker = DiffusionLossAttacker(checkpoint_path, device)
    text_embeddings = attacker.encode_prompt(prompt)
    generator = torch.Generator(device=device).manual_seed(seed)

    original = load_image_tensor(original_path, size, device).to(attacker.dtype)

    delta = torch.zeros_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=preset.lr)

    mode = f"EOT(resize, samples={eot_samples}, scale=[{eot_min_scale},{eot_max_scale}])" if eot else "no-EOT"
    print(f"[diffusion_attack] preset={preset_name} epsilon={preset.epsilon} steps={preset.steps} mode={mode}")
    for step in range(preset.steps):
        optimizer.zero_grad()
        x_adv = (original + delta).clamp(0, 1)

        if eot:
            loss = attacker.denoising_loss(x_adv, text_embeddings, generator)
            for _ in range(eot_samples):
                transformed = random_resize_round_trip(x_adv, eot_min_scale, eot_max_scale)
                loss = loss + attacker.denoising_loss(transformed, text_embeddings, generator)
            loss = loss / (eot_samples + 1)
        else:
            loss = attacker.denoising_loss(x_adv, text_embeddings, generator)

        # Gradient ASCENT on the denoising loss (we want training to start
        # from a WORSE loss for this image, not a better one) -- Adam
        # minimizes by default, so negate here rather than hand-rolling a
        # separate ascent optimizer.
        (-loss).backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-preset.epsilon, preset.epsilon)
            delta.copy_(((original + delta).clamp(0, 1) - original))

        if step % 50 == 0 or step == preset.steps - 1:
            print(f"  step {step:4d}  denoising_loss={loss.item():.6f}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv.float(), output_path)
    print(f"[diffusion_attack] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="out/original.png")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="out/diffusion_attacked.png")
    parser.add_argument("--preset", choices=list(DIFFUSION_ATTACK_PRESETS), default="L3_ANTI_TRAIN")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--eot", action="store_true")
    parser.add_argument("--eot-samples", type=int, default=2)
    parser.add_argument("--eot-min-scale", type=float, default=0.3)
    parser.add_argument("--eot-max-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    attack(
        args.original,
        args.checkpoint,
        args.prompt,
        args.output,
        args.preset,
        size=args.size,
        eot=args.eot,
        eot_samples=args.eot_samples,
        eot_min_scale=args.eot_min_scale,
        eot_max_scale=args.eot_max_scale,
        seed=args.seed,
    )
