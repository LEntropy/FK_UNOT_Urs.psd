"""Hybrid ensemble cloak: combines every mechanism this project has
validated so far into one joint objective, plus ASPL's alternating
surrogate retrain -- an ensemble-transferability attempt, not yet tried.

Individually validated on the same 5-painting, real-LoRA-retraining
benchmark, none reliably degrades a LoRA someone else trains from scratch
on the resulting image (every 95% CI includes zero):
    style_cloak      (VGG19 Gram-matrix drift):     n=10  delta +0.0113
    concept_misalign (CLIP embedding drift):         n=15  delta ~0
    diffusion_attack (frozen-checkpoint denoising):  n=10  delta +0.0071
    aspl_attack      (surrogate-retrain denoising):  n=10  delta +0.0050

Ensemble adversarial attacks are well documented in the classifier-attack
literature to transfer better than any single-objective attack -- a
direction that's simultaneously adversarial across several different
feature spaces is less likely to be an artifact of one specific extractor.
Nothing in this project has combined objectives before; every mechanism
above was validated in isolation. This module combines all three loss
signals -- VGG19 style drift, CLIP concept drift, and (ASPL-style)
denoising-loss maximization against a periodically-retrained surrogate
LoRA -- into a single perturbation, optimized jointly every PGD step:

    total_loss = w_style   * style_loss(x_adv, style_target)
               + w_concept * concept_loss(x_adv, concept_target)
               - w_denoise * denoising_loss(surrogate, x_adv)

then minimizes it (the negated denoise term makes minimizing total_loss
equivalent to maximizing denoising_loss, so all three terms push in a
"more adversarial" direction under one Adam optimizer on delta). Alternates
with a surrogate LoRA fine-tune step exactly like aspl_attack.py's ASPL
loop -- see that module's doc for why the alternation matters.

Must run under kohya_ss's venv (diffusers + peft + CUDA for the denoising
term; VGG19/CLIP run on the same device alongside it for the other two).
"""

import argparse
from dataclasses import dataclass

import torch
import torch.optim as optim

from aspl_attack import ASPLAttacker, _reinit_lora_params
from concept_misalign import concept_loss
from model import ConceptFeatureExtractor, StyleFeatureExtractor
from style_cloak import load_image_tensor, save_tensor_image, style_loss


@dataclass
class HybridPreset:
    epsilon: float
    outer_iters: int
    surrogate_steps: int
    pgd_steps: int
    reset_every: int
    surrogate_lr: float
    pgd_lr: float
    w_style: float
    w_concept: float
    w_denoise: float


HYBRID_PRESETS = {
    "L1_PREVIEW": HybridPreset(
        epsilon=0.02, outer_iters=10, surrogate_steps=2, pgd_steps=3, reset_every=5,
        surrogate_lr=1e-4, pgd_lr=0.01, w_style=1.0, w_concept=1.0, w_denoise=1.0,
    ),
    "L3_ANTI_TRAIN": HybridPreset(
        epsilon=0.08, outer_iters=30, surrogate_steps=3, pgd_steps=6, reset_every=10,
        surrogate_lr=1e-4, pgd_lr=0.01, w_style=1.0, w_concept=1.0, w_denoise=1.0,
    ),
}


def hybrid_attack(
    original_path: str,
    checkpoint_path: str,
    prompt: str,
    style_target_path: str,
    concept_target_path: str,
    output_path: str,
    preset_name: str,
    size: int = 512,
    seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = HYBRID_PRESETS[preset_name]

    attacker = ASPLAttacker(checkpoint_path, device)
    text_embeddings = attacker.encode_prompt(prompt)
    generator = torch.Generator(device=device).manual_seed(seed)

    style_extractor = StyleFeatureExtractor(device)
    concept_extractor = ConceptFeatureExtractor(device)

    original = load_image_tensor(original_path, size, device)
    style_target = load_image_tensor(style_target_path, size, device)
    concept_target = load_image_tensor(concept_target_path, size, device)
    with torch.no_grad():
        target_grams = style_extractor.gram_matrices(style_target)
        target_concept_embed = concept_extractor.embed(concept_target)

    delta = torch.zeros_like(original, requires_grad=True)

    surrogate = attacker.build_surrogate()
    lora_params = [p for p in surrogate.parameters() if p.requires_grad]

    print(
        f"[hybrid_attack] preset={preset_name} epsilon={preset.epsilon} outer_iters={preset.outer_iters} "
        f"w_style={preset.w_style} w_concept={preset.w_concept} w_denoise={preset.w_denoise}"
    )

    s_loss_val = c_loss_val = d_loss_val = total_loss_val = float("nan")
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            _reinit_lora_params(surrogate)

        # (a) briefly fine-tune the surrogate on the current perturbed image
        surrogate.train()
        for p in lora_params:
            p.requires_grad_(True)
        surrogate_opt = optim.Adam(lora_params, lr=preset.surrogate_lr)
        x_adv_detached = (original + delta).clamp(0, 1).detach()
        for _ in range(preset.surrogate_steps):
            surrogate_opt.zero_grad()
            loss = attacker.denoising_loss(surrogate, x_adv_detached, text_embeddings, generator)
            loss.backward()
            surrogate_opt.step()

        # (b) PGD-minimize the joint style + concept + (-denoising) objective
        # against the just-trained, now-frozen surrogate
        surrogate.eval()
        for p in lora_params:
            p.requires_grad_(False)
        pgd_opt = optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()
            x_adv = (original + delta).clamp(0, 1)

            grams = style_extractor.gram_matrices(x_adv)
            s_loss = style_loss(grams, target_grams)
            c_loss = concept_loss(concept_extractor.embed(x_adv), target_concept_embed)
            d_loss = attacker.denoising_loss(surrogate, x_adv, text_embeddings, generator)

            total_loss = preset.w_style * s_loss + preset.w_concept * c_loss - preset.w_denoise * d_loss
            total_loss.backward()
            pgd_opt.step()
            s_loss_val, c_loss_val, d_loss_val, total_loss_val = (
                s_loss.item(), c_loss.item(), d_loss.item(), total_loss.item(),
            )

            with torch.no_grad():
                delta.clamp_(-preset.epsilon, preset.epsilon)
                delta.copy_(((original + delta).clamp(0, 1) - original))

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            print(
                f"  outer {outer:3d}  style={s_loss_val:.4f} concept={c_loss_val:.4f} "
                f"denoise={d_loss_val:.4f} total={total_loss_val:.4f}"
            )

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[hybrid_attack] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--style-target", required=True)
    parser.add_argument("--concept-target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=list(HYBRID_PRESETS.keys()))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hybrid_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        style_target_path=args.style_target,
        concept_target_path=args.concept_target,
        output_path=args.output,
        preset_name=args.preset,
        size=args.size,
        seed=args.seed,
    )
