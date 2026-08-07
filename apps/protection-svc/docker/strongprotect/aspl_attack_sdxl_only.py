"""Single-architecture ASPL attack targeting SDXL alone -- the SDXL
counterpart to aspl_attack.py (which has only ever targeted SD1.5).

Why this exists: every attack mechanism tried against SDXL in this
project's history (see [[lora-protection-research]] memory,
PHASE4_SCOPING.md SS6) only ever attacked it *jointly* with SD1.5, inside
ensemble_attack_multiarch.py's shared-delta optimization. SD1.5 passed;
SDXL never showed an effect either direction. That leaves a real confound
unresolved: is SDXL fundamentally unresponsive to this attack class, or
was it being crowded out by the joint optimization sharing one epsilon
budget across two architectures with different loss landscapes -- exactly
the failure mode hybrid_attack.py already demonstrated once (combining
multiple objectives made things worse, not better, when this project first
tried it on style+concept+denoise jointly against a single SD1.5
surrogate). This module isolates the variable: same ASPL mechanism
(alternating surrogate fine-tune + PGD), same fp32/epsilon=0.08/iteration
counts as aspl_attack.py's L3_ANTI_TRAIN and multiarch_ensemble_attack's
MULTIARCH_FULL, but SDXL is the *only* surrogate being attacked -- nothing
else competing for the same epsilon budget.

Reuses ensemble_attack_multiarch.py's SDXLBranch class directly rather
than reimplementing SDXL's own forward-pass plumbing (dual text encoders,
added_cond_kwargs) a second time.

Must run under a venv with diffusers + peft + torch/CUDA (see
docker/strongprotect/Dockerfile or this project's other experiment pods).
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.optim as optim

from ensemble_attack_multiarch import SDXLBranch
from style_cloak import load_image_tensor, save_tensor_image


@dataclass
class SDXLOnlyPreset:
    epsilon: float
    outer_iters: int
    surrogate_steps: int
    pgd_steps: int
    reset_every: int
    surrogate_lr: float
    pgd_lr: float
    size: int
    lora_r: int


SDXL_ONLY_PRESETS = {
    # Same shape as ensemble_attack_multiarch.py's CALIBRATION preset --
    # fast timing/wiring check before the full-scale run.
    "CALIBRATION": SDXLOnlyPreset(
        epsilon=0.08, outer_iters=10, surrogate_steps=2, pgd_steps=3, reset_every=5,
        surrogate_lr=1e-4, pgd_lr=0.01, size=1024, lora_r=4,
    ),
    # Matches aspl_attack.py's L3_ANTI_TRAIN and multiarch_ensemble_attack's
    # MULTIARCH_FULL iteration counts exactly -- the only thing this preset
    # changes relative to those is which architecture(s) share the budget.
    "SDXL_FULL": SDXLOnlyPreset(
        epsilon=0.08, outer_iters=30, surrogate_steps=3, pgd_steps=6, reset_every=10,
        surrogate_lr=1e-4, pgd_lr=0.01, size=1024, lora_r=4,
    ),
}


def _lora_params(surrogate) -> list:
    return [p for n, p in surrogate.named_parameters() if "lora_A" in n or "lora_B" in n]


def _reinit_lora(surrogate) -> None:
    for name, param in surrogate.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name:
            torch.nn.init.zeros_(param)


def aspl_attack_sdxl_only(
    original_path: str,
    sdxl_checkpoint: str,
    prompt: str,
    output_path: str,
    preset_name: str,
    seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = SDXL_ONLY_PRESETS[preset_name]
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[aspl_attack_sdxl_only] loading SDXL branch ({sdxl_checkpoint})")
    branch = SDXLBranch(sdxl_checkpoint, device, dtype)
    cond = branch.encode_prompt(prompt, preset.size)
    surrogate = branch.build_surrogate(preset.lora_r)

    original = load_image_tensor(original_path, preset.size, device).to(dtype)
    delta = torch.zeros_like(original, requires_grad=True)

    print(
        f"[aspl_attack_sdxl_only] preset={preset_name} epsilon={preset.epsilon} "
        f"outer_iters={preset.outer_iters} surrogate_steps={preset.surrogate_steps} "
        f"pgd_steps={preset.pgd_steps} reset_every={preset.reset_every}"
    )

    loss_val = float("nan")
    for outer in range(preset.outer_iters):
        if outer > 0 and outer % preset.reset_every == 0:
            _reinit_lora(surrogate)

        # (a) briefly fine-tune the surrogate on the current perturbed image
        surrogate.train()
        params = _lora_params(surrogate)
        for p in params:
            p.requires_grad_(True)
        opt = optim.Adam(params, lr=preset.surrogate_lr)
        x_adv_detached = (original + delta).clamp(0, 1).detach()
        for _ in range(preset.surrogate_steps):
            opt.zero_grad()
            loss = branch.denoising_loss(surrogate, x_adv_detached, cond, generator)
            loss.backward()
            opt.step()
        for p in params:
            p.requires_grad_(False)
        surrogate.eval()

        # (b) PGD-update delta against the just-trained (now frozen) surrogate
        pgd_opt = optim.Adam([delta], lr=preset.pgd_lr)
        for _ in range(preset.pgd_steps):
            pgd_opt.zero_grad()
            x_adv = (original + delta).clamp(0, 1)
            loss = branch.denoising_loss(surrogate, x_adv, cond, generator)
            (-loss).backward()
            pgd_opt.step()
            loss_val = loss.item()

            with torch.no_grad():
                delta.clamp_(-preset.epsilon, preset.epsilon)
                delta.copy_(((original + delta).clamp(0, 1) - original))

        if outer % 5 == 0 or outer == preset.outer_iters - 1:
            print(f"  outer {outer:3d}  sdxl_loss={loss_val:.6f}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[aspl_attack_sdxl_only] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="SDXL_FULL", choices=list(SDXL_ONLY_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    aspl_attack_sdxl_only(
        original_path=args.original,
        sdxl_checkpoint=args.sdxl_checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        preset_name=args.preset,
        seed=args.seed,
    )
