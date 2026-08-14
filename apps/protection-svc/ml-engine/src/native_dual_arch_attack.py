"""Native-resolution DUAL-ARCHITECTURE attack: one delta, attacked
simultaneously against SD1.5's and SDXL's denoising losses, each through
its own architecture-appropriate EOT view.

WHY (2026-08-10). This session established two separate native-resolution
results on the same image, each tuned independently:

  * SD1.5 via VAE-uncertainty (native_lowfreq_attack.py):   delta +0.1371
  * SDXL via UNet-ASPL (native_aspl_sdxl_attack.py, 90
    outer iters, measured with the CORRECT r=32/200-step
    verification after finding the r=4 measurement bug):    delta +0.0934

Neither reached the 0.15 target alone. Attacking BOTH architectures with
ONE shared delta is the obvious next axis, and it is not merely "run both
and hope": the two objectives target genuinely different failure surfaces
(SD1.5's VAE encoder uncertainty vs SDXL's UNet denoising error), so
their gradients are not redundant the way two tunings of the same
objective are. A perturbation that satisfies both is being pushed into a
smaller, more constrained region -- which is exactly what should make it
harder for EITHER architecture's LoRA to learn around.

WHY NOT ensemble_attack_multiarch.py (which already does dual-arch): that
module attacks a flat delta at a fixed 1024x1024 letterbox, bilinearly
downsampling for SD1.5, and its own docstring records that SDXL never
showed an effect there -- the joint optimization shared one epsilon
budget across two architectures with different loss landscapes and SD1.5
crowded SDXL out. It also outputs 1024x1024, violating this project's
native-resolution requirement. This module differs in every one of those
respects:

  * delta is a LOW-RESOLUTION PARAMETER GRID upsampled to native, not a
    flat full-res tensor -- the band-limiting that made every native
    result this session possible (see native_lowfreq_attack.py's module
    docstring for the three failed attempts that led to it).
  * each branch gets its OWN EOT view at ITS OWN training resolution
    (SD1.5 ~512, SDXL ~1024), sampled per outer iteration, rather than
    one shared 1024 view bilinearly shrunk -- so neither architecture is
    attacked through the other's resize convention.
  * per-branch loss WEIGHTS (sd15_weight / sdxl_weight) exist precisely
    because the crowding-out failure is a known, measured risk here, not
    a hypothetical: they make the budget split explicit and tunable
    instead of implicit in whichever branch happens to have larger raw
    gradients.
  * all three quality safeguards this session validated are on by
    default (hue_lock, param_smooth_sigma, perceptual_mask) -- the raw
    unconstrained version of the SDXL ASPL attack produced DeepDream
    swirls and a NEGATIVE delta, so these are load-bearing, not cosmetic.

The SD1.5 side uses the UNet denoising loss (ASPL-style, matching the
SDXL side) rather than native_lowfreq_attack.py's VAE-uncertainty
objective, so that both branches ascend the same KIND of quantity and
the two weights are directly comparable. Whether that costs SD1.5 some
of its +0.1371 is exactly what the first real run measures.

STATUS: new 2026-08-10, untested at time of writing.
"""

import argparse
import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ensemble_attack_multiarch import SD15Branch, SDXLBranch
from native_eot_attack import differentiable_letterbox, load_native, save_native
from style_cloak import compute_perceptual_mask


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


def native_dual_arch_attack(
    original_path: str,
    sd15_checkpoint: str,
    sdxl_checkpoint: str,
    prompt: str,
    output_path: str,
    param_size: int = 1024,
    epsilon: float = 0.06,
    outer_iters: int = 90,
    surrogate_steps: int = 3,
    pgd_steps: int = 6,
    reset_every: int = 10,
    surrogate_lr: float = 1e-4,
    pgd_lr: float = 0.01,
    lora_r: int = 4,
    sd15_eot_size: int = 512,
    sdxl_eot_size: int = 1024,
    sd15_weight: float = 1.0,
    sdxl_weight: float = 1.0,
    eot_modes: tuple[tuple[str, bool], ...] = (("bicubic", True), ("bicubic", False), ("bilinear", True), ("area", False)),
    perceptual_mask: bool = True,
    mask_low: float = 0.3,
    mask_high: float = 1.7,
    hue_lock: bool = True,
    hue_lock_radius: int = 2,
    param_smooth_sigma: float = 5.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Defaults are the SDXL configuration that measured delta +0.0934
    (outer_iters=90, epsilon=0.06, param_smooth_sigma=5.0,
    perceptual_mask on, persistent PGD optimizer) -- the strongest
    SDXL-side setting found this session -- with the SD1.5 branch added on
    top at equal weight. That makes the first run a direct answer to "does
    adding SD1.5 to the best SDXL config help, hurt, or crowd it out?"
    rather than a fresh untethered search.

    Per-branch backward is sequential (each branch's forward graph freed
    before the next branch's forward runs), reusing
    ensemble_attack_multiarch.py's measured OOM fix verbatim: keeping a
    512-view SD1.5 graph and a 1024-view SDXL graph alive at once on an
    A40 was the actual documented cause of that module's crash, and this
    module holds MORE than it did (an upsample chain from the param grid
    on top of both). x_adv is recomputed per branch rather than shared,
    since two backward() calls through one shared node hits "backward
    through the graph a second time".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[native_dual_arch_attack] loading SD1.5 branch ({sd15_checkpoint})", flush=True)
    sd15 = SD15Branch(sd15_checkpoint, device, dtype)
    print(f"[native_dual_arch_attack] loading SDXL branch ({sdxl_checkpoint})", flush=True)
    sdxl = SDXLBranch(sdxl_checkpoint, device, dtype)

    branches = {"sd15": sd15, "sdxl": sdxl}
    eot_sizes = {"sd15": sd15_eot_size, "sdxl": sdxl_eot_size}
    weights = {"sd15": sd15_weight, "sdxl": sdxl_weight}
    cond = {
        "sd15": sd15.encode_prompt(prompt),
        "sdxl": sdxl.encode_prompt(prompt, sdxl_eot_size),
    }
    surrogates = {name: b.build_surrogate(lora_r) for name, b in branches.items()}

    x0 = load_native(original_path, device).to(dtype)
    h, w = x0.shape[-2:]
    scale = param_size / max(h, w)
    ph, pw = max(1, round(h * scale)), max(1, round(w * scale))
    delta_param = torch.zeros((1, 3, ph, pw), device=device, dtype=dtype, requires_grad=True)

    x0_low = F.interpolate(x0, size=(ph, pw), mode="bicubic", align_corners=False, antialias=True)
    epsilon_map = compute_perceptual_mask(x0_low, mask_low, mask_high) * epsilon if perceptual_mask else epsilon

    def upsample_delta(d: torch.Tensor) -> torch.Tensor:
        return F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False, antialias=True)

    c_hat = None
    if hue_lock:
        r = hue_lock_radius
        k = 2 * r + 1
        kernel = torch.ones(3, 1, k, k, device=device, dtype=dtype) / (k * k)
        local = F.conv2d(F.pad(x0_low, (r,) * 4, mode="reflect"), kernel, groups=3)
        c_hat = local / (local.norm(dim=1, keepdim=True) + 1e-6)

    def hue_project(d: torch.Tensor) -> torch.Tensor:
        if c_hat is None:
            return d
        return (d * c_hat).sum(dim=1, keepdim=True) * c_hat

    blur_kernel = None
    if param_smooth_sigma > 0:
        radius = max(1, int(3 * param_smooth_sigma))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        g = torch.exp(-(coords**2) / (2 * param_smooth_sigma**2))
        g = g / g.sum()
        blur_kernel = (g, radius)

    def param_blur(d: torch.Tensor) -> torch.Tensor:
        if blur_kernel is None:
            return d
        g, r = blur_kernel
        kh = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        kv = g.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        d = F.pad(d, (r, r, 0, 0), mode="reflect")
        d = F.conv2d(d, kh, groups=3)
        d = F.pad(d, (0, 0, r, r), mode="reflect")
        d = F.conv2d(d, kv, groups=3)
        return d

    print(
        f"[native_dual_arch_attack] param_size={param_size} ({pw}x{ph}) epsilon={epsilon} "
        f"outer_iters={outer_iters} surrogate_steps={surrogate_steps} pgd_steps={pgd_steps} "
        f"reset_every={reset_every} sd15_eot={sd15_eot_size} sdxl_eot={sdxl_eot_size} "
        f"weights=sd15:{sd15_weight}/sdxl:{sdxl_weight} sigma={param_smooth_sigma}"
        f"{' perceptual_mask=on' if perceptual_mask else ''}",
        flush=True,
    )

    # One persistent Adam for the whole run -- persist_pgd_optimizer=True
    # was measured this session as the setting that took SDXL from
    # noise-level to a real effect; recreating it per outer iteration
    # throws away momentum every pgd_steps.
    pgd_opt = optim.Adam([delta_param], lr=pgd_lr)

    losses_val = {name: float("nan") for name in branches}
    for outer in range(outer_iters):
        if outer > 0 and outer % reset_every == 0:
            for name in branches:
                _reinit_lora(surrogates[name])

        # One (mode, aa) draw per outer iteration, SHARED by both branches
        # (each still applied at its own size) -- the surrogate fine-tune
        # below needs a momentarily-fixed target, same reasoning as
        # native_aspl_sdxl_attack.py's per-outer-iteration sampling.
        mode, aa = rng.choice(eot_modes)

        with torch.no_grad():
            delta_native = upsample_delta(delta_param)
            x_adv_native_fixed = (x0 + delta_native).clamp(0, 1)
            views_fixed = {
                name: differentiable_letterbox(x_adv_native_fixed, eot_sizes[name], mode, aa).detach()
                for name in branches
            }

        # (a) fine-tune each branch's surrogate on its own fixed view
        for name, branch in branches.items():
            surrogate = surrogates[name]
            surrogate.train()
            params = _lora_params(surrogate)
            for p in params:
                p.requires_grad_(True)
            opt = optim.Adam(params, lr=surrogate_lr)
            for _ in range(surrogate_steps):
                opt.zero_grad()
                loss = branch.denoising_loss(surrogate, views_fixed[name], cond[name], generator)
                loss.backward()
                opt.step()
            for p in params:
                p.requires_grad_(False)
            surrogate.eval()

        # (b) PGD-update the SHARED delta_param against both frozen surrogates
        for _ in range(pgd_steps):
            pgd_opt.zero_grad()

            for name, branch in branches.items():
                # Recomputed per branch so each backward has its own graph
                # back to delta_param, and so each branch's graph is freed
                # before the next branch's forward runs (the measured OOM
                # fix from ensemble_attack_multiarch.py).
                delta_native = upsample_delta(delta_param)
                x_adv_native = (x0 + delta_native).clamp(0, 1)
                x_adv_view = differentiable_letterbox(x_adv_native, eot_sizes[name], mode, aa)
                d_loss = branch.denoising_loss(surrogates[name], x_adv_view, cond[name], generator)
                losses_val[name] = d_loss.item()
                (-weights[name] * d_loss).backward()  # ascent; grads accumulate into delta_param.grad

            with torch.no_grad():
                delta_param.grad.copy_(hue_project(param_blur(delta_param.grad)))
            pgd_opt.step()

            with torch.no_grad():
                delta_param.clamp_(min=-epsilon_map, max=epsilon_map)
                delta_param.copy_(hue_project(delta_param))

        if outer % 5 == 0 or outer == outer_iters - 1:
            print(
                f"  outer {outer:3d}  sd15={losses_val['sd15']:.6f}  sdxl={losses_val['sdxl']:.6f}  "
                f"view=({mode},{aa})",
                flush=True,
            )

    with torch.no_grad():
        x_final = (x0 + upsample_delta(delta_param)).clamp(0, 1)
    save_native(x_final, output_path)
    print(f"[native_dual_arch_attack] wrote {output_path} ({w}x{h}, native, param grid {pw}x{ph})", flush=True)
    return {"losses": losses_val, "native_size": (w, h), "param_size": (pw, ph)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--sd15-checkpoint", required=True)
    parser.add_argument("--sdxl-checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--param-size", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--outer-iters", type=int, default=90)
    parser.add_argument("--surrogate-steps", type=int, default=3)
    parser.add_argument("--pgd-steps", type=int, default=6)
    parser.add_argument("--reset-every", type=int, default=10)
    parser.add_argument("--surrogate-lr", type=float, default=1e-4)
    parser.add_argument("--pgd-lr", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--sd15-eot-size", type=int, default=512)
    parser.add_argument("--sdxl-eot-size", type=int, default=1024)
    parser.add_argument("--sd15-weight", type=float, default=1.0)
    parser.add_argument("--sdxl-weight", type=float, default=1.0)
    parser.add_argument("--no-perceptual-mask", action="store_true")
    parser.add_argument("--mask-low", type=float, default=0.3)
    parser.add_argument("--mask-high", type=float, default=1.7)
    parser.add_argument("--no-hue-lock", action="store_true")
    parser.add_argument("--hue-lock-radius", type=int, default=2)
    parser.add_argument("--param-smooth-sigma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    native_dual_arch_attack(
        original_path=args.original,
        sd15_checkpoint=args.sd15_checkpoint,
        sdxl_checkpoint=args.sdxl_checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        param_size=args.param_size,
        epsilon=args.epsilon,
        outer_iters=args.outer_iters,
        surrogate_steps=args.surrogate_steps,
        pgd_steps=args.pgd_steps,
        reset_every=args.reset_every,
        surrogate_lr=args.surrogate_lr,
        pgd_lr=args.pgd_lr,
        lora_r=args.lora_r,
        sd15_eot_size=args.sd15_eot_size,
        sdxl_eot_size=args.sdxl_eot_size,
        sd15_weight=args.sd15_weight,
        sdxl_weight=args.sdxl_weight,
        perceptual_mask=not args.no_perceptual_mask,
        mask_low=args.mask_low,
        mask_high=args.mask_high,
        hue_lock=not args.no_hue_lock,
        hue_lock_radius=args.hue_lock_radius,
        param_smooth_sigma=args.param_smooth_sigma,
        seed=args.seed,
    )
