"""Native-resolution ASPL attack for SDXL -- fuses the two independently
validated halves of this project's SDXL work instead of inventing a third
mechanism from scratch.

WHY THIS EXISTS (2026-08-09). native_lowfreq_attack.py's VAE-uncertainty
objective (logvar maximization + reconstruction error, targeting only the
VAE encoder) reached delta +0.1371 on SD1.5, native resolution, real
train+score verified. Porting that SAME objective to SDXL (same
param-grid/upsample/EOT scaffolding, --eot-sizes tuned to SDXL's 1024
training resolution) scored only +0.0080 -- and a controlled epsilon/steps
sweep (epsilon 0.05->0.15, steps 500->1500, each checked cheaply via a
fixed-EOT-view VAE-loss diagnostic before spending on real train+score)
showed the VAE-loss shift saturates at the SAME ceiling regardless of
budget or optimization time (see lora-protection-research memory for the
numbers). That is a real, structural ceiling for THIS OBJECTIVE on
SDXL's VAE -- not a tuning problem this module tries to solve by pushing
harder on the same lever.

This project already has a SEPARATE, independently validated attack
surface for SDXL that does NOT hit that ceiling: aspl_attack_sdxl_only.py
attacks the UNet's denoising loss directly via ASPL (alternating a brief
LoRA surrogate fine-tune with a PGD update against that just-trained,
now-frozen surrogate) -- the SDXL_FULL preset was real-train-score
validated at n=30 already (see that module's own docstring and the
project's PHASE4_SCOPING.md SS6). The problem with reusing it as-is is
resolution, not mechanism: it operates on a fixed 1024x1024 letterboxed
square via style_cloak.load_image_tensor, so its OUTPUT is 1024x1024 --
violating this project's standing requirement that a protected image's
resolution always match the original upload
([[feedback-protection-output-requirements]]). Restoring resolution
POST-HOC after an attack like this is exactly the move that already
failed once this session's SD1.5 line (resolution_restore.py, delta
+0.1594 -> +0.0519 after fixing a resize-kernel mismatch that first LOOKED
like a clean pixel-round-trip proof) -- so this module does not take that
path again.

THE FUSION: keep aspl_attack_sdxl_only.py's ASPL loop and denoising_loss
objective verbatim in spirit (same alternation structure, same optimizer
choice -- Adam for both the surrogate fine-tune and the delta PGD step,
unlike native_lowfreq_attack.py's sign-PGD-by-default), but replace its
flat full-resolution `delta` parameter with native_lowfreq_attack.py's
band-limited param-grid + bicubic-upsample-to-native construction, and
replace its single fixed 1024x1024 input with an EOT-sampled letterboxed
VIEW of the native-resolution composite (still centered on 1024, SDXL's
own real training resolution, matching what its own
[[lora-protection-research]] entry already established: eot_sizes must be
overridden away from native_eot_attack.py's SD1.5-weighted default, and
capped at 1024 alone to fit recon-weight-equivalent memory -- ASPL's
per-outer-iteration surrogate fine-tune is itself already a full backward
pass through the UNet, considerably heavier than the VAE-only loss this
constraint was first measured against, so the same single-size-1024 EOT
list is kept rather than reintroduced with more variety before a fresh
OOM check).

The EOT view is resampled once per OUTER iteration, not per PGD substep:
aspl_attack_sdxl_only.py's own alternation is already "fine-tune surrogate
on a momentarily-fixed image, then PGD against that frozen surrogate" --
substep-level resampling would make the surrogate's brief fine-tune see a
different image every step, undermining the point of that fine-tune
(learning what THIS specific composite currently looks like well enough
to attack it meaningfully). Resampling once per outer iteration instead
achieves the EOT averaging this project's whole native line depends on
(robustness to which exact resize a downstream trainer applies) at the
granularity the ASPL alternation was actually designed around.

STATUS: new 2026-08-09, not yet run. Untested at time of writing -- see
lora-protection-research memory for whatever this scores once it has
been.
"""

import argparse
import math
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ensemble_attack_multiarch import SDXLBranch
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


def native_aspl_sdxl_attack(
    original_path: str,
    checkpoint_path: str,
    prompt: str,
    output_path: str,
    param_size: int = 1024,
    epsilon: float = 0.08,
    outer_iters: int = 30,
    surrogate_steps: int = 3,
    pgd_steps: int = 6,
    reset_every: int = 10,
    surrogate_lr: float = 1e-4,
    pgd_lr: float = 0.01,
    lora_r: int = 4,
    eot_size: int = 1024,
    eot_modes: tuple[tuple[str, bool], ...] = (("bicubic", True), ("bicubic", False), ("bilinear", True), ("area", False)),
    perceptual_mask: bool = False,
    mask_low: float = 0.3,
    mask_high: float = 1.7,
    hue_lock: bool = True,
    hue_lock_radius: int = 2,
    param_smooth_sigma: float = 1.0,
    persist_pgd_optimizer: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """`eot_size` is a single int, not a tuple like native_lowfreq_attack.py's
    eot_sizes -- the OOM constraint measured this session capped SDXL's
    EOT views at one size (1024) for the VAE-only attack, and ASPL's
    surrogate fine-tune is heavier per step (full UNet forward+backward,
    not just a VAE encode/decode), so starting from that same single-size
    constraint rather than re-testing for headroom is the conservative
    choice; loosen only after a fresh OOM check if this needs more EOT
    diversity later. `eot_modes` still varies (kernel + antialias), so the
    EOT sampling is not fully degenerate -- only the target SIZE is fixed.

    `hue_lock`/`param_smooth_sigma` (2026-08-09, added after the first real
    run of this module came back with heavy swirling/paisley texture and a
    visible colour shift across the WHOLE image, not just texture-heavy
    regions -- much worse than anything the SD1.5 native line produced).
    The first version of this module ported ONLY the ASPL mechanism
    (surrogate alternation + denoising-loss PGD) from
    aspl_attack_sdxl_only.py, with none of the quality safeguards
    native_lowfreq_attack.py's SD1.5 line spent this whole session
    developing and validating for exactly this failure mode -- an
    oversight, not a deliberate simplification. Attacking a UNet's own
    learned features directly and unconstrained is well known to produce
    DeepDream-style swirls (this project's own hybrid_attack.py history
    already noted psychedelic artifacts from under-constrained
    feature-level attacks); this module's plain Adam-on-delta_param with
    no colour or smoothness constraint at all was always going to
    reproduce that failure mode once actually looked at, exactly as it did.

    `hue_lock` reuses native_lowfreq_attack.py's exact mechanism: project
    each PGD gradient onto the local original colour's own unit direction
    (computed once from a `hue_lock_radius`-blurred x0_low) before it
    updates delta_param, so the perturbation can only push each pixel's
    channels together along the direction the original artwork's own
    colour already points in -- it cannot invent a new hue. This directly
    targets "색감이 변했다" (colour looks shifted).

    `param_smooth_sigma` reuses the same "blur only the fresh step, never
    the accumulated parameter" pattern validated repeatedly this session:
    a Gaussian blur in low-res PARAMETER space applied to the raw gradient
    before Adam's moment accumulation sees it (not to delta_param, which
    would compound across all outer*pgd_steps iterations and oversmooth).
    This directly targets "물결 무늬 패턴" -- the swirl is high-frequency
    structure at the param grid's own resolution, upsampled; blurring the
    step before it accumulates is what fixed the analogous SD1.5 artifact
    (delta +0.1098 -> +0.1371, WITH a real quality improvement, not just a
    number -- see lora-protection-research memory).

    MEASURED (quality fix): hue_lock + param_smooth_sigma=5.0 +
    perceptual_mask solved the visual problem completely (confirmed by
    direct inspection and shown to the user across several runs). Delta
    across six real measurements sweeping epsilon/sigma/perceptual_mask/
    outer_iters as four SEPARATE single-value levers stayed inside
    roughly +/-0.02, indistinguishable from n=1 seed noise, never
    trending toward the 0.15 target.

    `persist_pgd_optimizer` (2026-08-09, a fifth idea -- not another
    single-lever magnitude tweak, a change to the OPTIMISATION DYNAMICS
    itself): every one of those six measurements, and
    aspl_attack_sdxl_only.py before them, recreates `pgd_opt =
    optim.Adam([delta_param], lr=pgd_lr)` FRESH inside the outer loop --
    once per outer iteration, i.e. every `pgd_steps` (6) substeps, Adam's
    first/second moment estimates are thrown away and bias-correction
    restarts from t=1. Adam's whole advantage over plain sign-PGD is
    using accumulated gradient history to scale and direct each step;
    resetting that history every 6 steps caps it at behaving like a
    noisier, badly-calibrated sign-PGD for most of each outer iteration,
    never accumulating the longer-horizon momentum that might let
    delta_param converge toward a more consistently hard-to-learn
    perturbation across the FULL run. This was inherited unexamined from
    aspl_attack_sdxl_only.py's original fixed-1024 design (where it may
    matter less -- no upsample/EOT chain sits between delta and the loss
    there) rather than chosen deliberately for this native/EOT-wrapped
    version.

    persist_pgd_optimizer=True creates ONE `pgd_opt` bound to
    `delta_param` before the outer loop and never recreates it, so its
    moment estimates accumulate across the entire attack (outer_iters *
    pgd_steps real Adam steps, not pgd_steps at a time) -- the surrogate's
    own optimizer is NOT changed (it is legitimately meant to restart each
    outer iteration, since aspl_attack_sdxl_only.py's own re-init logic
    already treats the surrogate's LoRA weights, not just its optimizer,
    as periodically reset via reset_every). Set False to reproduce the
    original per-outer-iteration-reset behaviour for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    branch = SDXLBranch(checkpoint_path, device, dtype)
    cond = branch.encode_prompt(prompt, eot_size)
    surrogate = branch.build_surrogate(lora_r)

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
        f"[native_aspl_sdxl_attack] param_size={param_size} ({pw}x{ph}) epsilon={epsilon} "
        f"outer_iters={outer_iters} surrogate_steps={surrogate_steps} pgd_steps={pgd_steps} "
        f"reset_every={reset_every} eot_size={eot_size} persist_pgd_optimizer={persist_pgd_optimizer}"
        f"{' perceptual_mask=on' if perceptual_mask else ''}",
        flush=True,
    )

    # persist_pgd_optimizer=True: one Adam bound to delta_param for the
    # WHOLE run, so its moment estimates accumulate across outer_iters *
    # pgd_steps real steps instead of resetting every pgd_steps -- see this
    # function's own docstring for why the reset (inherited unexamined
    # from aspl_attack_sdxl_only.py) was suspected of capping delta.
    persistent_pgd_opt = optim.Adam([delta_param], lr=pgd_lr) if persist_pgd_optimizer else None

    loss_val = float("nan")
    for outer in range(outer_iters):
        if outer > 0 and outer % reset_every == 0:
            _reinit_lora(surrogate)

        mode, aa = rng.choice(eot_modes)

        # (a) fine-tune the surrogate on the CURRENT composite, EOT-viewed
        # at this outer iteration's sampled (mode, aa) -- detached, a fixed
        # training target for this brief fine-tune, same as
        # aspl_attack_sdxl_only.py's x_adv_detached.
        with torch.no_grad():
            delta_native = upsample_delta(delta_param)
            x_adv_view_fixed = differentiable_letterbox((x0 + delta_native).clamp(0, 1), eot_size, mode, aa).detach()

        surrogate.train()
        params = _lora_params(surrogate)
        for p in params:
            p.requires_grad_(True)
        opt = optim.Adam(params, lr=surrogate_lr)
        for _ in range(surrogate_steps):
            opt.zero_grad()
            loss = branch.denoising_loss(surrogate, x_adv_view_fixed, cond, generator)
            loss.backward()
            opt.step()
        for p in params:
            p.requires_grad_(False)
        surrogate.eval()

        # (b) PGD-update delta_param against the just-trained, now-frozen
        # surrogate -- recomputes the EOT view fresh each substep from the
        # updated delta_param (same (mode, aa) for the whole outer
        # iteration, only the pixel values change as delta_param moves),
        # so gradient flows through both the letterbox and the upsample
        # back to delta_param, unlike aspl_attack_sdxl_only.py's flat delta
        # which needed neither step.
        pgd_opt = persistent_pgd_opt if persistent_pgd_opt is not None else optim.Adam([delta_param], lr=pgd_lr)
        for _ in range(pgd_steps):
            pgd_opt.zero_grad()
            delta_native = upsample_delta(delta_param)
            x_adv_native = (x0 + delta_native).clamp(0, 1)
            x_adv_view = differentiable_letterbox(x_adv_native, eot_size, mode, aa)
            loss = branch.denoising_loss(surrogate, x_adv_view, cond, generator)
            (-loss).backward()
            with torch.no_grad():
                # Constrain the FRESH gradient before Adam's moment
                # accumulation sees it, not delta_param after the fact --
                # blurring/re-projecting the accumulated parameter every
                # substep would compound across all outer*pgd_steps
                # iterations, smearing far past param_smooth_sigma.
                delta_param.grad.copy_(hue_project(param_blur(delta_param.grad)))
            pgd_opt.step()
            loss_val = loss.item()

            with torch.no_grad():
                delta_param.clamp_(min=-epsilon_map, max=epsilon_map)
                # The clamp is nonlinear and can leak a little energy off
                # the hue-locked line; re-project (idempotent, exact) to
                # hold the colour constraint exactly after every clamp.
                delta_param.copy_(hue_project(delta_param))

        if outer % 5 == 0 or outer == outer_iters - 1:
            print(f"  outer {outer:3d}  sdxl_loss={loss_val:.6f}  view=({eot_size},{mode},{aa})", flush=True)

    with torch.no_grad():
        x_final = (x0 + upsample_delta(delta_param)).clamp(0, 1)
    save_native(x_final, output_path)
    print(f"[native_aspl_sdxl_attack] wrote {output_path} ({w}x{h}, native, param grid {pw}x{ph})", flush=True)
    return {"final_loss": loss_val, "native_size": (w, h), "param_size": (pw, ph)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--param-size", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--outer-iters", type=int, default=30)
    parser.add_argument("--surrogate-steps", type=int, default=3)
    parser.add_argument("--pgd-steps", type=int, default=6)
    parser.add_argument("--reset-every", type=int, default=10)
    parser.add_argument("--surrogate-lr", type=float, default=1e-4)
    parser.add_argument("--pgd-lr", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--eot-size", type=int, default=1024)
    parser.add_argument("--perceptual-mask", action="store_true")
    parser.add_argument("--mask-low", type=float, default=0.3)
    parser.add_argument("--mask-high", type=float, default=1.7)
    parser.add_argument("--no-hue-lock", action="store_true")
    parser.add_argument("--hue-lock-radius", type=int, default=2)
    parser.add_argument("--param-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--no-persist-pgd-optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    native_aspl_sdxl_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
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
        eot_size=args.eot_size,
        perceptual_mask=args.perceptual_mask,
        mask_low=args.mask_low,
        mask_high=args.mask_high,
        hue_lock=not args.no_hue_lock,
        hue_lock_radius=args.hue_lock_radius,
        param_smooth_sigma=args.param_smooth_sigma,
        persist_pgd_optimizer=not args.no_persist_pgd_optimizer,
        seed=args.seed,
    )
