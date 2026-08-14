"""Attacks the VAE encoder directly, instead of the UNet's denoising loss --
a structurally different attack surface from every mechanism in this
project so far (aspl_attack.py, hybrid_protect.py's latent stage, all of
[[lora-protection-research]]'s prior experiments).

WHY (2026-08-08): hybrid_protect.py's latent-space stage assumed that
staying inside the VAE's latent ball keeps an image "plausible" and
therefore safe from over-distortion. Turned around, that assumption says
something else: the VAE can only represent what it already knows how to
represent. Passing an image through vae.encode() -> vae.decode() pulls it
toward the manifold of images the VAE reconstructs well -- which, for a
model pretrained on a huge generic image corpus, means pulling AWAY from
whatever makes one artist's brushwork distinctive. That is the opposite of
protection: it can make an image *more* typical, and therefore *easier*
for a LoRA to learn, not harder. This is a working hypothesis, prompted by
a real deployed-artwork test coming back with a negative delta (protected
image scored as MORE learnable than the original) plus visible texture
bleed matching this exact failure mode -- not yet confirmed at n=1 real
training, let alone n=30.

This module inverts the assumption instead of tuning its budget: rather
than staying inside the VAE's comfort zone, it pushes an image toward
where the VAE's own encoder is UNCERTAIN or reconstructs POORLY. Two
objectives, combinable:

  - logvar maximization: `vae.encode(x).latent_dist` is a diagonal
    Gaussian (mean, logvar) -- the encoder's own predicted posterior
    variance. A high logvar region is where the encoder itself is unsure
    what the "right" latent is. Every downstream training step samples
    `latent_dist.sample()` (see aspl_attack.py's denoising_loss and every
    real LoRA trainer this project's threat model is built on) -- high
    variance means each training step sees a DIFFERENT latent for the
    same pixels, which should make it harder for gradient descent to
    converge on a consistent representation of the image at all.
  - reconstruction-error maximization: `||x - vae.decode(vae.encode(x).mode())||`
    -- how unfaithfully the VAE reconstructs x. This is a proxy for "how
    far x sits from the manifold the VAE (and therefore every downstream
    LoRA built on that VAE) can represent well".

Neither objective is a proxy for what a human sees -- a perceptually
tiny perturbation can land in a region of very different encoder
uncertainty, which is exactly the point: this targets the compression
front-end every latent diffusion architecture shares (SD1.5, SDXL, SD3,
Flux all use a KL-VAE of this same family), not a UNet-specific weakness,
so it may transfer across architectures in a way UNet-targeted attacks
(this project's entire history before today) structurally cannot.

Pixel-space L-infinity ball for now, same as every attack in this project
before the latent-space experiment -- kept simple to test the underlying
hypothesis cheaply before adding a perceptual (LPIPS) constraint on top
(see the project's next planned experiment).

STATUS: brand new, 2026-08-08. Not even visually piloted yet, let alone
n=1 real-training-checked. This is the first artifact of that pilot.
"""

import argparse

import torch
import torch.nn.functional as F

from style_cloak import compute_perceptual_mask, load_image_tensor, save_tensor_image


def load_vae(checkpoint_path: str, device: torch.device, dtype: torch.dtype = torch.float32, eager_attention: bool = False):
    """Loads only the VAE component out of a merged SD/SDXL checkpoint --
    skips UNet/text-encoder entirely (unlike ASPLAttacker/SDXLBranch,
    which load the whole pipeline because they need all of it) since this
    attack never touches the denoising model at all.

    eager_attention=True swaps the mid-block's fused SDPA attention
    (diffusers' default AttnProcessor2_0) for the plain unfused
    implementation (AttnProcessor: separate matmul/softmax/matmul).
    Needed by quality_restore.py's restore_nullspace_cg, which
    differentiates through the encoder TWICE (a double-backward JVP
    trick) -- PyTorch's fused/efficient attention kernel has no
    second-order backward implemented at all (hit live: first
    forward-mode AD failed outright, then even reverse-mode
    double-backward failed with `derivative for
    aten::_scaled_dot_product_efficient_attention_backward is not
    implemented`). Plain eager attention is literally just tensor ops
    autograd already knows how to differentiate to any order, at the
    cost of some memory/speed -- irrelevant here since this is one VAE,
    not the full UNet training loop. Off by default since every other
    caller only needs a single backward pass, where the fused kernel is
    faster and this swap is pure overhead.
    """
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_single_file(checkpoint_path, torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    if eager_attention:
        from diffusers.models.attention_processor import AttnProcessor

        vae.set_attn_processor(AttnProcessor())
    return vae


def vae_uncertainty_loss(vae, x: torch.Tensor, dtype: torch.dtype, logvar_weight: float, recon_weight: float) -> torch.Tensor:
    """Higher is better for the attacker (we ascend this via gradient
    ascent -- see the PGD loop below, which adds rather than subtracts
    the gradient step)."""
    posterior = vae.encode(x.to(dtype) * 2 - 1).latent_dist
    loss = torch.zeros((), device=x.device)

    if logvar_weight > 0:
        # posterior.logvar is per-element; mean over the whole latent so
        # the loss scale doesn't depend on latent spatial size.
        loss = loss + logvar_weight * posterior.logvar.mean()

    if recon_weight > 0:
        # posterior.mode() is already in the VAE's own (unscaled) latent
        # convention -- decode() expects that directly, no scaling_factor
        # division needed (unlike the UNet path, which multiplies by
        # scaling_factor when ENcoding and must undo that before DEcoding;
        # mode()/sample() from .encode() are pre-scaling).
        recon = vae.decode(posterior.mode()).sample
        recon_01 = (recon / 2 + 0.5).clamp(0, 1)
        loss = loss + recon_weight * F.mse_loss(recon_01, x)

    return loss


def _load_native_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Loads at the image's own real resolution -- no letterbox square
    padding, no resize -- cropped only by the 0-7px needed to make each
    side a multiple of 8 (the VAE's spatial downsample factor; anything
    not a multiple of 8 doesn't round-trip through encode/decode cleanly).

    EXISTS BECAUSE (2026-08-08): the original design attacked at a fixed
    small size (512x512, letterboxed) and reconstructed full resolution
    afterward from the true original (resolution_restore.py). That
    reconstruction's pixel-level round-trip proof held (measured MSE
    ~1e-5) but real train+score still showed protection cut in half --
    this specific attack targets a point of deliberately high VAE
    sensitivity (that IS its mechanism), so even sub-pixel resize noise
    was enough to knock it off that point. Attacking directly at the
    resolution that gets published removes the round-trip entirely rather
    than trying to survive it -- there is no reconstruction step left to
    get subtly wrong.
    """
    from PIL import Image

    import numpy as np

    img = Image.open(path).convert("RGB")
    w, h = img.size
    w8, h8 = max(8, (w // 8) * 8), max(8, (h // 8) * 8)
    if (w8, h8) != (w, h):
        left, top = (w - w8) // 2, (h - h8) // 2
        img = img.crop((left, top, left + w8, top + h8))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def vae_uncertainty_attack(
    original_path: str,
    checkpoint_path: str,
    output_path: str,
    size: int | None = 512,
    epsilon: float = 0.03,
    steps: int = 100,
    step_size: float | None = None,
    logvar_weight: float = 1.0,
    recon_weight: float = 1.0,
    hue_lock: bool = False,
    hue_lock_radius: int = 2,
    smooth_sigma: float = 0.0,
    structure_align: bool = False,
    structure_align_radius: int = 3,
    perceptual_mask: bool = False,
    mask_low: float = 0.3,
    mask_high: float = 1.7,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """size=None attacks at the image's own native resolution (see
    _load_native_tensor) -- the output is already publish-resolution, no
    separate restoration step needed or wanted. size=<int> keeps the
    original small-square-letterbox behaviour (cheaper, but needs
    resolution_restore.py afterward, which this mechanism's own
    sensitivity currently breaks -- see that module's status note).

    hue_lock=True constrains the perturbation, at every PGD step, to lie
    along each pixel's own local colour direction -- so it can only
    brighten or darken the colour already present and is structurally
    incapable of introducing a hue the artwork does not contain (which is
    what makes an adversarial pattern read as foreign speckle rather than
    as brushwork or grain).

    smooth_sigma>0 additionally forces the perturbation to be spatially
    smooth, by Gaussian-blurring the GRADIENT STEP before it accumulates
    into delta (2026-08-08 addition -- hue_lock alone still left visible
    pixel-to-pixel grain: PGD's grad.sign() picks +-1 independently per
    pixel, which is maximally HIGH-frequency noise by construction,
    regardless of colour direction). Blurring the step, not the final
    delta, keeps this in-loop for the same reason hue_lock is in-loop
    (see below) -- the optimizer only ever explores smooth perturbations,
    rather than finding a jagged one and softening it afterwards.

    WHY THESE ARE IN THE ATTACK LOOP AND NOT A POST-PROCESS (2026-08-08):
    harmonize_perturbation.py first tried hue-lock as a projection applied
    AFTER the attack finished. It looked good and its latent drift (9.5)
    sat comfortably inside the drift gate calibrated that same day -- but
    a real train+score run showed protection had dropped 39% anyway
    (delta +0.1402 -> +0.0857). Two lessons, both worth keeping: (1) a
    post-hoc projection can only redistribute a perturbation the optimizer
    already committed to, and the component it discards is exactly the
    component that was doing work; (2) the drift gate was calibrated by
    blending toward the ORIGINAL, and does not transfer to transforms that
    change the perturbation's colour DIRECTION -- equal drift, very
    different cost.

    Projecting inside the loop has neither problem: the constraint is part
    of the feasible set PGD optimizes over, so gradient ascent finds the
    strongest perturbation that was hue-locked (and/or smooth) to begin
    with, rather than the strongest perturbation overall followed by a
    lossy repair. Confirmed for hue_lock alone (delta went UP, +0.1608 vs
    +0.1448 unconstrained). smooth_sigma follows the identical principle
    but was NOT free when measured (delta -> +0.1388, ~14% of hue_lock's
    strength given up for visibly less pixel-grain) -- isotropic blur
    trades away some real signal even done in-loop, unlike hue_lock's
    pure redistribution. Recorded honestly rather than only reporting the
    free win.

    structure_align=True (2026-08-08) replaces isotropic smoothing with
    ANISOTROPIC smoothing along the artwork's own local edge direction
    (from x0's structure tensor, computed once -- not per step, since the
    image itself doesn't change). Blurring only along edges (not across
    them) should in principle give up less than smooth_sigma's isotropic
    blur for the same visual gain: noise that runs with brushwork reads
    as texture, and this discards less high-frequency content in the
    directions the encoder is actually watching (across edges, where real
    image structure lives) while still discarding it along flat runs.
    Unvalidated -- measure before trusting the "should" above.

    perceptual_mask=True (2026-08-08) makes the epsilon clamp spatially
    varying instead of a single scalar, using style_cloak.py's
    compute_perceptual_mask (Sobel-based: less budget in smooth regions
    a human eye is sensitive to, more in already-textured regions where
    noise hides). This redistributes the SAME total budget rather than
    shrinking it, which is exactly the property that made hue_lock free --
    the expectation is this should be close to free too, but that is a
    hypothesis carried over from a different mechanism (style_cloak's own
    attack, +1.37dB PSNR there) and this project's 11th experiment already
    found perceptual masking does NOT help the (unrelated) pixel-space
    ASPL chain at large epsilon. Applying it in-loop here, to a
    genuinely different attack, is the fair test this deserves -- not
    assumed transferable either direction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(checkpoint_path, device, dtype)

    x0 = _load_native_tensor(original_path, device) if size is None else load_image_tensor(original_path, size, device)
    delta = torch.zeros_like(x0, requires_grad=True)
    step_size = step_size if step_size is not None else epsilon / 4

    import torch.nn.functional as _F

    c_hat = None
    if hue_lock:
        r = hue_lock_radius
        k = 2 * r + 1
        kernel = torch.ones(3, 1, k, k, device=device, dtype=x0.dtype) / (k * k)
        # Local average colour rather than the raw pixel: a single pixel's
        # colour is noisy, and locking to it would make the projection
        # itself high-frequency -- a new artifact in place of the old one.
        local = _F.conv2d(_F.pad(x0, (r,) * 4, mode="reflect"), kernel, groups=3)
        c_hat = local / (local.norm(dim=1, keepdim=True) + 1e-6)

    blur_kernel = None
    if smooth_sigma > 0:
        radius = max(1, int(3 * smooth_sigma))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=x0.dtype)
        g = torch.exp(-(coords**2) / (2 * smooth_sigma**2))
        g = g / g.sum()
        blur_kernel = (g, radius)

    def blur(d: torch.Tensor) -> torch.Tensor:
        if blur_kernel is None:
            return d
        g, r = blur_kernel
        kh = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        kv = g.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        d = _F.pad(d, (r, r, 0, 0), mode="reflect")
        d = _F.conv2d(d, kh, groups=3)
        d = _F.pad(d, (0, 0, r, r), mode="reflect")
        d = _F.conv2d(d, kv, groups=3)
        return d

    align_dir = None
    if structure_align:
        r = structure_align_radius
        lum = 0.299 * x0[:, 0:1] + 0.587 * x0[:, 1:2] + 0.114 * x0[:, 2:3]
        gx = _F.pad(lum[..., :, 1:] - lum[..., :, :-1], (0, 1, 0, 0), mode="replicate")
        gy = _F.pad(lum[..., 1:, :] - lum[..., :-1, :], (0, 0, 0, 1), mode="replicate")

        def _box(t: torch.Tensor) -> torch.Tensor:
            k = torch.ones(1, 1, 2 * r + 1, 2 * r + 1, device=device, dtype=x0.dtype) / (2 * r + 1) ** 2
            return _F.conv2d(_F.pad(t, (r,) * 4, mode="reflect"), k)

        jxx, jyy, jxy = _box(gx * gx), _box(gy * gy), _box(gx * gy)
        theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy + 1e-6)
        h, w = x0.shape[-2:]
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing="ij"
        )
        step_x = (2.0 / max(w - 1, 1)) * torch.cos(theta)[:, 0]
        step_y = (2.0 / max(h - 1, 1)) * torch.sin(theta)[:, 0]
        align_dir = (xs, ys, step_x, step_y)

    def structure_smooth(d: torch.Tensor) -> torch.Tensor:
        # Three-tap average along the artwork's own local edge direction --
        # noise that runs with brushwork instead of cutting across it. x0
        # never changes, so align_dir is computed once outside the loop.
        if align_dir is None:
            return d
        xs, ys, step_x, step_y = align_dir
        n = d.shape[0]

        def sample(sign: float) -> torch.Tensor:
            grid = torch.stack([xs.unsqueeze(0) + sign * step_x, ys.unsqueeze(0) + sign * step_y], dim=-1)
            return _F.grid_sample(d, grid.expand(n, -1, -1, -1), mode="bilinear", padding_mode="reflection", align_corners=True)

        return (sample(-1.0) + d + sample(1.0)) / 3.0

    epsilon_map = epsilon
    if perceptual_mask:
        # compute_perceptual_mask returns a [mask_low, mask_high] multiplier
        # averaging ~1.0 -- the SAME total budget, redistributed toward
        # already-textured regions and away from smooth ones (see this
        # function's own doc for why that keeps the effect while hiding
        # the pattern). style_cloak.py already implements this; reused
        # directly rather than re-derived.
        epsilon_map = compute_perceptual_mask(x0, mask_low, mask_high) * epsilon

    def hue_project(d: torch.Tensor) -> torch.Tensor:
        if c_hat is None:
            return d
        return (d * c_hat).sum(dim=1, keepdim=True) * c_hat

    def project_step(d: torch.Tensor) -> torch.Tensor:
        # Applied once per iteration to the fresh gradient increment, not
        # to the accumulated delta -- neither blur nor structure_smooth is
        # idempotent (unlike the colour projection, which is), so
        # reapplying either to the same growing delta every step would
        # compound across all `steps` iterations and smear far beyond the
        # intended radius. Smoothing only the increment keeps total
        # smoothing bounded regardless of step count.
        return hue_project(structure_smooth(blur(d)))

    losses = []
    for i in range(steps):
        x_adv = (x0 + delta).clamp(0, 1)
        loss = vae_uncertainty_loss(vae, x_adv, dtype, logvar_weight, recon_weight)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            # Ascend (not descend): we WANT loss (uncertainty/recon-error)
            # to go up, so step in the direction of the gradient, not
            # against it -- the one sign flip that separates this from a
            # normal training step.
            #
            # Project the STEP (not just the accumulated delta) so every
            # iterate stays inside the hue-locked/smooth subspace: the
            # optimizer never explores outside the feasible set, which is
            # what makes this lossless compared to projecting once at the
            # end.
            delta += step_size * project_step(grad.sign())
            # epsilon_map is either the scalar epsilon or a per-pixel
            # tensor (perceptual_mask=True) -- clamp accepts both.
            delta.clamp_(min=-epsilon_map, max=epsilon_map)
            # Re-project after clamping: the epsilon box is axis-aligned in
            # RGB and can push a vector slightly off the colour direction.
            # Hue-projection only (idempotent) -- NOT blur, which would
            # compound across all `steps` iterations if reapplied to the
            # same growing delta every time (see project_step's own note).
            delta.copy_(hue_project(delta))
        if i % 10 == 0 or i == steps - 1:
            losses.append(float(loss.item()))
            print(f"  step {i:3d}  loss={loss.item():.6f}", flush=True)

    x_final = (x0 + delta).clamp(0, 1)
    save_tensor_image(x_final, output_path)
    print(
        f"[vae_uncertainty_attack] wrote {output_path} (hue_lock={hue_lock}, smooth_sigma={smooth_sigma}, "
        f"structure_align={structure_align}, perceptual_mask={perceptual_mask})",
        flush=True,
    )
    return {"loss_trace": losses, "final_loss": losses[-1] if losses else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--native", action="store_true", help="attack at the image's own resolution instead of --size; no restoration step needed afterward")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--logvar-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--hue-lock", action="store_true")
    parser.add_argument("--hue-lock-radius", type=int, default=2)
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    parser.add_argument("--structure-align", action="store_true")
    parser.add_argument("--structure-align-radius", type=int, default=3)
    parser.add_argument("--perceptual-mask", action="store_true")
    parser.add_argument("--mask-low", type=float, default=0.3)
    parser.add_argument("--mask-high", type=float, default=1.7)
    args = parser.parse_args()

    vae_uncertainty_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        size=None if args.native else args.size,
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        logvar_weight=args.logvar_weight,
        recon_weight=args.recon_weight,
        hue_lock=args.hue_lock,
        hue_lock_radius=args.hue_lock_radius,
        smooth_sigma=args.smooth_sigma,
        structure_align=args.structure_align,
        structure_align_radius=args.structure_align_radius,
        perceptual_mask=args.perceptual_mask,
        mask_low=args.mask_low,
        mask_high=args.mask_high,
    )
