"""Native-resolution protection that optimizes THROUGH the training
pipeline's own resize, made robust across resizes by EOT.

Solves two failures at once, both measured for real on 2026-08-08:

  1. RESOLUTION. vae_uncertainty_attack.py works at a fixed small
     letterboxed square (512x512) and its output is therefore 512px.
     Three attempts to get native resolution out of it all failed:
       - resolution_restore.py rebuilding native from the 512 protected
         image + the true original's high-frequency detail, verified with
         torch's bicubic:      delta +0.1594 -> +0.0808
       - same, verified with PIL BICUBIC to exactly match the training
         pipeline's own resize:  delta +0.1556 -> +0.0519 (worse)
       - attacking directly at 1920x1080:  delta +0.1632 -> -0.0214,
         i.e. actively counter-productive
     The third result explains all three. Training resizes ANY upload
     down to its own resolution before encoding. A perturbation
     optimized in 1920x1080 pixel coordinates is spatially fine-grained
     relative to a 3.75x downsample, so most of it averages away -- and
     what survives is smoothed (structure_align/blur make the image
     *cleaner* after downsampling), which makes the artwork EASIER to
     learn. The perturbation was optimized in a coordinate frame training
     never sees.

  2. BRITTLENESS, which the numbers exposed almost by accident. In
     attempt 2 the restored image differed from the protected one by a
     round-trip pixel MSE of 1.24e-5 -- RMS under 1/255, against an
     epsilon of 0.03 (7.65/255), so ~12% relative error. That destroyed
     two thirds of the effect. An attack that fragile does not survive a
     JPEG save, a thumbnail, or any resize a real scraper applies. Fixing
     only resolution would have shipped something that looks validated
     and fails in the field.

THE FIX, one idea for both: keep delta at native resolution (so the
OUTPUT is native, no reconstruction step to get subtly wrong), but
compute the loss on the view TRAINING ACTUALLY SEES -- by putting a
differentiable resize inside the attack loop. This is the same
"constrain in-loop, never post-hoc" principle already validated three
times today ([[feedback-inloop-not-posthoc]]), applied to the resize
itself instead of to the perturbation's colour or smoothness.

Then, rather than matching one exact resize kernel (which attempt 2
proved is both fragile and insufficient), sample a DISTRIBUTION of
plausible pipelines each step -- different target resolutions (512 for
SD1.5, up to 1024 for SDXL), different interpolation kernels, antialias
on and off. Expectation-over-transformation: the perturbation must work
after *any* of them, not one. This project already uses EOT for the same
reason in style_cloak.py (`random_resize_round_trip`, the `--eot` flag).

A useful consequence, not a coincidence: to survive a 3.75x downsample a
perturbation has to be spatially COHERENT at native scale -- broad, smooth
modulation rather than per-pixel noise. The robustness constraint
therefore forces exactly the visual character today's hue_lock /
structure_align work was chasing by hand. Being robust and being
invisible turn out to be the same requirement here.

The visual constraints from vae_uncertainty_attack (hue_lock,
structure_align, perceptual_mask -- all validated free or better there,
delta +0.1714 combined) are reused unchanged, applied to the native
delta.

STATUS: new 2026-08-08, unvalidated. Every quantitative claim above is
about the OLD attacks; this module's own numbers do not exist yet. Run a
real train+score before believing anything about it -- three plausible
fixes have already failed today, two of them with clean-looking proxy
measurements.
"""

import argparse
import random

import torch
import torch.nn.functional as F
from PIL import Image

from style_cloak import compute_perceptual_mask
from vae_uncertainty_attack import load_vae, vae_uncertainty_loss

# Gray letterbox fill, matching style_cloak.letterbox_resize's (114,114,114).
_PAD_VALUE = 114 / 255.0

# Resolutions a real trainer might use: 512 is SD1.5's native and this
# project's validated default; 1024 is SDXL's. 640/768 cover the bucketed
# resolutions kohya-style trainers commonly generate in between.
_EOT_SIZES = (512, 512, 640, 768, 1024)
# antialias is only meaningful for the downsampling directions here; both
# settings appear in real tooling (PIL always antialiases, raw torch
# interpolate does not by default), so the perturbation must survive both.
_EOT_MODES = (("bicubic", True), ("bicubic", False), ("bilinear", True), ("area", False))


def load_native(path: str, device: torch.device) -> torch.Tensor:
    import numpy as np

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def save_native(x: torch.Tensor, path: str) -> None:
    import numpy as np

    arr = (x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def differentiable_letterbox(x: torch.Tensor, size: int, mode: str, antialias: bool) -> torch.Tensor:
    """Differentiable equivalent of style_cloak.letterbox_resize: fit the
    long edge to `size` preserving aspect, then pad to a `size` square
    with the same neutral gray. Gradients flow back to `x`, which is what
    lets the optimizer shape a native-resolution perturbation according to
    what survives this exact operation."""
    _, _, h, w = x.shape
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    # antialias is only supported (and only meaningful) for bilinear/bicubic
    # downsampling; `area` is inherently averaging.
    kwargs = {"antialias": antialias} if mode in ("bilinear", "bicubic") else {}
    resized = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=False, **kwargs) if mode != "area" else F.interpolate(x, size=(new_h, new_w), mode="area")

    canvas = torch.full((x.shape[0], 3, size, size), _PAD_VALUE, device=x.device, dtype=x.dtype)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[:, :, top : top + new_h, left : left + new_w] = resized
    return canvas


def native_eot_attack(
    original_path: str,
    checkpoint_path: str,
    output_path: str,
    epsilon: float = 0.03,
    steps: int = 150,
    step_size: float | None = None,
    eot_samples: int = 2,
    logvar_weight: float = 1.0,
    recon_weight: float = 0.0,
    hue_lock: bool = True,
    hue_lock_radius: int = 2,
    structure_align: bool = True,
    structure_align_radius: int = 3,
    perceptual_mask: bool = True,
    mask_low: float = 0.3,
    mask_high: float = 1.7,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """recon_weight defaults to 0 here (unlike vae_uncertainty_attack's
    1.0): the reconstruction term needs a full VAE *decode* per view, and
    with eot_samples views per step that was the term that OOM'd an A40 at
    native resolution. The logvar term needs only the encoder. Whether the
    reconstruction term was pulling real weight is measurable -- run it at
    a smaller eot_samples if the logvar-only result underperforms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    vae = load_vae(checkpoint_path, device, dtype)

    x0 = load_native(original_path, device)
    delta = torch.zeros_like(x0, requires_grad=True)
    step_size = step_size if step_size is not None else epsilon / 4

    # --- constraint setup, all in NATIVE coordinates -------------------
    c_hat = None
    if hue_lock:
        r = hue_lock_radius
        k = 2 * r + 1
        kernel = torch.ones(3, 1, k, k, device=device, dtype=x0.dtype) / (k * k)
        local = F.conv2d(F.pad(x0, (r,) * 4, mode="reflect"), kernel, groups=3)
        c_hat = local / (local.norm(dim=1, keepdim=True) + 1e-6)

    align_dir = None
    if structure_align:
        r = structure_align_radius
        lum = 0.299 * x0[:, 0:1] + 0.587 * x0[:, 1:2] + 0.114 * x0[:, 2:3]
        gx = F.pad(lum[..., :, 1:] - lum[..., :, :-1], (0, 1, 0, 0), mode="replicate")
        gy = F.pad(lum[..., 1:, :] - lum[..., :-1, :], (0, 0, 0, 1), mode="replicate")
        bk = torch.ones(1, 1, 2 * r + 1, 2 * r + 1, device=device, dtype=x0.dtype) / (2 * r + 1) ** 2

        def _box(t):
            return F.conv2d(F.pad(t, (r,) * 4, mode="reflect"), bk)

        jxx, jyy, jxy = _box(gx * gx), _box(gy * gy), _box(gx * gy)
        theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy + 1e-6)
        h, w = x0.shape[-2:]
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing="ij"
        )
        align_dir = (xs, ys, (2.0 / max(w - 1, 1)) * torch.cos(theta)[:, 0], (2.0 / max(h - 1, 1)) * torch.sin(theta)[:, 0])

    epsilon_map = epsilon
    if perceptual_mask:
        epsilon_map = compute_perceptual_mask(x0, mask_low, mask_high) * epsilon

    def hue_project(d):
        if c_hat is None:
            return d
        return (d * c_hat).sum(dim=1, keepdim=True) * c_hat

    def align_smooth(d):
        if align_dir is None:
            return d
        xs, ys, sx, sy = align_dir
        n = d.shape[0]

        def sample(sign):
            grid = torch.stack([xs.unsqueeze(0) + sign * sx, ys.unsqueeze(0) + sign * sy], dim=-1)
            return F.grid_sample(d, grid.expand(n, -1, -1, -1), mode="bilinear", padding_mode="reflection", align_corners=True)

        return (sample(-1.0) + d + sample(1.0)) / 3.0

    # --- PGD, loss evaluated through the training-pipeline resize ------
    losses = []
    for i in range(steps):
        x_adv = (x0 + delta).clamp(0, 1)

        total = torch.zeros((), device=device)
        views = []
        for _ in range(eot_samples):
            size = rng.choice(_EOT_SIZES)
            mode, aa = rng.choice(_EOT_MODES)
            views.append((size, mode, aa))
            train_view = differentiable_letterbox(x_adv, size, mode, aa)
            total = total + vae_uncertainty_loss(vae, train_view, dtype, logvar_weight, recon_weight)
        loss = total / eot_samples

        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta += step_size * hue_project(align_smooth(grad.sign()))
            delta.clamp_(min=-epsilon_map, max=epsilon_map)
            delta.copy_(hue_project(delta))

        if i % 10 == 0 or i == steps - 1:
            losses.append(float(loss.item()))
            print(f"  step {i:3d}  loss={loss.item():.6f}  views={views}", flush=True)

    x_final = (x0 + delta).clamp(0, 1)
    save_native(x_final, output_path)
    h, w = x0.shape[-2:]
    print(f"[native_eot_attack] wrote {output_path} ({w}x{h}, native)", flush=True)
    return {"loss_trace": losses, "final_loss": losses[-1] if losses else None, "native_size": (w, h)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--eot-samples", type=int, default=2)
    parser.add_argument("--logvar-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=0.0)
    parser.add_argument("--no-hue-lock", action="store_true")
    parser.add_argument("--no-structure-align", action="store_true")
    parser.add_argument("--no-perceptual-mask", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    native_eot_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        eot_samples=args.eot_samples,
        logvar_weight=args.logvar_weight,
        recon_weight=args.recon_weight,
        hue_lock=not args.no_hue_lock,
        structure_align=not args.no_structure_align,
        perceptual_mask=not args.no_perceptual_mask,
        seed=args.seed,
    )
