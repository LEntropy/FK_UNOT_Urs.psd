"""Makes the protective perturbation BLEND INTO the artwork's own colour
and mood instead of reading as foreign speckle -- without giving back
protection.

THE OBSERVATION THIS ANSWERS: the pattern left by every attack in this
project is visible less because of its *magnitude* than because of its
*direction*. An epsilon-bounded PGD step is free to move each pixel
anywhere in RGB, so it routinely pushes a pixel in a direction the
artwork's own palette never goes -- magenta flecks in a blue night scene,
green speckle on skin. The human visual system is extremely good at
spotting exactly that: a hue that does not belong reads as damage, while
the SAME magnitude of change expressed as brightness variation within the
existing hue reads as brushwork, grain, or lighting.

So: keep the perturbation, change the basis it lives in.

  hue_lock (the flagship): project each pixel's perturbation onto that
      pixel's own colour direction. delta' = (delta . c_hat) c_hat, where
      c_hat is the local colour unit vector. The perturbation can then
      only brighten or darken the colour already there -- it becomes
      structurally incapable of introducing a hue the artwork does not
      contain. This is what "동화" means made precise.
  chroma_fold: move perturbation energy out of chroma (Cb/Cr) into luma
      (Y). Human chroma acuity is far below luma acuity, so the same
      energy is markedly less visible as brightness texture than as
      colour fringing -- and colour fringing is specifically what reads
      as "번짐".
  luma_scale: rescale the perturbation proportionally to local
      brightness, turning an additive perturbation into a roughly
      multiplicative one. Matches Weber's law (the eye judges contrast
      ratios, not absolute differences), so noise stops glowing in dark
      regions -- which is where flat additive noise is most obvious.
  structure_align: steer the perturbation along the local edge
      orientation (from the structure tensor) rather than across it.
      Noise running along brushstrokes looks like brushwork; noise
      cutting across them looks like damage.

WHY THIS IS ALLOWED TO WORK AT ALL: every transform here changes the
perturbation, therefore changes the latent, therefore could in principle
cost protection. What makes it viable is that this is not a free-for-all
-- it is spending a MEASURED budget. drift_calibration.py (2026-08-08)
swept latent drift against real SD1.5 train+score delta on a real
illustration and found protection intact (in fact slightly stronger) out
to drift ~16-25, only collapsing past ~50. quality_restore.py's
DRIFT_GATE encodes that. These transforms aim to spend that same
headroom on *harmony* rather than on moving back toward the original.

STATUS: new 2026-08-08, unvalidated. Latent drift is reported for every
variant, and drift under the calibrated gate is a strong argument that
training cannot tell the difference -- but only a real train-and-score
run proves it. Do that before shipping any of these.

THE PRINCIPLED NEXT STEP, not done here: rather than harmonizing a
perturbation after the fact, parametrize the attack's delta in the
harmonized basis from the start, so PGD optimizes protection strength
*within* the space of harmonious perturbations and nothing has to be
given back afterwards. Post-hoc harmonization can only redistribute what
the attack already chose; in-attack parametrization would let the
optimizer find the strongest perturbation that was harmonious to begin
with.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from style_cloak import load_image_tensor, save_tensor_image

_EPS = 1e-6


def _box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    c = x.shape[1]
    kernel = torch.ones(c, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    return F.conv2d(F.pad(x, (radius,) * 4, mode="reflect"), kernel, groups=c)


def _luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def harmonize_hue_lock(x_orig: torch.Tensor, delta: torch.Tensor, alpha: float = 1.0, radius: int = 2) -> torch.Tensor:
    """Projects the perturbation onto each pixel's own colour direction, so
    it can only modulate the brightness of the colour already present and
    never introduce a foreign hue.

    The colour direction is taken from a small local average rather than
    the single pixel: a lone pixel's colour is noisy, and locking to it
    would make the projection itself high-frequency (a new artifact).
    """
    local_colour = _box_blur(x_orig, radius)
    c_hat = local_colour / (local_colour.norm(dim=1, keepdim=True) + _EPS)
    projected = (delta * c_hat).sum(dim=1, keepdim=True) * c_hat
    return (1 - alpha) * delta + alpha * projected


def harmonize_chroma_fold(x_orig: torch.Tensor, delta: torch.Tensor, chroma_keep: float = 0.25) -> torch.Tensor:
    """Attenuates the perturbation's chroma components, folding the removed
    energy back into luma so the total magnitude is roughly preserved.

    Operates on the delta directly: RGB->YCbCr is linear, so it applies to
    a difference just as it does to an image (no +0.5 offset needed here,
    unlike converting an actual image).
    """
    dy = _luma(delta)
    dcb = (delta[:, 2:3] - dy) * 0.564
    dcr = (delta[:, 0:1] - dy) * 0.713

    dcb, dcr = dcb * chroma_keep, dcr * chroma_keep
    # Push the energy taken out of chroma back into luma so the
    # perturbation doesn't simply get quieter (which would trade
    # protection for visibility rather than redistributing it).
    lost = (1 - chroma_keep)
    dy = dy * (1 + lost * 0.5)

    r = dy + 1.403 * dcr
    b = dy + 1.773 * dcb
    g = (dy - 0.299 * r - 0.114 * b) / 0.587
    return torch.cat([r, g, b], dim=1)


def harmonize_luma_scale(x_orig: torch.Tensor, delta: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Scales the perturbation by local brightness, making it behave
    multiplicatively (Weber's law) instead of additively -- so it stops
    glowing out of dark regions, where flat additive noise is most
    conspicuous. Normalized by the mean so total magnitude is preserved.
    """
    lum = _box_blur(_luma(x_orig), 2)
    scale = (lum + 0.05) / (lum.mean() + 0.05)
    scale = 1 + strength * (scale - 1)
    return delta * scale


def harmonize_structure_align(x_orig: torch.Tensor, delta: torch.Tensor, alpha: float = 1.0, radius: int = 3) -> torch.Tensor:
    """Steers the perturbation along the local edge orientation instead of
    across it, via the structure tensor's dominant direction -- noise that
    runs with the brushwork instead of cutting across it.

    Implemented as an oriented one-dimensional smoothing of the delta
    along the local edge direction: rather than rotating each perturbation
    vector (which is a colour-space operation and would fight hue_lock),
    this smears it spatially so its own texture follows the artwork's.
    """
    lum = _luma(x_orig)
    gx = lum[..., :, 1:] - lum[..., :, :-1]
    gy = lum[..., 1:, :] - lum[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")

    jxx = _box_blur(gx * gx, radius)
    jyy = _box_blur(gy * gy, radius)
    jxy = _box_blur(gx * gy, radius)

    # Dominant orientation of the structure tensor, as a unit vector.
    theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy + _EPS)
    ux, uy = torch.cos(theta), torch.sin(theta)

    # Three-tap smoothing along (ux, uy) by sampling the delta one step
    # forward and back along that direction, via grid_sample.
    n, _, h, w = delta.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=delta.device),
        torch.linspace(-1, 1, w, device=delta.device),
        indexing="ij",
    )
    step_x = (2.0 / max(w - 1, 1)) * ux[:, 0]
    step_y = (2.0 / max(h - 1, 1)) * uy[:, 0]

    def sample(sign: float) -> torch.Tensor:
        grid = torch.stack([xs.unsqueeze(0) + sign * step_x, ys.unsqueeze(0) + sign * step_y], dim=-1)
        return F.grid_sample(delta, grid.expand(n, -1, -1, -1), mode="bilinear", padding_mode="reflection", align_corners=True)

    smoothed = (sample(-1.0) + delta + sample(1.0)) / 3.0
    return (1 - alpha) * delta + alpha * smoothed


TRANSFORMS = {
    "hue_lock": harmonize_hue_lock,
    "chroma_fold": harmonize_chroma_fold,
    "luma_scale": harmonize_luma_scale,
    "structure_align": harmonize_structure_align,
}


def apply_chain(x_orig: torch.Tensor, x_prot: torch.Tensor, chain: list[tuple[str, dict]]) -> torch.Tensor:
    delta = x_prot - x_orig
    for name, kwargs in chain:
        delta = TRANSFORMS[name](x_orig, delta, **kwargs)
    return (x_orig + delta).clamp(0, 1)


def run_all(original_path: str, protected_path: str, checkpoint_path: str, out_prefix: str, size: int = 512) -> dict:
    from quality_restore import DRIFT_GATE, LatentProbe, Perceptual
    from vae_uncertainty_attack import load_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(checkpoint_path, device, torch.float32)
    probe = LatentProbe(vae, torch.float32)
    percep = Perceptual(device)

    x_orig = load_image_tensor(original_path, size, device)
    x_prot = load_image_tensor(protected_path, size, device)
    probe.set_target(x_prot)
    base_percep = percep.value(x_prot, x_orig)
    print(f"[harmonize] baseline perceptual(protected, original) = {base_percep:.5f}", flush=True)

    chains = {
        "hue_lock_full": [("hue_lock", {"alpha": 1.0})],
        "hue_lock_half": [("hue_lock", {"alpha": 0.5})],
        "chroma_fold": [("chroma_fold", {"chroma_keep": 0.25})],
        "luma_scale": [("luma_scale", {"strength": 1.0})],
        "structure_align": [("structure_align", {"alpha": 1.0})],
        # Combined: lock the hue, quiet the remaining chroma, then let
        # local brightness decide how loud it gets. Each step targets a
        # different reason the pattern is visible, so they compose rather
        # than compete (the same "don't sum objectives into one budget"
        # rule hybrid_protect.py is built on).
        "harmony_full": [
            ("hue_lock", {"alpha": 1.0}),
            ("chroma_fold", {"chroma_keep": 0.25}),
            ("luma_scale", {"strength": 1.0}),
        ],
        "harmony_full_aligned": [
            ("hue_lock", {"alpha": 1.0}),
            ("chroma_fold", {"chroma_keep": 0.25}),
            ("luma_scale", {"strength": 1.0}),
            ("structure_align", {"alpha": 0.6}),
        ],
    }

    results = {}
    for name, chain in chains.items():
        x = apply_chain(x_orig, x_prot, chain)
        drift = probe.drift_value(x)
        p = percep.value(x, x_orig)
        path = f"{out_prefix}_{name}.png"
        save_tensor_image(x, path)
        ok = drift <= DRIFT_GATE
        results[name] = {
            "latent_drift": drift,
            "perceptual": p,
            "perceptual_gain": base_percep - p,
            "passes_drift_gate": ok,
        }
        print(
            f"  [{'OK ' if ok else 'REJECT'}] {name:22s} drift={drift:.3e}  percep={p:.5f}  gain={base_percep - p:+.5f}",
            flush=True,
        )

    return {"baseline_perceptual": base_percep, "drift_gate": DRIFT_GATE, "candidates": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--protected", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    out = run_all(args.original, args.protected, args.checkpoint, args.out_prefix, args.size)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
