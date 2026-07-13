"""Glaze-style "style confusion" cloaking PoC.

Implements the optimization described in PROJECT_DESIGN.md §3-3 / §8:

    maximize   Feature_Drift (toward a different style's Gram-matrix
               representation)
    subject to Perceptual_Distance < epsilon (bounded pixel-space
               perturbation, so the image looks unchanged to a human)

This is a simplified, from-scratch reimplementation of the *mechanism*
Glaze/Nightshade-style tools use (adversarial perturbation optimized against
a feature extractor's style representation) — not a copy of their published
implementation, and not tuned to their published fidelity. It's a PoC to
prove the pipeline end-to-end: load image -> optimize -> bounded perturbation
-> style embedding measurably drifts while pixels stay visually unchanged.

Usage:
    python src/style_cloak.py --original out/original.png \\
        --style-target out/style_target.png --preset L3_ANTI_TRAIN
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from model import StyleFeatureExtractor

STYLE_LOSS_WEIGHTS = {
    "relu1_1": 1.0,
    "relu2_1": 1.0,
    "relu3_1": 1.0,
    "relu4_1": 1.0,
    "relu5_1": 1.0,
}


@dataclass
class Preset:
    """Maps to the protection strength presets in PROJECT_DESIGN.md §3-4.
    epsilon is the L-infinity pixel-space perturbation budget (in [0,1]
    image scale); steps is the optimization iteration count.
    """

    epsilon: float
    steps: int
    lr: float


PRESETS = {
    "L1_PREVIEW": Preset(epsilon=0.02, steps=150, lr=0.01),
    "L2_PORTFOLIO": Preset(epsilon=0.04, steps=300, lr=0.01),
    "L3_ANTI_TRAIN": Preset(epsilon=0.08, steps=500, lr=0.01),
}


def image_to_tensor(img: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    """Converts a PIL image (any mode/size) to the 1x3xHxW [0,1] tensor the
    model expects. Split out from load_image_tensor so callers that already
    have a PIL image in memory (e.g. robustness_test.py, after simulating a
    JPEG re-encode) don't have to round-trip through a file.
    """
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(
        __import__("numpy").array(img).astype("float32") / 255.0
    )  # HWC in [0,1]
    return arr.permute(2, 0, 1).unsqueeze(0).to(device)  # 1x3xHxW


def load_image_tensor(path: str, size: int, device: torch.device) -> torch.Tensor:
    return image_to_tensor(Image.open(path), size, device)


def save_tensor_image(x: torch.Tensor, path: str) -> None:
    import numpy as np

    arr = x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


def style_loss(grams_a: dict[str, torch.Tensor], grams_b: dict[str, torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=next(iter(grams_a.values())).device)
    for layer, weight in STYLE_LOSS_WEIGHTS.items():
        total = total + weight * F.mse_loss(grams_a[layer], grams_b[layer])
    return total


def random_resize_round_trip(
    x: torch.Tensor,
    min_scale: float = 0.3,
    max_scale: float = 1.0,
    scales: list[float] | None = None,
) -> torch.Tensor:
    """Differentiable stand-in for "someone's upload pipeline downscaled this
    image" — downsamples to a random scale then back to the original size,
    using torch's own (differentiable) interpolation so gradients can flow
    back through it into `delta`. This is what EOT training actually needs:
    F.interpolate, not PIL, because PIL round-trips aren't part of the
    autograd graph.

    If `scales` is given, sample uniformly from that discrete set instead of
    a continuous [min_scale, max_scale) range — widening the continuous
    range dilutes how often training actually hits a specific troublesome
    scale (e.g. 0.25x), since most draws land elsewhere in the range. A
    discrete set guarantees every listed scale gets trained against roughly
    equally often.
    """
    _, _, h, w = x.shape
    if scales:
        scale = scales[torch.randint(0, len(scales), (1,)).item()]
    else:
        scale = torch.empty(1).uniform_(min_scale, max_scale).item()
    small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
    down = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def cloak(
    original_path: str,
    style_target_path: str,
    output_path: str,
    preset_name: str,
    size: int = 256,
    eot: bool = False,
    eot_samples: int = 2,
    eot_min_scale: float = 0.3,
    eot_max_scale: float = 1.0,
    eot_scales: list[float] | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = PRESETS[preset_name]
    extractor = StyleFeatureExtractor(device)

    original = load_image_tensor(original_path, size, device)
    style_target = load_image_tensor(style_target_path, size, device)
    target_grams = extractor.gram_matrices(style_target)

    delta = torch.zeros_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=preset.lr)

    scale_desc = f"discrete{eot_scales}" if eot_scales else f"[{eot_min_scale},{eot_max_scale}]"
    mode = f"EOT(resize, samples={eot_samples}, scale={scale_desc})" if eot else "no-EOT (clean image only)"
    print(f"[cloak] preset={preset_name} epsilon={preset.epsilon} steps={preset.steps} mode={mode}")
    for step in range(preset.steps):
        optimizer.zero_grad()
        x_adv = (original + delta).clamp(0, 1)

        if eot:
            # Expectation over transformation: average the loss over several
            # random resize round-trips (plus the clean image) instead of
            # optimizing the clean image alone, so the perturbation survives
            # in expectation across the transform distribution, not just at
            # exact original resolution.
            loss = style_loss(extractor.gram_matrices(x_adv), target_grams)
            for _ in range(eot_samples):
                transformed = random_resize_round_trip(x_adv, eot_min_scale, eot_max_scale, eot_scales)
                loss = loss + style_loss(extractor.gram_matrices(transformed), target_grams)
            loss = loss / (eot_samples + 1)
        else:
            loss = style_loss(extractor.gram_matrices(x_adv), target_grams)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-preset.epsilon, preset.epsilon)
            delta.copy_(((original + delta).clamp(0, 1) - original))  # keep x_adv in [0,1]

        if step % 50 == 0 or step == preset.steps - 1:
            print(f"  step {step:4d}  style_loss={loss.item():.6f}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[cloak] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="out/original.png")
    parser.add_argument("--style-target", default="out/style_target.png")
    parser.add_argument("--output", default="out/cloaked.png")
    parser.add_argument("--preset", choices=list(PRESETS), default="L3_ANTI_TRAIN")
    parser.add_argument(
        "--size", type=int, default=256, help="square processing resolution (all presets were tuned at 256)"
    )
    parser.add_argument("--eot", action="store_true", help="optimize against random resize round-trips too")
    parser.add_argument("--eot-samples", type=int, default=2)
    parser.add_argument("--eot-min-scale", type=float, default=0.3)
    parser.add_argument("--eot-max-scale", type=float, default=1.0)
    parser.add_argument(
        "--eot-scales",
        type=str,
        default=None,
        help="comma-separated discrete scales, e.g. '0.25,0.5,1.0' (overrides --eot-min/max-scale)",
    )
    args = parser.parse_args()

    eot_scales = [float(s) for s in args.eot_scales.split(",")] if args.eot_scales else None

    cloak(
        args.original,
        args.style_target,
        args.output,
        args.preset,
        size=args.size,
        eot=args.eot,
        eot_samples=args.eot_samples,
        eot_min_scale=args.eot_min_scale,
        eot_max_scale=args.eot_max_scale,
        eot_scales=eot_scales,
    )
