"""Wraps the existing letterboxed-square latent-space ASPL attacks
(aspl_attack_latent.py / aspl_attack_sdxl_only_latent.py) so they can be
chained into a native-resolution pipeline without breaking the
output-resolution-must-match-original requirement.

Why this exists: aspl_attack_latent.py and aspl_attack_sdxl_only_latent.py
both operate on a fixed size x size letterboxed canvas (VAE latent space
needs a fixed-shape input) and write a size x size square PNG -- directly
unusable as a final deliverable for a 1920x1080 original. This module
runs the underlying attack unmodified, then recovers a native-resolution
result the same way native_lowfreq_attack.py/native_aspl_sdxl_attack.py
already do for pixel-space attacks: take the DELTA the attack produced
(not the raw decoded pixels), crop out the letterbox padding, upsample
the delta to native resolution, and add it onto the true native-resolution
input -- so the final output is always native_input.size, exactly.

This is deliberately a wrapper, not a reimplementation: the actual latent
attack (VAE encode -> PGD in latent space -> VAE decode) is untouched,
imported directly from the existing modules.
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from aspl_attack import ASPLAttacker
from aspl_attack_latent import aspl_attack_latent, ASPL_PRESETS
from aspl_attack_sdxl_only import SDXL_ONLY_PRESETS
from aspl_attack_sdxl_only_latent import aspl_attack_sdxl_only_latent
from style_cloak import letterbox_content_box, letterbox_resize, load_image_tensor, save_tensor_image


def _native_wrap(native_input_path: str, square_output_path: str, size: int, output_path: str) -> None:
    native_img = Image.open(native_input_path).convert("RGB")
    w, h = native_img.size

    square_input = letterbox_resize(native_img, size)
    square_output = Image.open(square_output_path).convert("RGB")

    import numpy as np

    a = torch.from_numpy(np.array(square_input).astype("float32") / 255.0)
    b = torch.from_numpy(np.array(square_output).astype("float32") / 255.0)
    delta_square = (b - a).permute(2, 0, 1).unsqueeze(0)  # 1x3xSxS

    left, top, right, bottom = letterbox_content_box(w, h, size)
    delta_content = delta_square[:, :, top:bottom, left:right]
    delta_native = F.interpolate(delta_content, size=(h, w), mode="bicubic", align_corners=False)

    native_arr = torch.from_numpy(np.array(native_img).astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0)
    final = (native_arr + delta_native).clamp(0, 1)
    save_tensor_image(final, output_path)
    print(f"[native_latent_topup] wrote {output_path} ({w}x{h}, native, via {size}x{size} letterboxed latent attack)")


def native_latent_topup_sd15(
    original_path: str, checkpoint_path: str, prompt: str, output_path: str,
    preset_name: str = "L3_ANTI_TRAIN", size: int = 1024, seed: int = 0, latent_epsilon: float = 0.15,
) -> None:
    tmp_square = output_path + ".square_tmp.png"
    aspl_attack_latent(original_path, checkpoint_path, prompt, tmp_square, preset_name, size, seed, latent_epsilon)
    _native_wrap(original_path, tmp_square, size, output_path)


def native_latent_topup_sdxl(
    original_path: str, sdxl_checkpoint: str, prompt: str, output_path: str,
    preset_name: str = "SDXL_FULL", seed: int = 0, latent_epsilon: float = 0.15,
) -> None:
    tmp_square = output_path + ".square_tmp.png"
    size = SDXL_ONLY_PRESETS[preset_name].size
    aspl_attack_sdxl_only_latent(original_path, sdxl_checkpoint, prompt, tmp_square, preset_name, seed, latent_epsilon)
    _native_wrap(original_path, tmp_square, size, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["sd15", "sdxl"])
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-epsilon", type=float, default=0.15)
    args = parser.parse_args()

    if args.arch == "sd15":
        native_latent_topup_sd15(
            args.original, args.checkpoint, args.prompt, args.output,
            preset_name=args.preset or "L3_ANTI_TRAIN", size=args.size, seed=args.seed, latent_epsilon=args.latent_epsilon,
        )
    else:
        native_latent_topup_sdxl(
            args.original, args.checkpoint, args.prompt, args.output,
            preset_name=args.preset or "SDXL_FULL", seed=args.seed, latent_epsilon=args.latent_epsilon,
        )
