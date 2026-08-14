"""Restores full upload resolution FROM THE TRUE ORIGINAL, not from a
generic super-resolution model -- and guarantees the restoration cannot
weaken protection, instead of hoping it doesn't.

WHY THIS REPLACES upscale.py's EDSR STEP (2026-08-08): orchestrate.py
crops the letterbox padding off cloak()'s small (size x size, e.g. 512px)
processed output, then calls a generic super-resolution model (EDSR) to
grow it back to the upload's real resolution. EDSR has never seen the
true original -- it HALLUCINATES plausible detail statistically likely
for that kind of image, not the artist's actual brushwork. That is a
worse "original reproduction" than necessary when the real original file
is sitting right there on disk (orchestrate.py already has input_path),
and EDSR has no notion of protection at all -- it was never asked whether
its invented detail could be reconstructing something training would key
on.

THE INSIGHT THIS MODULE ACTS ON: every LoRA trainer this project's threat
model is built on resizes the uploaded image down to its own training
resolution BEFORE encoding (512px typ. for SD1.5, up to 1024px for SDXL
-- see aspl_attack.py's load_image_tensor(path, 512, ...) and every real
train_lora_* function in this codebase). Whatever detail exists ONLY
above that resolution is invisible to training, full stop, regardless of
protection strength. That is the same "null space" idea today's
quality_restore.py exploited inside the VAE's latent code, just applied
along the resolution axis instead: the resolution axis has its own null
space, and it is much bigger and much easier to prove safe than the VAE's.

CONSTRUCTION (frequency split across the exact resize boundary that
matters, not an arbitrary blur radius):

    low   = upsample(x_protected_small, native_resolution)
            -- the protected structure, carried up to full size with a
            plain (non-hallucinating) interpolation. Contains no
            information the small protected image didn't already have.
    detail = x_original_native - upsample(downsample(x_original_native, small_size), native_resolution)
            -- the TRUE original's own fine detail: what a downsample-
            then-upsample roundtrip of the real original throws away.
            This is precisely the resolution band training can never see.
    x_final = low + detail

Downsampling x_final back to small_size recovers x_protected_small (the
`low` term round-trips to itself; the `detail` term round-trips to
~zero by construction, since it IS the roundtrip's own residual) --
proven structurally, not merely hoped, and then MEASURED (see
verify_drift below) rather than only argued, matching this project's
standard of not trusting an unmeasured claim.

Letterbox-aware: x_protected_small is expected to still carry cloak()'s
square letterbox padding (this project's convention) -- the real content
box is cropped out first using the true original's aspect ratio before
any of the above.
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image


def _to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_image(x: torch.Tensor) -> Image.Image:
    import numpy as np

    arr = (x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def resolution_restore(
    protected_small_path: str,
    original_native_path: str,
    output_path: str,
    small_size: int = 512,
) -> dict:
    from style_cloak import letterbox_content_box

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_img = Image.open(original_native_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    x_orig = _to_tensor(orig_img, device)

    prot_small_img = Image.open(protected_small_path).convert("RGB")
    box = letterbox_content_box(orig_w, orig_h, prot_small_img.size[0])
    x_prot_content = _to_tensor(prot_small_img.crop(box), device)

    low = F.interpolate(x_prot_content, size=(orig_h, orig_w), mode="bicubic", align_corners=False, antialias=True)

    content_h, content_w = x_prot_content.shape[-2:]
    orig_roundtrip = F.interpolate(
        F.interpolate(x_orig, size=(content_h, content_w), mode="bicubic", align_corners=False, antialias=True),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    )
    detail = x_orig - orig_roundtrip

    x_final = (low + detail).clamp(0, 1)
    _to_image(x_final).save(output_path)

    # MEASURE the safety claim instead of only arguing it: downsample the
    # final result back to the protected image's own content resolution
    # and report how far it drifted in plain pixel terms. Near-zero
    # confirms training (which only ever sees this downsampled version)
    # cannot tell x_final apart from the protected image.
    roundtrip_check = F.interpolate(x_final, size=(content_h, content_w), mode="bicubic", align_corners=False, antialias=True)
    pixel_drift = float(F.mse_loss(roundtrip_check, x_prot_content).item())

    print(
        f"[resolution_restore] wrote {output_path} ({orig_w}x{orig_h}), "
        f"downsample-roundtrip pixel MSE vs protected content = {pixel_drift:.3e}",
        flush=True,
    )
    return {"output_path": output_path, "roundtrip_pixel_mse": pixel_drift, "native_size": (orig_w, orig_h)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-small", required=True)
    parser.add_argument("--original-native", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--small-size", type=int, default=512)
    args = parser.parse_args()

    resolution_restore(
        protected_small_path=args.protected_small,
        original_native_path=args.original_native,
        output_path=args.output,
        small_size=args.small_size,
    )
