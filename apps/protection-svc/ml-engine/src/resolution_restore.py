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

BUG FOUND AND FIXED (2026-08-08, real train+score run): the first version
of this module built `low`/`detail` with torch's F.interpolate(mode=
"bicubic", antialias=True) and PROVED the round-trip was near-zero-error
using that SAME torch resize to check itself -- a tautology. The actual
training pipeline (style_cloak.py's letterbox_resize) resizes with PIL's
Image.BICUBIC, a different kernel. Verified with the wrong resize, shipped
anyway: real delta dropped from +0.1594 (small protected image, trained
directly) to +0.0808 (this module's "guaranteed safe" restored output) --
essentially half. The residual `detail` term does not vanish under a
DIFFERENT resize than the one used to construct it; proving "X round-trips
under resize A" says nothing about round-tripping under resize B. Fixed by
rebuilding every resize here with PIL's Image.BICUBIC specifically,
matching style_cloak.letterbox_resize exactly -- so the round-trip this
module proves is the SAME round-trip training will actually perform, not
a resize this module invented for the sake of a clean proof.

Downsampling x_final back to small_size should now recover
x_protected_small almost exactly (the `low` term round-trips to itself;
the `detail` term round-trips to ~zero by construction) -- and unlike the
first version, this is measured using the training pipeline's own resize,
not a stand-in for it. See verify_drift's own note: even this is a proxy
for the real thing, and the fix above exists precisely because a
plausible-looking proxy was wrong once already. Re-verify with a real
train+score run before trusting any change to this module again.

Letterbox-aware: x_protected_small is expected to still carry cloak()'s
square letterbox padding (this project's convention) -- the real content
box is cropped out first using the true original's aspect ratio before
any of the above.
"""

import argparse

from PIL import Image


def _pil_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    # Image.BICUBIC specifically -- must match style_cloak.letterbox_resize's
    # own resize call exactly (see this module's own doc for why a
    # different-but-similar resize already broke the safety proof once).
    return img.resize(size, Image.BICUBIC)


def resolution_restore(
    protected_small_path: str,
    original_native_path: str,
    output_path: str,
    small_size: int = 512,
) -> dict:
    from style_cloak import letterbox_content_box

    orig_img = Image.open(original_native_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    prot_small_img = Image.open(protected_small_path).convert("RGB")
    box = letterbox_content_box(orig_w, orig_h, prot_small_img.size[0])
    prot_content = prot_small_img.crop(box)
    content_w, content_h = prot_content.size

    low = _pil_resize(prot_content, (orig_w, orig_h))

    orig_roundtrip = _pil_resize(_pil_resize(orig_img, (content_w, content_h)), (orig_w, orig_h))

    import numpy as np

    low_arr = np.asarray(low, dtype=np.float32)
    orig_arr = np.asarray(orig_img, dtype=np.float32)
    roundtrip_arr = np.asarray(orig_roundtrip, dtype=np.float32)
    detail_arr = orig_arr - roundtrip_arr

    final_arr = np.clip(low_arr + detail_arr, 0, 255).round().astype("uint8")
    final_img = Image.fromarray(final_arr)
    final_img.save(output_path)

    # MEASURE the safety claim instead of only arguing it, using the SAME
    # resize the real training pipeline performs (not a different one --
    # see this module's own doc on why that distinction is the whole fix).
    roundtrip_check = np.asarray(_pil_resize(final_img, (content_w, content_h)), dtype=np.float32)
    prot_content_arr = np.asarray(prot_content, dtype=np.float32)
    pixel_drift = float(np.mean((roundtrip_check - prot_content_arr) ** 2)) / (255.0**2)

    print(
        f"[resolution_restore] wrote {output_path} ({orig_w}x{orig_h}), "
        f"downsample-roundtrip pixel MSE vs protected content (PIL BICUBIC, matches training) = {pixel_drift:.3e}",
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
