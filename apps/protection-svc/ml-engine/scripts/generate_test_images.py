"""Generates two synthetic test images so the PoC is self-contained and
reproducible without needing real copyrighted artwork:

  original.png     - a flat-color, clean-edge "illustration" standing in for
                      a creator's artwork (the thing we're protecting).
  style_target.png - a noisy, high-texture "painterly" pattern standing in
                      for a distinct art style we cloak *towards*.

Real usage (Phase 2) swaps `original.png` for an actual uploaded artwork; the
cloaking algorithm and metrics don't change.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 256
OUT_DIR = "out"


def make_original() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (245, 240, 230))
    draw = ImageDraw.Draw(img)

    # Flat-color shapes with clean edges, like simplified anime/illustration
    # line art (the kind of style Glaze targets protecting).
    draw.ellipse([40, 40, 200, 200], fill=(235, 150, 160), outline=(60, 40, 40), width=4)
    draw.polygon([(128, 30), (220, 220), (36, 220)], fill=(150, 190, 220), outline=(30, 40, 60), width=3)
    draw.rectangle([90, 90, 170, 170], fill=(250, 210, 90), outline=(80, 60, 20), width=3)
    return img


def make_style_target() -> Image.Image:
    rng = np.random.default_rng(42)

    # Blotchy multi-frequency noise, like impasto brushwork, in a very
    # different palette from the "original" so style drift is visible.
    low_freq = rng.uniform(0, 1, (SIZE // 16, SIZE // 16, 3))
    low_freq_img = Image.fromarray((low_freq * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)

    high_freq = rng.normal(0, 1, (SIZE, SIZE, 3))
    high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min())
    high_freq_img = Image.fromarray((high_freq * 255).astype(np.uint8))

    blended = Image.blend(low_freq_img.convert("RGB"), high_freq_img.convert("RGB"), alpha=0.35)
    painterly = blended.filter(ImageFilter.GaussianBlur(radius=2))

    # Push toward a cool teal/violet palette distinct from `original`'s
    # warm palette, so "style drift toward this target" is meaningful.
    arr = np.array(painterly).astype(np.float32)
    arr[:, :, 0] *= 0.6  # less red
    arr[:, :, 2] *= 1.2  # more blue
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    make_original().save(f"{OUT_DIR}/original.png")
    make_style_target().save(f"{OUT_DIR}/style_target.png")
    print(f"wrote {OUT_DIR}/original.png and {OUT_DIR}/style_target.png")
