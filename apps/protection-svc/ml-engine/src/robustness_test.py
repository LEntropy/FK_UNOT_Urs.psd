"""Does the cloak survive what actually happens to an image after upload?

Real platforms re-encode everything: thumbnails get downscaled, most image
hosts re-compress to JPEG regardless of the original format, CDNs sometimes
do both. If a cloaked image only works at its exact original resolution and
bit-for-bit format, the protection is close to theater. This script applies
a handful of realistic transforms and re-measures style drift after each one
to see how much survives.

Usage:
    python src/robustness_test.py --original out/original.png \\
        --cloaked out/cloaked.png --style-target out/style_target.png
"""

import argparse
import io

import torch
from PIL import Image

from evaluate import gram_cosine_similarity, mean_sim
from model import StyleFeatureExtractor
from style_cloak import image_to_tensor

TRANSFORMS: dict[str, "callable"] = {}


def register(name):
    def deco(fn):
        TRANSFORMS[name] = fn
        return fn

    return deco


@register("none")
def _t_none(img: Image.Image) -> Image.Image:
    return img


def _jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


@register("jpeg_q95")
def _t_jpeg95(img: Image.Image) -> Image.Image:
    return _jpeg_recompress(img, 95)


@register("jpeg_q75")
def _t_jpeg75(img: Image.Image) -> Image.Image:
    return _jpeg_recompress(img, 75)


@register("jpeg_q50")
def _t_jpeg50(img: Image.Image) -> Image.Image:
    return _jpeg_recompress(img, 50)


def _resize_round_trip(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


@register("resize_0.5x")
def _t_resize_half(img: Image.Image) -> Image.Image:
    return _resize_round_trip(img, 0.5)


@register("resize_0.25x")
def _t_resize_quarter(img: Image.Image) -> Image.Image:
    return _resize_round_trip(img, 0.25)


@register("sns_pipeline")
def _t_sns(img: Image.Image) -> Image.Image:
    """Downscale to a typical timeline-thumbnail size, then JPEG-recompress —
    approximates what most social platforms actually do on upload."""
    return _jpeg_recompress(_resize_round_trip(img, 0.5), 75)


def run(original_path: str, cloaked_path: str, style_target_path: str, size: int = 256) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StyleFeatureExtractor(device)

    original_img = Image.open(original_path)
    cloaked_img = Image.open(cloaked_path)
    style_target_tensor = image_to_tensor(Image.open(style_target_path), size, device)
    grams_target = extractor.gram_matrices(style_target_tensor)

    baseline_drift = None

    print(f"{'transform':<16} {'orig->target':>12} {'cloaked->target':>16} {'drift':>10} {'retained':>10}")
    for name, transform in TRANSFORMS.items():
        orig_t = image_to_tensor(transform(original_img), size, device)
        cloaked_t = image_to_tensor(transform(cloaked_img), size, device)

        grams_orig = extractor.gram_matrices(orig_t)
        grams_cloaked = extractor.gram_matrices(cloaked_t)

        sim_orig = mean_sim(gram_cosine_similarity(grams_orig, grams_target))
        sim_cloaked = mean_sim(gram_cosine_similarity(grams_cloaked, grams_target))
        drift = sim_cloaked - sim_orig

        if name == "none":
            baseline_drift = drift
            retained_str = "100%  (baseline)"
        else:
            retained_pct = (drift / baseline_drift * 100) if baseline_drift else float("nan")
            retained_str = f"{retained_pct:.0f}%"

        print(f"{name:<16} {sim_orig:>12.4f} {sim_cloaked:>16.4f} {drift:>+10.4f} {retained_str:>10}")

    print()
    print("retained% = drift after transform / drift with no transform.")
    print("Low retained% means the transform washes out the cloak's effect --")
    print("the perturbation was tuned to survive at exact original resolution")
    print("only, which is a real limitation noted in PROJECT_DESIGN.md section 12.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="out/original.png")
    parser.add_argument("--cloaked", default="out/cloaked.png")
    parser.add_argument("--style-target", default="out/style_target.png")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    run(args.original, args.cloaked, args.style_target, args.size)
