"""Creator-opt-in "original preview" derivative (2026-08-10 coin-system
feature). NOT the real original bytes -- a downscaled (long side capped at
PREVIEW_MAX_DIMENSION, never upscaled) copy with a visible tiled watermark
overlaid, meant to sit as close to the real original as this project is
willing to expose publicly (same resolution ceiling as the project's
existing top asset_versions tier, public_preview_2048).

Pure PIL, no torch/model involved -- unlike style_cloak.py's cloak() or
protection_score.py's LoRA training, this never needs GPU/RunPod (see
server.py's /original-preview endpoint, which calls this synchronously on
protection-svc's own CPU).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PREVIEW_MAX_DIMENSION = 2048
WATERMARK_TEXT = "DONTAI · 원본 미리보기 · 재배포 금지"


def _resize_capped(image: Image.Image, max_dimension: int) -> Image.Image:
    """Long side capped at max_dimension, aspect preserved, never upscaled --
    same "don't upscale past the source" rule as rust-core's variants.rs."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    return image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)


def _tile_watermark(image: Image.Image, text: str) -> Image.Image:
    """Diagonal repeating text overlay, semi-transparent -- visible enough to
    deter casual reuse of this derivative without making it unreadable as a
    preview. Drawn on its own RGBA layer and composited, so it works
    regardless of the source image's own mode."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, image.width // 40))
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    step_x, step_y = text_w + 80, text_h + 80

    tile = Image.new("RGBA", (step_x, step_y), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.text((0, 0), text, font=font, fill=(255, 255, 255, 90))
    tile = tile.rotate(30, expand=True)

    for y in range(-tile.height, image.height + tile.height, tile.height):
        for x in range(-tile.width, image.width + tile.width, tile.width):
            overlay.alpha_composite(tile, (x, y))

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def make_original_preview(input_path: str, output_path: str) -> None:
    image = Image.open(input_path)
    image = _resize_capped(image, PREVIEW_MAX_DIMENSION)
    watermarked = _tile_watermark(image, WATERMARK_TEXT)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    watermarked.convert("RGB").save(output_path, "PNG")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make_original_preview(args.input, args.output)
    print(f"wrote {args.output}")
