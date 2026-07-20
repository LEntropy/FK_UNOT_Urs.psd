"""letterbox_content_box/letterbox_resize -- the fix for a real reported bug:
every upload was silently stretched into a square (plain .resize((size,size))),
distorting non-square images and permanently capping the published
resolution at `size`. Pure PIL/math, no torch/VGG needed, so this runs fast
without a GPU.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "ml-engine" / "src"))
from style_cloak import letterbox_content_box, letterbox_resize  # noqa: E402


def test_content_box_is_full_canvas_for_a_square_image():
    box = letterbox_content_box(300, 300, 256)
    assert box == (0, 0, 256, 256)


def test_content_box_centers_a_landscape_image_with_vertical_padding():
    # 960x645 (~1.4884 aspect) fit into 256x256: width fills exactly 256,
    # height is padded top/bottom.
    box = letterbox_content_box(960, 645, 256)
    left, top, right, bottom = box
    assert left == 0
    assert right == 256
    assert top > 0
    assert bottom < 256
    assert (right - left) / (bottom - top) == pytest.approx(960 / 645, rel=0.02)


def test_content_box_centers_a_portrait_image_with_horizontal_padding():
    box = letterbox_content_box(645, 960, 256)
    left, top, right, bottom = box
    assert top == 0
    assert bottom == 256
    assert left > 0
    assert right < 256


def test_letterbox_resize_produces_exactly_size_by_size_canvas():
    img = Image.new("RGB", (960, 645), (200, 50, 50))
    out = letterbox_resize(img, 256)
    assert out.size == (256, 256)


def test_letterbox_resize_preserves_content_without_stretching():
    # A distinctly-colored real image content region should land inside the
    # content box computed independently, not get squashed to fill the
    # whole canvas.
    img = Image.new("RGB", (960, 645), (10, 200, 10))
    out = letterbox_resize(img, 256)
    left, top, right, bottom = letterbox_content_box(960, 645, 256)

    # Center of content box: real image color.
    cx, cy = (left + right) // 2, (top + bottom) // 2
    assert out.getpixel((cx, cy))[1] > 150  # green channel dominates

    # A pixel in the padding (if any exists) should be the neutral gray
    # pad color, not the image's color -- proves it wasn't stretched to fill.
    if top > 2:
        pad_pixel = out.getpixel((cx, 1))
        assert pad_pixel == (114, 114, 114)
