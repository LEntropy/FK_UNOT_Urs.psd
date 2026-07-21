"""choose_processing_size -- replaces always forcing size=256 regardless
of the real upload, which a real GPU measurement found genuinely worse on
both quality (PSNR) and protection strength (styleDriftScore) than
processing closer to the real resolution. See orchestrate.py's module-
level comment above this function for the numbers.
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestrate import choose_processing_size  # noqa: E402


def test_uses_the_images_own_long_edge_when_below_the_cap(tmp_path):
    p = tmp_path / "small.png"
    Image.new("RGB", (400, 300), (0, 0, 0)).save(p)
    assert choose_processing_size(str(p), max_size=1024) == 400


def test_uses_the_taller_edge_for_a_portrait_image(tmp_path):
    p = tmp_path / "portrait.png"
    Image.new("RGB", (300, 500), (0, 0, 0)).save(p)
    assert choose_processing_size(str(p), max_size=1024) == 500


def test_caps_at_max_size_for_a_real_high_resolution_upload(tmp_path):
    p = tmp_path / "huge.png"
    Image.new("RGB", (2835, 4289), (0, 0, 0)).save(p)
    assert choose_processing_size(str(p), max_size=1024) == 1024


def test_a_tiny_upload_is_not_upsized_beyond_its_own_resolution(tmp_path):
    p = tmp_path / "tiny.png"
    Image.new("RGB", (64, 48), (0, 0, 0)).save(p)
    assert choose_processing_size(str(p), max_size=1024) == 64
