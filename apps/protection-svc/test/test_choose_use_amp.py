"""choose_use_amp -- real GPU measurement at size=1024 found mixed
precision (fp16 forward/backward through VGG, fp32 forced for the Gram
matrix reduction to avoid overflow) matched fp32's styleDriftScore/PSNR
almost exactly while running 2.2x faster and using 29% less peak VRAM.
See orchestrate.py's module-level comment above the function for the
numbers, and choose_eot_samples's doc for why sizes above 1536 still hit
the same VRAM-pressure wall (AMP didn't unlock a higher processing-size
cap, just made the existing 1024 cap cheaper).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestrate import choose_use_amp  # noqa: E402


def test_disabled_at_or_below_the_originally_validated_256px_envelope():
    assert choose_use_amp(256) is False
    assert choose_use_amp(128) is False


def test_enabled_above_256px():
    assert choose_use_amp(1024) is True
    assert choose_use_amp(512) is True
