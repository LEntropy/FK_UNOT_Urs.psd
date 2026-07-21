"""choose_perceptual_mask -- real GPU measurement on top of the now-fixed
native-resolution pipeline found real PSNR wins for negligible styleDriftScore
cost on both L2_PORTFOLIO and L3_ANTI_TRAIN (see orchestrate.py's module-level
comment above the function for the numbers). L1_PREVIEW hasn't been measured
and stays off -- it's already the cheap/no-EOT tier and was never the preset
the noise-visibility complaint was about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestrate import choose_perceptual_mask  # noqa: E402


def test_enabled_for_l3_anti_train():
    assert choose_perceptual_mask("L3_ANTI_TRAIN") is True


def test_enabled_for_l2_portfolio():
    assert choose_perceptual_mask("L2_PORTFOLIO") is True


def test_disabled_for_l1_preview_not_measured():
    assert choose_perceptual_mask("L1_PREVIEW") is False
