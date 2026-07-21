"""choose_perceptual_mask -- real GPU measurement on top of the now-fixed
native-resolution pipeline found +1.37dB PSNR for only -1.9% styleDriftScore
on a real high-res L3 upload (see orchestrate.py's module-level comment
above the function for the numbers). Only measured against L3_ANTI_TRAIN,
so it stays scoped to that preset rather than assumed to generalize to L1/L2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestrate import choose_perceptual_mask  # noqa: E402


def test_enabled_for_l3_anti_train():
    assert choose_perceptual_mask("L3_ANTI_TRAIN") is True


def test_disabled_for_l1_preview():
    assert choose_perceptual_mask("L1_PREVIEW") is False


def test_disabled_for_l2_portfolio_not_yet_measured():
    assert choose_perceptual_mask("L2_PORTFOLIO") is False
