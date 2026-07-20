"""compute_perceptual_mask -- redistributes the epsilon clamp toward
already-textured regions and away from flat ones (a real "the noise looks
too visible" fix, not just a smaller epsilon). Pure tensor math, no VGG/GPU
needed, so this runs fast.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ml-engine" / "src"))
from style_cloak import compute_perceptual_mask  # noqa: E402


def test_mask_shape_matches_input():
    original = torch.rand(1, 3, 64, 64)
    mask = compute_perceptual_mask(original)
    assert mask.shape == original.shape


def test_mask_stays_within_low_high_bounds():
    original = torch.rand(1, 3, 64, 64)
    mask = compute_perceptual_mask(original, low=0.3, high=1.7)
    assert mask.min() >= 0.3 - 1e-4
    assert mask.max() <= 1.7 + 1e-4


def test_flat_image_gets_a_uniform_low_mask():
    # No edges anywhere -- every pixel should land near the low end, not
    # some being flagged "textured" by noise/rounding.
    flat = torch.full((1, 3, 32, 32), 0.5)
    mask = compute_perceptual_mask(flat, low=0.3, high=1.7)
    assert mask.max() - mask.min() < 0.2


def test_a_sharp_edge_gets_a_higher_mask_than_the_flat_region_around_it():
    img = torch.zeros(1, 3, 64, 64)
    img[:, :, :, 32:] = 1.0  # a hard vertical edge down the middle
    mask = compute_perceptual_mask(img, low=0.3, high=1.7)

    near_edge = mask[0, 0, 32, 30].item()
    far_from_edge = mask[0, 0, 32, 2].item()
    assert near_edge > far_from_edge
