"""Picks the style-target image that maximizes real-world cloak
effectiveness, per the LoRA validation experiment's follow-up analysis
(experiments/lora_validation/correlate_drift.py, documented in
ml-engine/README.md's "Final report" section).

That analysis found: the Gram-matrix similarity between an original image
and its chosen cloak-target, measured *before* cloaking, has a strong
negative correlation (Pearson r = -0.929, n=10 real paintings) with the
cloak's real CLIP-measured effect on LoRA training -- images cloaked
toward a style-wise more DISSIMILAR target showed a bigger real
degradation effect. The cloak's own optimization-time metric (how far it
moved the image in VGG19 space) did NOT predict this (r = +0.206) -- only
how different the original and target already were before cloaking even
started.

This directly contradicts today's actual behavior: orchestrate.py/
INTEGRATION.md's default style-target is one fixed image
(ml-engine/out/style_target.png) applied to every upload regardless of
its own style, which per this finding is sometimes close to worst-case
(if a creator's art happens to already resemble that fixed target in
VGG19 style-space, protection is weak) rather than chosen for
effectiveness.

This module doesn't change orchestrate.py's default (that integration
decision belongs to whoever owns that call site) -- it provides the
selection function so a caller CAN opt into "pick the most protective
target from a candidate pool" instead of a fixed default.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import gram_cosine_similarity, mean_sim  # noqa: E402
from model import StyleFeatureExtractor  # noqa: E402
from style_cloak import load_image_tensor  # noqa: E402


def select_most_dissimilar_target(
    original_path: str, candidate_paths: list[str], size: int = 512, device: torch.device | None = None
) -> tuple[str, float]:
    """Returns (path, similarity) for the candidate with the LOWEST
    Gram-matrix cosine similarity to `original_path` -- i.e. the
    candidate expected (per the correlation above) to give the biggest
    real LoRA-degradation effect if used as cloak()'s style_target_path.

    `size` should match the actual training/cloak resolution being used
    elsewhere in the pipeline (512 in the LoRA validation experiment; this
    project's presets were separately validated at 256 -- see
    ml-engine/README.md -- so pick whichever this call site's cloak() call
    itself uses, not a hardcoded default here).
    """
    if not candidate_paths:
        raise ValueError("candidate_paths must be non-empty")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StyleFeatureExtractor(device)

    original_grams = extractor.gram_matrices(load_image_tensor(original_path, size, device))

    best_path, best_sim = None, float("inf")
    for candidate in candidate_paths:
        candidate_grams = extractor.gram_matrices(load_image_tensor(candidate, size, device))
        sim = mean_sim(gram_cosine_similarity(original_grams, candidate_grams))
        if sim < best_sim:
            best_path, best_sim = candidate, sim

    return best_path, best_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--candidates", nargs="+", required=True, help="candidate style-target image paths")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    chosen_path, similarity = select_most_dissimilar_target(args.original, args.candidates, args.size)

    print("=== style-target candidates (lower similarity = more dissimilar = expected bigger real effect) ===")
    print(f"chosen: {chosen_path} (similarity={similarity:.4f})")
