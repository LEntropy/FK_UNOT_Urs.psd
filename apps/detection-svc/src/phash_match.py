"""Perceptual-hash comparison, reusing ml-engine's already-validated
implementation directly rather than re-vendoring the algorithm (see that
module's own docstring for why hand-rolling pHash would be a real risk).

Also re-exports compute_perceptual_hash (the Image-object variant, not just
the from-path one) so the newer local-only modules that build on top of it
-- phash_index.py's BK-tree and robust_fingerprint.py's mirror/rotation
comparison -- have one shared pHash implementation to import from, without
either duplicating ml-engine's algorithm or requiring an image round-trip
to disk just to hash an in-memory PIL.Image.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "protection-svc" / "ml-engine" / "src"))

from perceptual_hash import compute_perceptual_hash, compute_perceptual_hash_from_path, hamming_distance  # noqa: E402


def is_likely_match(registered_hash: str, candidate_image_path: str, threshold: int) -> tuple[bool, int]:
    candidate_hash = compute_perceptual_hash_from_path(candidate_image_path)
    distance = hamming_distance(registered_hash, candidate_hash)
    return distance <= threshold, distance
