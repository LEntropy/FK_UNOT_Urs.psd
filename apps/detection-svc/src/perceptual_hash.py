"""Backward-compatible public pHash API kept inside detection-svc."""

from phash_match import compute_perceptual_hash, compute_perceptual_hash_from_path, hamming_distance

__all__ = ["compute_perceptual_hash", "compute_perceptual_hash_from_path", "hamming_distance"]
