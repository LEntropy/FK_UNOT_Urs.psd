"""Multi-view fingerprint comparison resilient to mirroring and right-angle rotation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

from phash_match import compute_perceptual_hash, hamming_distance


@dataclass(frozen=True)
class FingerprintResult:
    matched: bool
    distance: int
    transform: str
    threshold: int

    def to_dict(self) -> dict:
        return asdict(self)


def compare_robust_fingerprint(
    registered_hash: str, candidate_path: str | Path, threshold: int = 20
) -> FingerprintResult:
    with Image.open(candidate_path) as source:
        image = source.convert("RGB")
        views = {
            "identity": image,
            "mirror": ImageOps.mirror(image),
            "rotate_90": image.rotate(90, expand=True),
            "rotate_180": image.rotate(180, expand=True),
            "rotate_270": image.rotate(270, expand=True),
        }
        distances = {name: hamming_distance(registered_hash, compute_perceptual_hash(view)) for name, view in views.items()}
    transform, distance = min(distances.items(), key=lambda item: item[1])
    return FingerprintResult(distance <= threshold, distance, transform, threshold)
