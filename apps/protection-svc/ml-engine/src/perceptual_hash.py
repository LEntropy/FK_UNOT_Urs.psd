"""Standard perceptual hash (pHash), producing the exact bytes32 format
apps/blockchain-svc/src/hash.ts expects as `perceptualHash`.

This is deliberately NOT the Gram-matrix cosine similarity used in
evaluate.py/robustness_test.py -- that measures "does this look like a
target *style*" for our own cloaking evaluation. This module answers a
different question: "is this the same *image* (or a near-duplicate of it)",
which is what blockchain-svc's on-chain content hash and the future
Monitoring & Detection service (PROJECT_DESIGN.md section 3-7, "pHash
유사도 검색") both need. Conflating the two would be a real bug, not just a
naming nit -- a Gram-matrix vector isn't hashable/comparable the way a
content-identity fingerprint needs to be.

Uses the standard DCT-based pHash algorithm (via the well-tested `imagehash`
library) rather than a from-scratch reimplementation -- this one has no
research judgment calls the way the cloaking algorithm does, so there's
nothing to gain from writing it by hand and real risk of a subtle bug if we
did.

hash_size=16 gives a 16x16 = 256-bit hash = exactly 32 bytes, matching
`bytes32` on the contract side with no padding or truncation.
"""

import argparse

import imagehash
from PIL import Image

HASH_SIZE = 16  # 16*16 = 256 bits = 32 bytes, matches bytes32 exactly


def compute_perceptual_hash(image: Image.Image, hash_size: int = HASH_SIZE) -> str:
    """Returns a 0x-prefixed 64-hex-char (32 byte) perceptual hash string,
    ready to pass as `perceptualHash` to blockchain-svc's POST /assets/register
    (see apps/protection-svc/INTEGRATION.md).
    """
    h = imagehash.phash(image.convert("RGB"), hash_size=hash_size)
    hex_str = str(h)  # imagehash's own hex encoding of the bit matrix
    expected_hex_len = hash_size * hash_size // 4
    if len(hex_str) != expected_hex_len:
        # imagehash pads/truncates internally in ways that depend on
        # hash_size; fail loudly rather than silently emit a hash of the
        # wrong byte length that blockchain-svc's isHexString(v, 32) check
        # would then reject.
        raise ValueError(
            f"expected {expected_hex_len} hex chars for hash_size={hash_size}, got {len(hex_str)}"
        )
    return "0x" + hex_str


def compute_perceptual_hash_from_path(path: str, hash_size: int = HASH_SIZE) -> str:
    return compute_perceptual_hash(Image.open(path), hash_size)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit-distance between two 0x-prefixed hex hashes of equal length.
    0 = identical, higher = more different. Useful for the Monitoring &
    Detection "near-duplicate" search (PROJECT_DESIGN.md section 3-7), and
    for sanity-checking this module itself (see __main__ below).
    """
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    return bin(a ^ b).count("1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pHash for one or more images, and pairwise Hamming distances.")
    parser.add_argument("images", nargs="+", help="image file paths")
    args = parser.parse_args()

    hashes = {path: compute_perceptual_hash_from_path(path) for path in args.images}

    print("=== perceptual hashes ===")
    for path, h in hashes.items():
        print(f"{path:<40} {h}")

    if len(args.images) > 1:
        print()
        print("=== pairwise Hamming distance (0 = identical, 256 = maximally different) ===")
        paths = args.images
        header = " " * 30 + "".join(f"{p.split('/')[-1][:14]:>16}" for p in paths)
        print(header)
        for p1 in paths:
            row = f"{p1.split('/')[-1][:28]:<30}"
            for p2 in paths:
                d = hamming_distance(hashes[p1], hashes[p2])
                row += f"{d:>16}"
            print(row)
