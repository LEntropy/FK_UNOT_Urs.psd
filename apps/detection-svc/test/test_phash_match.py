"""Hamming-distance threshold logic, reusing ml-engine's perceptual_hash.py
directly (see src/phash_match.py). Uses structured (non-uniform) test
images -- solid-color images are a bad choice here because pHash's DCT
mostly captures AC coefficients, which can coincide for two different
uniform colors.
"""

from PIL import Image

from phash_match import is_likely_match


def _gradient_image(path, seed_offset=0):
    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for x in range(64):
        for y in range(64):
            pixels[x, y] = ((x * 4 + seed_offset) % 256, (y * 4) % 256, ((x + y) * 2) % 256)
    img.save(path)


def _checkerboard_image(path):
    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for x in range(64):
        for y in range(64):
            on = (x // 8 + y // 8) % 2 == 0
            pixels[x, y] = (255, 255, 255) if on else (0, 0, 0)
    img.save(path)


def test_exact_duplicate_is_a_match(tmp_path):
    from perceptual_hash import compute_perceptual_hash_from_path

    original = tmp_path / "original.png"
    duplicate = tmp_path / "duplicate.png"
    _gradient_image(original)
    _gradient_image(duplicate)  # identical pixels

    registered_hash = compute_perceptual_hash_from_path(str(original))
    is_match, distance = is_likely_match(registered_hash, str(duplicate), threshold=20)

    assert is_match is True
    assert distance == 0


def test_unrelated_image_is_not_a_match(tmp_path):
    from perceptual_hash import compute_perceptual_hash_from_path

    original = tmp_path / "original.png"
    unrelated = tmp_path / "unrelated.png"
    _gradient_image(original)
    _checkerboard_image(unrelated)

    registered_hash = compute_perceptual_hash_from_path(str(original))
    is_match, distance = is_likely_match(registered_hash, str(unrelated), threshold=20)

    assert is_match is False
    assert distance > 20
