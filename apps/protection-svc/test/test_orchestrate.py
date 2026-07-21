"""_maybe_auto_select_style_target's branch logic (real selection itself is
select_style_target.py's own concern -- mocked here to keep this fast and
GPU-free).
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import orchestrate  # noqa: E402


def test_returns_given_target_unchanged_when_env_var_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("STYLE_TARGET_CANDIDATES_DIR", raising=False)
    result = orchestrate._maybe_auto_select_style_target("original.png", "given_target.png", 256)
    assert result == "given_target.png"


def test_skips_when_using_remote_gpu(monkeypatch, tmp_path):
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "a.png").write_bytes(b"fake")

    monkeypatch.setenv("STYLE_TARGET_CANDIDATES_DIR", str(candidates_dir))
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", True)

    result = orchestrate._maybe_auto_select_style_target("original.png", "given_target.png", 256)
    assert result == "given_target.png"


def test_returns_given_target_unchanged_when_candidates_dir_is_empty(monkeypatch, tmp_path):
    candidates_dir = tmp_path / "empty_candidates"
    candidates_dir.mkdir()

    monkeypatch.setenv("STYLE_TARGET_CANDIDATES_DIR", str(candidates_dir))
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", False)

    result = orchestrate._maybe_auto_select_style_target("original.png", "given_target.png", 256)
    assert result == "given_target.png"


def test_calls_select_most_dissimilar_target_when_candidates_exist(monkeypatch, tmp_path):
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "a.png").write_bytes(b"fake")
    (candidates_dir / "b.jpg").write_bytes(b"fake")
    (candidates_dir / "not_an_image.txt").write_bytes(b"fake")  # must be filtered out

    monkeypatch.setenv("STYLE_TARGET_CANDIDATES_DIR", str(candidates_dir))
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", False)

    captured = {}

    def fake_select(original_path, candidate_paths, size):
        captured["original_path"] = original_path
        captured["candidate_paths"] = sorted(Path(p).name for p in candidate_paths)
        captured["size"] = size
        return (str(candidates_dir / "b.jpg"), 0.42)

    monkeypatch.setitem(sys.modules, "select_style_target", type(sys)("select_style_target"))
    sys.modules["select_style_target"].select_most_dissimilar_target = fake_select

    result = orchestrate._maybe_auto_select_style_target("original.png", "given_target.png", 512)

    assert result == str(candidates_dir / "b.jpg")
    assert captured["original_path"] == "original.png"
    assert captured["candidate_paths"] == ["a.png", "b.jpg"]
    assert captured["size"] == 512


def test_resolution_restoration_produces_a_fully_loadable_non_truncated_file(monkeypatch, tmp_path):
    """Regression test for a real production failure: the crop-back-out-of-
    letterbox-padding step used to do `Image.open(cloaked_path).crop(box)
    .save(cloaked_path)` -- opening and saving the *same* path. PIL's
    Image.open() is lazy, so .save() truncated the file for writing before
    .crop() ever read its pixel data, corrupting it (valid PNG header, empty
    body). A real high-res upload hit this: rust-core's watermark step
    failed with `IoError(UnexpectedEof)` reading the corrupted cloaked.png.
    Fixed by forcing an eager load (`.convert("RGB")`) before the in-place
    save. This test drives protect() with everything except that
    resolution-restoration block mocked out, and asserts the resulting file
    survives a real `Image.load()` and has the cropped-back-out aspect
    ratio, not the padded square.
    """
    input_path = tmp_path / "original.png"
    Image.new("RGB", (400, 200), (10, 20, 30)).save(input_path)  # 2:1 aspect

    style_target_path = tmp_path / "style_target.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(style_target_path)

    out_dir = tmp_path / "out"

    def fake_cloak(original_path, style_target_path, output_path, preset_name, eot, size, eot_samples):
        # Mirrors letterbox_resize's real output shape: a padded square at
        # the processing size, not yet cropped back to the real aspect ratio.
        Image.new("RGB", (size, size), (50, 60, 70)).save(output_path)

    monkeypatch.setattr(orchestrate, "cloak", fake_cloak)
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", False)
    monkeypatch.setattr(orchestrate, "run_rust_core", lambda *a, **k: "")
    monkeypatch.setattr(orchestrate, "parse_variants_output", lambda output: [])
    monkeypatch.setattr(orchestrate, "compute_perceptual_hash_from_path", lambda path: "deadbeef")

    result = orchestrate.protect(
        input_path=str(input_path),
        out_dir=str(out_dir),
        preset_name="L1_PREVIEW",
        style_target_path=str(style_target_path),
        title="t",
        creator_id="c",
        allow_ai_training=False,
        watermark_payload_hex="deadbeefcafef00d",
        size=256,
    )

    assert result["status"] == "completed"

    cloaked_path = out_dir / "cloaked.png"
    restored = Image.open(cloaked_path)
    restored.load()  # raises OSError("image file is truncated") if corrupted
    assert restored.size[0] > restored.size[1]  # 2:1 aspect restored, not square
