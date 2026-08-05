"""_maybe_auto_select_style_target's branch logic (real selection itself is
select_style_target.py's own concern -- mocked here to keep this fast and
GPU-free).
"""

import json
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

    def fake_cloak(original_path, style_target_path, output_path, preset_name, eot, size, eot_samples, perceptual_mask, use_amp):
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


def _protect_with_rust_core_stub(monkeypatch, tmp_path, rust_core_calls, run_rust_core_impl):
    """Shared setup for the two C2PA-step tests below -- everything except
    run_rust_core and compute_perceptual_hash_from_path is stubbed the same
    way test_resolution_restoration_produces_a_fully_loadable_non_truncated_file
    stubs it, just also recording each run_rust_core call's args."""
    input_path = tmp_path / "original.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(input_path)
    style_target_path = tmp_path / "style_target.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(style_target_path)

    def fake_cloak(original_path, style_target_path, output_path, preset_name, eot, size, eot_samples, perceptual_mask, use_amp):
        Image.new("RGB", (size, size), (50, 60, 70)).save(output_path)

    def recording_run_rust_core(*args):
        rust_core_calls.append(args)
        return run_rust_core_impl(*args)

    monkeypatch.setattr(orchestrate, "cloak", fake_cloak)
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", False)
    monkeypatch.setattr(orchestrate, "run_rust_core", recording_run_rust_core)
    monkeypatch.setattr(orchestrate, "parse_variants_output", lambda output: [])
    monkeypatch.setattr(orchestrate, "compute_perceptual_hash_from_path", lambda path: "deadbeef")

    return str(input_path), str(style_target_path)


def test_c2pa_sign_is_called_with_real_pipeline_data(monkeypatch, tmp_path):
    """Confirms C2PA embedding actually runs as part of protect() (it
    previously didn't -- see rust-core/README.md's C2PA section), and that
    the ownership assertion carries this call's real doNotTrain/title/
    creatorId/perceptualHash, not placeholders."""
    rust_core_calls = []
    input_path, style_target_path = _protect_with_rust_core_stub(
        monkeypatch, tmp_path, rust_core_calls, lambda *a: ""
    )

    result = orchestrate.protect(
        input_path=input_path,
        out_dir=str(tmp_path / "out"),
        preset_name="L1_PREVIEW",
        style_target_path=style_target_path,
        title="My Artwork",
        creator_id="creator_42",
        allow_ai_training=False,
        watermark_payload_hex="deadbeefcafef00d",
        size=256,
    )

    assert result["c2paApplied"] is True

    c2pa_calls = [c for c in rust_core_calls if c[0] == "c2pa-sign"]
    assert len(c2pa_calls) == 1
    args = c2pa_calls[0]
    assert "--title" in args and args[args.index("--title") + 1] == "My Artwork"

    ownership_json = args[args.index("--ownership-json") + 1]
    ownership = json.loads(ownership_json)
    assert ownership == {
        "doNotTrain": True,  # allow_ai_training=False above
        "title": "My Artwork",
        "creatorId": "creator_42",
        "perceptualHash": "deadbeef",  # from the stubbed compute_perceptual_hash_from_path
    }

    signing_key_path = args[args.index("--signing-key-path") + 1]
    assert signing_key_path == str(orchestrate.C2PA_SIGNING_KEY_PATH)


def test_c2pa_sign_failure_does_not_fail_the_whole_upload(monkeypatch, tmp_path):
    """Best-effort like protection_metrics/concept-misalign elsewhere in
    protect() -- a C2PA signing failure (rust-core binary missing, disk
    full persisting the signing key, whatever) shouldn't take down a real
    upload that otherwise succeeded."""
    rust_core_calls = []

    def flaky_run_rust_core(*args):
        if args[0] == "c2pa-sign":
            raise RuntimeError("rust-core c2pa-sign failed: simulated failure")
        return ""

    input_path, style_target_path = _protect_with_rust_core_stub(
        monkeypatch, tmp_path, rust_core_calls, flaky_run_rust_core
    )

    result = orchestrate.protect(
        input_path=input_path,
        out_dir=str(tmp_path / "out"),
        preset_name="L1_PREVIEW",
        style_target_path=style_target_path,
        title="t",
        creator_id="c",
        allow_ai_training=True,
        watermark_payload_hex="deadbeefcafef00d",
        size=256,
    )

    assert result["status"] == "completed"
    assert result["c2paApplied"] is False
    # The rest of the pipeline (variants) still ran despite the C2PA failure.
    assert any(c[0] == "variants" for c in rust_core_calls)


def test_strong_protection_falls_back_to_style_cloak_on_failure(monkeypatch, tmp_path):
    """PHASE4_SCOPING.md §6's core safety property: if the dedicated
    A40-class pod is unreachable/unconfigured (MULTIARCH_GPU_HOST unset,
    SSH failure, whatever), a strong_protection=True upload must still end
    up protected by the proven style_cloak mechanism, never published
    unprotected."""
    input_path = tmp_path / "original.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(input_path)
    style_target_path = tmp_path / "style_target.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(style_target_path)

    style_cloak_calls = []

    def fake_cloak(original_path, style_target_path, output_path, preset_name, eot, size, eot_samples, perceptual_mask, use_amp):
        style_cloak_calls.append(output_path)
        Image.new("RGB", (size, size), (50, 60, 70)).save(output_path)

    def failing_remote_multiarch_cloak(original_path, output_path, prompt, preset_name="MULTIARCH_FULL"):
        raise RuntimeError("MULTIARCH_GPU_HOST not set")

    monkeypatch.setattr(orchestrate, "cloak", fake_cloak)
    monkeypatch.setattr(orchestrate, "remote_multiarch_cloak", failing_remote_multiarch_cloak)
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", False)
    monkeypatch.setattr(orchestrate, "run_rust_core", lambda *a, **k: "")
    monkeypatch.setattr(orchestrate, "parse_variants_output", lambda output: [])
    monkeypatch.setattr(orchestrate, "compute_perceptual_hash_from_path", lambda path: "deadbeef")

    result = orchestrate.protect(
        input_path=str(input_path),
        out_dir=str(tmp_path / "out"),
        preset_name="L1_PREVIEW",
        style_target_path=str(style_target_path),
        title="t",
        creator_id="c",
        allow_ai_training=False,
        watermark_payload_hex="deadbeefcafef00d",
        size=256,
        strong_protection=True,
    )

    assert result["status"] == "completed"
    assert len(style_cloak_calls) == 1  # fell back to style_cloak exactly once


def test_strong_protection_success_skips_style_cloak(monkeypatch, tmp_path):
    """When the multiarch pod succeeds, style_cloak (and its style-drift
    metric, which doesn't apply to a mechanism with no style_target_path
    input) should not run at all -- strong_protection replaces style-cloak,
    it doesn't stack with it (this project's own hybrid_attack experiment
    found combining attack objectives makes things worse, not better)."""
    input_path = tmp_path / "original.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(input_path)
    style_target_path = tmp_path / "style_target.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(style_target_path)

    style_cloak_calls = []
    multiarch_calls = []

    def fake_cloak(*args, **kwargs):
        style_cloak_calls.append(True)

    def fake_remote_multiarch_cloak(original_path, output_path, prompt, preset_name="MULTIARCH_FULL"):
        multiarch_calls.append((original_path, prompt))
        Image.new("RGB", (64, 64), (90, 90, 90)).save(output_path)

    monkeypatch.setattr(orchestrate, "cloak", fake_cloak)
    monkeypatch.setattr(orchestrate, "remote_multiarch_cloak", fake_remote_multiarch_cloak)
    monkeypatch.setattr(orchestrate, "USE_REMOTE_GPU", False)
    monkeypatch.setattr(orchestrate, "run_rust_core", lambda *a, **k: "")
    monkeypatch.setattr(orchestrate, "parse_variants_output", lambda output: [])
    monkeypatch.setattr(orchestrate, "compute_perceptual_hash_from_path", lambda path: "deadbeef")

    result = orchestrate.protect(
        input_path=str(input_path),
        out_dir=str(tmp_path / "out"),
        preset_name="L1_PREVIEW",
        style_target_path=str(style_target_path),
        title="My Artwork",
        creator_id="c",
        allow_ai_training=False,
        watermark_payload_hex="deadbeefcafef00d",
        size=256,
        strong_protection=True,
    )

    assert result["status"] == "completed"
    assert len(multiarch_calls) == 1
    assert multiarch_calls[0][0] == str(input_path)
    assert multiarch_calls[0][1] == "My Artwork"  # title used as the prompt proxy
    assert style_cloak_calls == []  # style_cloak never ran
