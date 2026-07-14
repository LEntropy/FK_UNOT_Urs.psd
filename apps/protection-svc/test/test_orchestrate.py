"""_maybe_auto_select_style_target's branch logic (real selection itself is
select_style_target.py's own concern -- mocked here to keep this fast and
GPU-free).
"""

import sys
from pathlib import Path

import pytest

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
