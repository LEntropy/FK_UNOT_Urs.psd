import pytest

from rust_watermark import detect_watermark


def test_missing_binary_raises_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("RUST_CORE_BIN", str(tmp_path / "does-not-exist"))

    with pytest.raises(FileNotFoundError):
        detect_watermark(str(tmp_path / "irrelevant.png"), "deadbeefcafef00d")
