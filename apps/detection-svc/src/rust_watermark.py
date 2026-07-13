"""Wraps rust-core's already-built `detect` CLI subcommand (see
apps/protection-svc/rust-core/src/main.rs) instead of reimplementing
watermark detection -- same pattern as orchestrate.py's run_rust_core().

rust-core detect's output (`[detect] recovered=<hex> avg_confidence=<f>
min_confidence=<f>` + optional `[detect] bit error rate vs expected: <f>%`)
is parsed here. Confidence/BER numbers reused directly from
rust-core/README.md's measured robustness table are NOT reinvented -- this
module just recovers them per-image; the caller decides what confidence
counts as a match.
"""

import os
import re
import subprocess
from pathlib import Path

_candidates = [
    Path(__file__).resolve().parents[2] / "protection-svc" / "rust-core" / "target" / "release" / "rust-core",
    Path(__file__).resolve().parents[2] / "protection-svc" / "rust-core" / "target" / "release" / "rust-core.exe",
    Path(__file__).resolve().parents[2] / "protection-svc" / "rust-core" / "target" / "debug" / "rust-core.exe",
    Path(__file__).resolve().parents[2] / "protection-svc" / "rust-core" / "target" / "debug" / "rust-core",
]


def _rust_core_bin() -> Path:
    if "RUST_CORE_BIN" in os.environ:
        return Path(os.environ["RUST_CORE_BIN"])
    return next((p for p in _candidates if p.exists()), _candidates[0])


class WatermarkDetectResult:
    def __init__(self, recovered_hex: str, avg_confidence: float, min_confidence: float, bit_error_rate: float | None):
        self.recovered_hex = recovered_hex
        self.avg_confidence = avg_confidence
        self.min_confidence = min_confidence
        self.bit_error_rate = bit_error_rate  # None if no expected_hex was supplied

    @property
    def is_match(self) -> bool:
        # bit_error_rate is only meaningful when an expected payload was
        # given (the caller's use case here). 0% BER = exact bit match.
        return self.bit_error_rate is not None and self.bit_error_rate < 5.0


def detect_watermark(image_path: str, expected_hex: str, bits: int = 64) -> WatermarkDetectResult:
    rust_core_bin = _rust_core_bin()
    if not rust_core_bin.exists():
        raise FileNotFoundError(
            f"rust-core binary not found at {rust_core_bin} -- run `cargo build --release` in protection-svc/rust-core first"
        )

    result = subprocess.run(
        [str(rust_core_bin), "detect", "--input", image_path, "--bits", str(bits), "--expected-hex", expected_hex],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rust-core detect failed:\n{result.stderr}")

    stdout = result.stdout
    m = re.search(r"recovered=([0-9a-f]+) avg_confidence=([\d.]+) min_confidence=([\d.]+)", stdout)
    if not m:
        raise RuntimeError(f"could not parse rust-core detect output:\n{stdout}")
    recovered_hex, avg_conf, min_conf = m.group(1), float(m.group(2)), float(m.group(3))

    ber_match = re.search(r"bit error rate vs expected: ([\d.]+)%", stdout)
    bit_error_rate = float(ber_match.group(1)) if ber_match else None

    return WatermarkDetectResult(recovered_hex, avg_conf, min_conf, bit_error_rate)
