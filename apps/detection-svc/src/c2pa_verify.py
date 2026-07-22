"""Wraps rust-core's already-built `c2pa-verify` CLI subcommand (see
apps/protection-svc/rust-core/src/main.rs) -- same pattern as
rust_watermark.py's detect_watermark(), reused as-is rather than
reimplemented here.

Lets detection-svc's evidence pipeline check whether a candidate image
carries a C2PA manifest DONTAI's own protect() pipeline embedded (see
orchestrate.py's step 2b/4) -- a second, independent provenance signal
alongside the existing perceptual-hash and watermark checks, not a
replacement for either (a redistribution can strip a C2PA manifest far
more easily than it can defeat the invisible watermark, so this is
"bonus evidence when present," never the sole basis for a match).
"""

import json
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


class C2paVerifyResult:
    def __init__(self, manifest: dict | None, validation_issues: list[str] | None):
        self.manifest = manifest  # None means no C2PA manifest was found -- the ordinary case, not an error
        self.validation_issues = validation_issues

    @property
    def has_manifest(self) -> bool:
        return self.manifest is not None

    @property
    def ownership(self) -> dict | None:
        """The `com.dontai.ownership` custom assertion's data (doNotTrain/
        title/creatorId/perceptualHash -- see orchestrate.py's protect()),
        if this file has one. None if there's no manifest, or a manifest
        exists but wasn't produced by this project's own signing code
        (no com.dontai.ownership assertion -- some other, unrelated C2PA
        tool embedded it)."""
        if not self.manifest:
            return None
        active = self.manifest.get("active_manifest")
        active_manifest = self.manifest.get("manifests", {}).get(active, {})
        for assertion in active_manifest.get("assertions", []):
            if assertion.get("label") == "com.dontai.ownership":
                return assertion.get("data")
        return None

    @property
    def signed_by_dontai(self) -> bool:
        """Whether the manifest's signing certificate identifies it as this
        project's own protection-svc identity (issuer=DONTAI) -- a
        specific signal that this exact file passed through DONTAI's own
        protect() pipeline at some point, not just "some C2PA manifest
        exists" (which any tool could have embedded, including a bad actor
        trying to launder provenance)."""
        if not self.manifest:
            return False
        active = self.manifest.get("active_manifest")
        active_manifest = self.manifest.get("manifests", {}).get(active, {})
        return active_manifest.get("signature_info", {}).get("issuer") == "DONTAI"


def verify_c2pa(image_path: str, image_format: str = "png") -> C2paVerifyResult:
    rust_core_bin = _rust_core_bin()
    if not rust_core_bin.exists():
        raise FileNotFoundError(
            f"rust-core binary not found at {rust_core_bin} -- run `cargo build --release` in protection-svc/rust-core first"
        )

    result = subprocess.run(
        [str(rust_core_bin), "c2pa-verify", "--input", image_path, "--format", image_format],
        capture_output=True,
        text=True,
        # Explicit UTF-8, not text=True's locale-dependent default -- the
        # manifest JSON rust-core prints is always UTF-8 (real titles can
        # be non-ASCII, e.g. Korean), but Python's subprocess text-mode
        # decodes with locale.getpreferredencoding() by default, which is
        # cp949 (not UTF-8) on this project's Korean-locale Windows dev
        # environment -- hit for real, a UnicodeDecodeError crashing this
        # function on legitimate UTF-8 output.
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"rust-core c2pa-verify failed:\n{result.stderr}")

    stdout = result.stdout
    if "[c2pa-verify] no manifest found" in stdout:
        return C2paVerifyResult(manifest=None, validation_issues=None)

    # Anchored on our own literal marker strings (not JSON brace-matching,
    # which would be fragile against the pretty-printed manifest's own
    # internal `}` lines) -- "[c2pa-verify] validation" only ever appears
    # once, right after the manifest JSON block, in main.rs's own print
    # calls.
    m = re.search(r"\[c2pa-verify\] manifest:\n(.*)\n\[c2pa-verify\] validation", stdout, re.DOTALL)
    if not m:
        raise RuntimeError(f"could not parse rust-core c2pa-verify output:\n{stdout}")
    manifest = json.loads(m.group(1))

    validation_issues = re.findall(r"^\s+-\s+(.+)$", stdout, re.MULTILINE) or None

    return C2paVerifyResult(manifest=manifest, validation_issues=validation_issues)
