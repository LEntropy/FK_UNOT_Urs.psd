"""Best-effort C2PA inspection using the independent `c2pa-python` bindings,
with explicit unavailable/not-found/invalid/valid states.

Distinct from, and complementary to, src/c2pa_verify.py: that module shells
out to rust-core's own `c2pa-verify` CLI to answer one specific question --
"does this file carry a manifest DONTAI's own protect() pipeline signed?"
(hasManifest/signedByDontai/ownership) -- and requires the rust-core binary
to be built. This module answers a more general question -- "is there a
structurally valid C2PA manifest on this file at all, from any tool?" --
via the c2pa-python library directly, with no dependency on rust-core being
built or reachable. Useful as a second, independent check when rust-core
isn't available (development machines without Cargo, some CI environments)
and as a generic "did *anyone's* C2PA signing survive re-encoding" signal
for candidates that were never through DONTAI's own pipeline at all. Kept
side-by-side rather than merged into c2pa_verify.py -- callers that want
"is this specifically DONTAI-signed" should keep using verify_c2pa from
c2pa_verify.py; callers that want a general validity signal (e.g.
verdict.py's c2pa_status input) can use this one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class C2paResult:
    status: str
    manifest_present: bool
    valid: bool | None
    active_manifest: str | None = None
    validation_issues: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["validation_issues"] = list(self.validation_issues)
        return data


def verify_c2pa(path: str | Path) -> C2paResult:
    try:
        from c2pa import Reader
    except ImportError:
        return C2paResult("UNAVAILABLE", False, None, error="c2pa-python is not installed")

    try:
        with Reader(str(path)) as reader:
            report = json.loads(reader.json())
    except Exception as exc:  # c2pa bindings expose version-specific exception types
        message = str(exc)
        not_found = any(token in message.lower() for token in ("manifest", "jumbf", "not found"))
        return C2paResult("NOT_FOUND" if not_found else "INVALID", False, False, error=message)

    active = report.get("active_manifest")
    manifests = report.get("manifests") or {}
    active_data = manifests.get(active, {}) if active else {}
    statuses = active_data.get("validation_status") or report.get("validation_status") or []
    issues = tuple(
        str(item.get("code") or item.get("explanation") or item) if isinstance(item, dict) else str(item)
        for item in statuses
    )
    present = bool(active or manifests)
    return C2paResult("VALID" if present and not issues else "INVALID", present, present and not issues, active, issues)
