"""Assembles the evidence bundle exactly per PROJECT_DESIGN.md §3-7's field
list: 원본해시, 보호본해시, 등록시각, 권리자, 워터마크검출, 발견URL, 발견시각,
스크린샷, HTTP헤더, 온체인 트랜잭션, 내부 서명 -> JSON (always) + best-effort PDF.

`signature` starts null here and is filled in by the caller (server.py) via
evidence_signing.py's sign_bundle(), which calls api-gateway's real Ed25519
signing endpoint (KMS envelope-encrypted key -- the KMS C server itself has
no Sign() RPC, see api-gateway/src/evidenceSigning.ts's doc). Kept as a
separate step here rather than signed inline because build_bundle stays a
pure, synchronous, easily-testable function; the signing step is the one
part of this module that does real network I/O and can fail independently.

Also carries write_evidence_manifest(), which hashes every file written
into a case's evidence directory (the candidate image, headers.json,
screenshot, bundle.json/pdf, etc.) into one manifest.json -- the input
evidence_integrity.EvidenceSigner.sign_manifest() signs and
verify_evidence_directory() later re-checks, independent of and
complementary to evidence_signing.py's bundle-level Ed25519 signature.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def build_bundle(
    *,
    artwork: dict,
    source_url: str | None,
    detected_at: float,
    phash_distance: int | None = None,
    watermark_result: dict | None = None,
    c2pa_result: dict | None = None,
    headers: dict | None = None,
    screenshot_path: str | None = None,
    model_leak_result: dict | None = None,
    fingerprint_result: dict | None = None,
    verdict_result: dict | None = None,
) -> dict[str, Any]:
    ownership_records = artwork.get("ownershipRecords") or []
    onchain = ownership_records[0] if ownership_records else None

    return {
        "originalHash": artwork.get("perceptualHash"),
        "protectedHash": artwork.get("perceptualHash"),  # registered hash IS the watermarked/published image's hash
        "registeredAt": onchain.get("registeredAt") if onchain else None,
        "rightsHolder": artwork.get("ownerWalletAddress"),
        "watermarkDetection": watermark_result,
        # Supplementary provenance signal, not part of PROJECT_DESIGN.md
        # §3-7's original field list (predates C2PA being wired into the
        # protect() pipeline) -- included when present, never the deciding
        # factor for whether a case even reaches this bundle-building step
        # (see server.py's _run_case_for_urls: computed but not consulted
        # for is_match). None if C2PA verification itself failed to run
        # (rust-core unreachable etc.), distinct from a real
        # {"hasManifest": False, ...} result meaning verification *did* run
        # and genuinely found no manifest.
        "c2paDetection": c2pa_result,
        # Set only for a model-leak report (server.py's _run_model_leak_case)
        # -- protection_client.poll_leak_detection_job's completed job
        # result (perPrompt/meanDelta/verdict, see ml-engine/src/
        # model_leak_detect.py's module doc), None for every other
        # evidence path (a scraped webpage has no suspect model to score).
        "modelLeakDetection": model_leak_result,
        # Multi-view (mirror/rotate) pHash comparison (src/robust_fingerprint.py)
        # and the explainable MATCH/REVIEW/NO_MATCH/INCONCLUSIVE aggregation
        # (src/verdict.py) -- both informational alongside the fields above,
        # not (yet) the trigger for whether this bundle gets built at all;
        # see server.py's _run_case_for_urls doc for why that trigger still
        # runs on the original phash/watermark check.
        "robustFingerprint": fingerprint_result,
        "verdict": verdict_result,
        "discoveredUrl": source_url,
        "discoveredAt": detected_at,
        "phashDistance": phash_distance,
        "screenshotPath": screenshot_path,
        "httpHeaders": headers,
        "onchainTransaction": {
            "chain": onchain.get("chain"),
            "registryAddress": onchain.get("registryAddress"),
            "txHash": onchain.get("txHash"),
            "blockNumber": onchain.get("blockNumber"),
        }
        if onchain
        else None,
        "signature": None,  # filled in by the caller via evidence_signing.sign_bundle()
    }


def write_json(bundle: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence_manifest(directory: Path, out_path: Path | None = None) -> Path:
    """Hash every evidence artifact except the manifest itself -- the input
    evidence_integrity.EvidenceSigner.sign_manifest() (when
    EVIDENCE_SIGNING_KEY is configured) signs and
    evidence_integrity.verify_evidence_directory() later re-checks file by
    file, independent of evidence_signing.py's bundle-level signature."""
    destination = out_path or directory / "manifest.json"
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.resolve() != destination.resolve():
            files.append({"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)})
    payload = {"algorithm": "SHA-256", "signatureStatus": "UNSIGNED", "files": files}
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def write_pdf_best_effort(bundle: dict, out_path: Path) -> str | None:
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "DONTAI Evidence Bundle", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)

        def safe(value: Any) -> str:
            text = json.dumps(value, ensure_ascii=True, default=str) if isinstance(value, (dict, list)) else str(value)
            return text.encode("latin-1", "replace").decode("latin-1")

        verdict = bundle.get("verdict") or {}
        summary = [
            ("Verdict", verdict.get("verdict")),
            ("Confidence", verdict.get("confidence")),
            ("Discovered URL", bundle.get("discoveredUrl")),
            ("Discovered at", bundle.get("discoveredAt")),
            ("Rights holder", bundle.get("rightsHolder")),
            ("pHash distance", bundle.get("phashDistance")),
        ]
        for label, value in summary:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(38, 6, label)
            pdf.set_font("Helvetica", size=9)
            pdf.multi_cell(0, 6, safe(value))

        for heading, key in (
            ("Watermark", "watermarkDetection"),
            ("Robust fingerprint", "robustFingerprint"),
            ("C2PA detection", "c2paDetection"),
            ("Model-leak detection", "modelLeakDetection"),
            ("On-chain record", "onchainTransaction"),
            ("Verdict reasons", "verdict"),
        ):
            value = bundle.get(key)
            if value is None:
                continue
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", size=7)
            pdf.multi_cell(0, 4, safe(value)[:3000])

        screenshot = bundle.get("screenshotPath")
        if screenshot and Path(screenshot).is_file():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Captured page", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(screenshot), x=10, y=25, w=190)
        pdf.output(str(out_path))
        return str(out_path)
    except Exception:
        return None
