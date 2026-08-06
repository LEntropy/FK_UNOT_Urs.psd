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
"""

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
    out_path.write_text(json.dumps(bundle, indent=2, default=str))


def write_pdf_best_effort(bundle: dict, out_path: Path) -> str | None:
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 10, "DONTAI Evidence Bundle", ln=True)
        pdf.set_font("Helvetica", size=9)
        for key, value in bundle.items():
            line = f"{key}: {value}"
            pdf.multi_cell(0, 6, line[:500])
        pdf.output(str(out_path))
        return str(out_path)
    except Exception:
        return None
