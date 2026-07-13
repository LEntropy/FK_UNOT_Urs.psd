"""Assembles the evidence bundle exactly per PROJECT_DESIGN.md §3-7's field
list: 원본해시, 보호본해시, 등록시각, 권리자, 워터마크검출, 발견URL, 발견시각,
스크린샷, HTTP헤더, 온체인 트랜잭션, 내부 서명 -> JSON (always) + best-effort PDF.

`signature` is deliberately always null. KMS (the signing authority per
§6-1) is a separate, in-progress workstream owned by another teammate --
inventing a placeholder signing scheme here would just create a second
thing to reconcile later instead of one clean integration point. Same
treatment as the documented C2PA claim-signature gap in
apps/protection-svc/rust-core/README.md: flag the gap, don't fake it.
"""

import json
from pathlib import Path
from typing import Any


def build_bundle(
    *,
    artwork: dict,
    source_url: str | None,
    detected_at: float,
    phash_distance: int | None,
    watermark_result: dict | None,
    headers: dict | None,
    screenshot_path: str | None,
) -> dict[str, Any]:
    ownership_records = artwork.get("ownershipRecords") or []
    onchain = ownership_records[0] if ownership_records else None

    return {
        "originalHash": artwork.get("perceptualHash"),
        "protectedHash": artwork.get("perceptualHash"),  # registered hash IS the watermarked/published image's hash
        "registeredAt": onchain.get("registeredAt") if onchain else None,
        "rightsHolder": artwork.get("ownerWalletAddress"),
        "watermarkDetection": watermark_result,
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
        "signature": None,  # see module docstring: KMS signing not wired up yet
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
