"""Calls api-gateway's /internal/sign-evidence to fill in an evidence
bundle's previously-always-null `signature` field (see evidence_bundle.py's
former docstring) -- api-gateway holds the Ed25519 signing key (KMS
envelope-encrypted, same pattern as blockchain-svc's relayer key; see
apps/api-gateway/src/evidenceSigning.ts) since the real KMS C server has no
Sign() RPC of its own to call directly from here.

Best-effort, matching this module's siblings (evidence_capture.py's
screenshot, evidence_bundle.py's PDF): a signing failure (api-gateway down,
KMS unreachable) degrades the bundle to an unsigned one rather than losing
the whole case, same reasoning as those other non-critical-enrichment steps.
"""

import os

import httpx

API_GATEWAY_URL = os.environ.get("API_GATEWAY_URL", "http://localhost:4000")


def sign_bundle(bundle: dict) -> dict | None:
    """Returns {"signature", "publicKeyPem", "algorithm"} or None on any
    failure. `bundle` should NOT include the `signature` key itself -- the
    caller sets it from this function's result afterward, same as every
    other field this module doesn't compute (watermark_result, headers,
    etc. are all filled in by the caller before this runs)."""
    try:
        resp = httpx.post(f"{API_GATEWAY_URL}/internal/sign-evidence", json=bundle, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None
