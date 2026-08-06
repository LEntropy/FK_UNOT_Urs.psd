"""Calls api-gateway's /internal/notify-evidence-ready to email an
artwork's creator when a case reaches EVIDENCE_READY -- same "detection-svc
can't do this itself, api-gateway owns the resource this needs" reasoning
as evidence_signing.py's sign_bundle() (api-gateway owns the signing key
there; here it owns user email, via its own users table). Best-effort,
matching every other non-critical-enrichment step in this pipeline
(screenshot, PDF, C2PA, signing): a notification failure (api-gateway
down, SMTP not configured/unreachable) must never fail or reopen an
already-successful case.
"""

import os

import httpx

API_GATEWAY_URL = os.environ.get("API_GATEWAY_URL", "http://localhost:4000")


def notify_evidence_ready(creator_id: str, artwork_title: str, case_id: str, evidence_type: str) -> bool:
    """Returns whether an email was actually sent (api-gateway's own
    response) -- False both when the request fails outright and when it
    succeeds but SMTP isn't configured on that end (see
    apps/api-gateway/src/notify.ts's own best-effort posture). Never
    raises -- callers should not need a try/except around this."""
    try:
        resp = httpx.post(
            f"{API_GATEWAY_URL}/internal/notify-evidence-ready",
            json={"creatorId": creator_id, "artworkTitle": artwork_title, "caseId": case_id, "evidenceType": evidence_type},
            timeout=10.0,
        )
        resp.raise_for_status()
        return bool(resp.json().get("sent", False))
    except (httpx.HTTPError, ValueError):
        return False
