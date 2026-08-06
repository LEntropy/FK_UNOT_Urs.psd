"""Fills RUNBOOK.md's own DMCA/infringement notice template from a real
evidence bundle (GET /evidence/{caseId}'s own bundle shape), so runbook
step 5 doesn't require a human hand-copying values out of a JSON blob one
field at a time.

Leaves fields this service genuinely has no data for as bracketed
placeholders, exactly as RUNBOOK.md's own template already does -- the
host's DMCA/abuse contact and the rights holder's real name/signature both
still require a human decision or lookup this service doesn't have
(`rightsHolder` on a bundle is a wallet address, not a legal name). This
narrows the manual work to "fill in who you are and where you're sending
it," not "find every value in the bundle and copy it in by hand."

Not applicable to a model-leak bundle (RUNBOOK.md Step 5's third bullet --
"Suspected AI training dataset inclusion" -- points to a different,
manual escalation path, not a DMCA takedown notice; a suspect LoRA file
url isn't "a URL that hosts a copy of this work").
"""

from datetime import date, datetime, timezone

TEMPLATE = """To: [host/platform's designated DMCA agent or abuse contact]
From: [rights holder name / contact]
Date: {today}

Re: Notice of Copyright Infringement

I am the copyright owner (or authorized to act on the owner's behalf) of
the artwork registered on-chain at:

  Content hash:     {content_hash}
  Chain / registry:  {chain} / {registry_address}
  Transaction:        {tx_hash}
  Registered at:       {registered_at}

The following URL hosts a copy of this work without authorization:

  {discovered_url}
  Discovered at: {discovered_at}

Supporting evidence (attached):
  - Perceptual-hash similarity to the registered work
    (distance: {phash_distance} / 256, lower = more similar)
  - {watermark_line}
  - Screenshot and HTTP headers captured at time of discovery

I have a good-faith belief that this use is not authorized by the
copyright owner, its agent, or the law. I swear, under penalty of perjury,
that the information in this notice is accurate and that I am the
copyright owner or authorized to act on the owner's behalf.

Signature: [name]
"""


def build_dmca_notice(bundle: dict) -> str:
    """`bundle` is exactly what GET /evidence/{caseId} returns per entry
    (evidence_bundle.build_bundle's own shape) -- call
    is_dmca_applicable(bundle) first; this doesn't check evidence_type
    itself."""
    onchain = bundle.get("onchainTransaction") or {}
    watermark = bundle.get("watermarkDetection")
    if watermark and watermark.get("isMatch"):
        watermark_line = f"Embedded watermark match: {watermark['recoveredHex']} (confidence {watermark['avgConfidence']:.0%})"
    else:
        watermark_line = "[No watermark match recorded for this bundle]"

    discovered_at = bundle.get("discoveredAt")
    discovered_at_str = (
        datetime.fromtimestamp(discovered_at, tz=timezone.utc).isoformat()
        if discovered_at is not None
        else "[unknown]"
    )
    phash_distance = bundle.get("phashDistance")

    return TEMPLATE.format(
        today=date.today().isoformat(),
        content_hash=bundle.get("originalHash") or "[unavailable]",
        chain=onchain.get("chain") or "[unavailable]",
        registry_address=onchain.get("registryAddress") or "[unavailable]",
        tx_hash=onchain.get("txHash") or "[unavailable]",
        registered_at=bundle.get("registeredAt") or "[not registered on-chain]",
        discovered_url=bundle.get("discoveredUrl") or "[unavailable]",
        discovered_at=discovered_at_str,
        phash_distance=phash_distance if phash_distance is not None else "[not measured]",
        watermark_line=watermark_line,
    )


def is_dmca_applicable(bundle: dict) -> bool:
    return bundle.get("modelLeakDetection") is None
