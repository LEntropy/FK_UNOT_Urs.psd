"""Best-effort on-chain timestamp anchor for an evidence bundle -- reuses
blockchain-svc's existing POST /assets/register (the same
commit-reveal-on-OwnershipRegistry flow real artwork registration already
uses), via its `content` convenience field (an arbitrary string, hashed
with keccak256 -- explicitly NOT the pHash-formula path other services
use for real artwork registration, which is exactly why this fits: an
evidence bundle isn't "an artwork," it's some other content whose
existence-at-this-timestamp is worth proving the same way).

Why reuse the artwork registry instead of a new contract: the registry's
actual job is "this hash existed, owned by this address, at this block" --
nothing about that is artwork-specific, and standing up a second contract
(deploy, ABI, tests, a second ETH_RPC_URL/PRIVATE_KEY wiring) for the same
underlying guarantee would be real, avoidable engineering cost for a
project already relying on one relayer wallet.

Real (if testnet) gas cost per call, and the commit-reveal flow is two
on-chain transactions with a wait in between -- see blockchain-svc's own
src/routes/register.ts. Opt-in (EVIDENCE_ANCHOR_ENABLED, server.py) for
exactly this reason: automatically anchoring every EVIDENCE_READY case
could drain the relayer wallet's testnet funds faster than anyone
notices, a real recurring problem this project's own history already
hit (see the relayer-balance-monitoring work elsewhere in this repo).
Best-effort like every other enrichment step here: a failure (relayer
out of funds, blockchain-svc down) must never fail an otherwise-complete
case.
"""

import os

import httpx

BLOCKCHAIN_SVC_URL = os.environ.get("BLOCKCHAIN_SVC_URL", "http://localhost:3001")


def anchor_evidence_bundle(bundle_content: str, owner_address: str) -> dict | None:
    """`bundle_content` should be the same canonical (signature-excluded)
    JSON string used for evidence_signing.sign_bundle, so the anchored
    hash and the signed content agree on what "the bundle" was at the
    moment both were computed. Returns {txHash, blockNumber, contentHash}
    or None on any failure (relayer down, insufficient funds, timeout,
    unexpected response shape)."""
    try:
        resp = httpx.post(
            f"{BLOCKCHAIN_SVC_URL}/assets/register",
            json={"ownerAddress": owner_address, "doNotTrain": False, "content": bundle_content},
            timeout=60.0,  # commit + wait-for-maturity + reveal -- two real on-chain txs, not a quick call
        )
        resp.raise_for_status()
        body = resp.json()
        return {"txHash": body["txHash"], "blockNumber": body["blockNumber"], "contentHash": body["contentHash"]}
    except (httpx.HTTPError, KeyError, ValueError):
        return None
