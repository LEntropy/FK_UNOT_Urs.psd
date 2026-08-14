"""Thin HTTP client for asset-service's GET /artworks/:id -- the ONLY way
detection-svc reads artwork data. Deliberately not a DB connection: per the
project's established pattern (protection-svc and blockchain-svc are both
pure APIs that don't touch each other's storage), detection-svc is a peer
service, not a co-owner of asset-service's SQLite file. This also means
zero coordination is needed with whoever is actively changing asset-service.

GET /artworks/:id already returns everything a scan needs: perceptualHash,
protectedImageUri, ownerWalletAddress, and embedded ownershipRecords with
the on-chain txHash/blockNumber -- see apps/asset-service/src/routes/artworks.ts.
"""

import httpx


class ArtworkNotFoundError(Exception):
    pass


async def get_artwork(asset_service_url: str, artwork_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{asset_service_url}/artworks/{artwork_id}")
    if resp.status_code == 404:
        raise ArtworkNotFoundError(artwork_id)
    resp.raise_for_status()
    return resp.json()


async def list_artworks(asset_service_url: str) -> list[dict]:
    """Fetch the catalog for scheduled monitoring; accepts common envelope shapes."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{asset_service_url}/artworks")
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, list):
        return payload
    for key in ("artworks", "items", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("asset-service artwork list response has no list payload")
