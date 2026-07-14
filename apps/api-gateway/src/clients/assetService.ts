import { env } from "../env.js";

/**
 * Thin pass-through to asset-service's own contract (apps/asset-service/
 * README.md) -- api-gateway's job here is auth + injecting the caller's
 * identity, not reshaping the artwork payload.
 */

export interface CreateArtworkRequest {
  title: string;
  sourceImageUri: string;
  protectionProfile?: "L1_PREVIEW" | "L2_PORTFOLIO" | "L3_ANTI_TRAIN";
  allowAiTraining?: boolean;
}

export async function createArtwork(req: CreateArtworkRequest, creatorId: string, ownerWalletAddress: string) {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...req, creatorId, ownerWalletAddress }),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export async function listArtworks(creatorId: string) {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks?creatorId=${encodeURIComponent(creatorId)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export async function getArtwork(id: string) {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export class AssetServiceError extends Error {
  constructor(public status: number, public body: unknown) {
    super(`asset-service request failed: ${status}`);
  }
}
