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

export interface CreateArtworkWithFileRequest {
  title: string;
  protectionProfile?: "L1_PREVIEW" | "L2_PORTFOLIO" | "L3_ANTI_TRAIN";
  allowAiTraining?: boolean;
  file: { buffer: Buffer; originalname: string; mimetype: string };
}

/**
 * Real browser file uploads (multipart/form-data all the way through, not
 * a server-side sourceImageUri path -- see routes/artworks.ts's module
 * doc). Re-packs the file bytes into a fresh multipart body for
 * asset-service using Node's built-in FormData/Blob (global since Node
 * 18, no extra dependency) -- api-gateway's only job here is auth +
 * injecting identity, same as createArtwork above, just carrying bytes
 * instead of a JSON string this time.
 */
export async function createArtworkWithFile(
  req: CreateArtworkWithFileRequest,
  creatorId: string,
  ownerWalletAddress: string,
) {
  const form = new FormData();
  form.set("title", req.title);
  form.set("creatorId", creatorId);
  form.set("ownerWalletAddress", ownerWalletAddress);
  if (req.protectionProfile) form.set("protectionProfile", req.protectionProfile);
  if (req.allowAiTraining !== undefined) form.set("allowAiTraining", String(req.allowAiTraining));
  form.set("image", new Blob([new Uint8Array(req.file.buffer)], { type: req.file.mimetype }), req.file.originalname);

  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks`, { method: "POST", body: form });
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
