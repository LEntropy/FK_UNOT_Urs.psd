import { env } from "../env.js";

/**
 * Thin pass-through to asset-service's own contract (apps/asset-service/
 * README.md) -- api-gateway's job here is auth + injecting the caller's
 * identity, not reshaping the artwork payload.
 */

export interface CreateArtworkRequest {
  title: string;
  sourceImageUri: string;
  protectionProfile?: "L1_PREVIEW" | "L2_PORTFOLIO" | "L3_ANTI_TRAIN" | "STRONG_PROTECTION";
  strongProtectionLatentEpsilon?: number;
  strongProtectionPixelEpsilon?: number;
  allowAiTraining?: boolean;
  tags?: string[];
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
  protectionProfile?: "L1_PREVIEW" | "L2_PORTFOLIO" | "L3_ANTI_TRAIN" | "STRONG_PROTECTION";
  strongProtectionLatentEpsilon?: number;
  strongProtectionPixelEpsilon?: number;
  allowAiTraining?: boolean;
  tags?: string[];
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
  if (req.strongProtectionLatentEpsilon !== undefined) form.set("strongProtectionLatentEpsilon", String(req.strongProtectionLatentEpsilon));
  if (req.strongProtectionPixelEpsilon !== undefined) form.set("strongProtectionPixelEpsilon", String(req.strongProtectionPixelEpsilon));
  if (req.allowAiTraining !== undefined) form.set("allowAiTraining", String(req.allowAiTraining));
  if (req.tags !== undefined) form.set("tags", JSON.stringify(req.tags)); // multipart has no array type -- see asset-service's own identical parsing
  form.set("image", new Blob([new Uint8Array(req.file.buffer)], { type: req.file.mimetype }), req.file.originalname);

  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks`, { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export interface SuggestedTag {
  tag: string;
  score: number;
}

/**
 * Upload-preview step: proxies the picked file to asset-service's own
 * suggest-tags route (which in turn calls protection-svc's CLIP-based
 * ranking, see ml-engine/src/tag_suggest.py) before the user commits to
 * the real upload. Same re-packing-into-a-fresh-FormData approach as
 * createArtworkWithFile above, for the same reason (carrying raw bytes
 * through, not a JSON string).
 */
export async function suggestTags(file: { buffer: Buffer; originalname: string; mimetype: string }): Promise<SuggestedTag[]> {
  const form = new FormData();
  form.set("image", new Blob([new Uint8Array(file.buffer)], { type: file.mimetype }), file.originalname);

  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/suggest-tags`, { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body.tags;
}

export async function listArtworks(creatorId: string) {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks?creatorId=${encodeURIComponent(creatorId)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Profile-page feature (2026-08-14) -- unlike listArtworks above (always
 * "the caller's own gallery", every status/visibility included), this is
 * for viewing anyone's public work grid: always publicOnly=true,
 * server-enforced (see asset-service's own route doc), so a caller can
 * never use this to peek at another creator's drafts/failed/private rows. */
export async function listPublicArtworksByCreator(creatorId: string) {
  const res = await fetch(
    `${env.ASSET_SERVICE_URL}/artworks?creatorId=${encodeURIComponent(creatorId)}&publicOnly=true`,
  );
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Tag-search feature (2026-08-14) -- global discovery across every
 * creator's work, so asset-service always forces publicOnly here itself
 * (see that route's own doc); nothing to pass explicitly on this side. */
export async function searchArtworksByTag(tag: string) {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks?tag=${encodeURIComponent(tag)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Platform-wide real counters (2026-08-16 redesign) -- backs the web
 * right-sidebar widget. See asset-service's GET /stats doc for why these
 * are always real aggregate counts, never placeholder numbers. */
export async function getPlatformStats() {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/stats`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body as { publishedArtworks: number; strongProtectionArtworks: number };
}

export interface RemeasureResult {
  styleDriftScore: number | null;
  styleSimilarityToOriginal: number | null;
  perceptualPsnrDb: number | null;
  perceptualRmse: number | null;
}

/** Thin pass-through to asset-service's own POST /:id/remeasure-protection (the Test Lab's "재실행" button). */
export async function remeasureProtection(id: string): Promise<RemeasureResult> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/remeasure-protection`, {
    method: "POST",
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Per-architecture (SD1.5/SDXL) result of the Test Lab's real-LoRA-training
 * score -- matches asset-service's (and protection_score.py's) shape
 * exactly. Sample images are base64-encoded PNGs. */
export interface ScoreProtectionArchResult {
  baselineSimilarity: number;
  protectedSimilarity: number;
  delta: number;
  verdict: "PROTECTED" | "WEAK" | "NOT_PROTECTED";
  baselineSamples: string[];
  protectedSamples: string[];
}

export interface ScoreProtectionJob {
  status: "queued" | "processing" | "completed" | "failed";
  sd15?: ScoreProtectionArchResult;
  sdxl?: ScoreProtectionArchResult;
  threshold?: number;
  error?: string;
}

/** Thin pass-through to asset-service's own POST /:id/score-protection --
 * kicks off the Test Lab's real-LoRA-training score job and returns
 * immediately with a jobId (this job runs minutes on RunPod, unlike
 * remeasureProtection above). */
export async function createScoreProtectionJob(id: string): Promise<{ jobId: string }> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/score-protection`, {
    method: "POST",
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Thin pass-through to asset-service's own GET /score-protection-jobs/:jobId. */
export async function getScoreProtectionJob(jobId: string): Promise<ScoreProtectionJob> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/score-protection-jobs/${encodeURIComponent(jobId)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Thin pass-through to asset-service's own GET /:id/score-protection-result
 * -- the persisted last real result for this artwork, retrievable any time
 * (not just while the job that produced it is still live in the caller's
 * own state). Returns null on asset-service's 404 ("no stored result yet")
 * rather than throwing, since that's an expected, common state for this
 * route (an artwork simply hasn't had a score-protection run) not an error. */
export async function getStoredScoreProtectionResult(id: string): Promise<(ScoreProtectionJob & { checkedAt: number }) | null> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/score-protection-result`);
  if (res.status === 404) return null;
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** `viewerId` (when given) makes asset-service's response include
 * originalPreviewUnlockedByViewer for that specific requester -- omitted
 * for callers that don't need it (e.g. detectionRouter's own getArtwork
 * calls, which only read creatorId). */
export async function getArtwork(id: string, viewerId?: string) {
  const query = viewerId ? `?userId=${encodeURIComponent(viewerId)}` : "";
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}${query}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Cancel-upload feature (2026-08-14) -- thin pass-through to asset-
 * service's own POST /:id/cancel. Two different response shapes depending
 * on artwork state (see that route's own doc): {status:"FAILED",
 * cancelled:true} while in flight, or {deleted:true} once already
 * terminal. Both are returned as-is; api-gateway doesn't need to
 * distinguish them, just proxy. */
export async function cancelArtwork(id: string): Promise<{ id: string; status?: string; cancelled?: boolean; deleted?: boolean }> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/cancel`, { method: "POST" });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Creator-only opt-OUT from the coin-unlock original-preview feature
 * (2026-08-14 redesign -- see asset-service schema.ts's originalPreviewBlocked
 * doc for why this replaced the old opt-IN enable/disable pair). The coin
 * button is available by default on every artwork; this only ever turns it
 * off (blocked: true) or back on (blocked: false). */
export async function setOriginalPreviewBlocked(id: string, blocked: boolean): Promise<{ originalPreviewBlocked: boolean }> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/original-preview-blocked`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blocked }),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Viewer-side unlock -- free for the creator, coin-gated for anyone else
 * (asset-service's own coins.ts). May 402 with {error: "insufficient_coins",
 * required, balance}. */
export async function unlockOriginalPreview(id: string, userId: string): Promise<{ originalPreviewUnlockedByViewer: boolean }> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/original-preview/unlock`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId }),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export interface CoinTransaction {
  id: string;
  userId: string;
  amount: number;
  reason: string;
  relatedArtworkId: string | null;
  chainTxHash: string | null;
  balanceAfter: number;
  createdAt: string;
}

/** Thin pass-through to asset-service's own GET /coins/balance (coins.ts's
 * lazy signup-bonus grant happens on the other side of this call). */
export async function getCoinBalance(userId: string): Promise<{ balance: number; recentTransactions: CoinTransaction[] }> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/coins/balance?userId=${encodeURIComponent(userId)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export interface LoraGenerationJob {
  id: string;
  userId: string;
  sourceArtworkId: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
  resultPath: string | null;
  coinCost: number;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

/** Coin-system feature (2026-08-10) -- thin pass-throughs to asset-service's
 * own /lora-jobs routes (routes/loraJobs.ts). userId always comes from the
 * verified JWT (routes/artworks.ts), never the request body. */
export async function createLoraJob(sourceArtworkId: string, userId: string): Promise<{ id: string }> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/lora-jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sourceArtworkId, userId }),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export async function getLoraJob(id: string, userId: string): Promise<LoraGenerationJob> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/lora-jobs/${encodeURIComponent(id)}?userId=${encodeURIComponent(userId)}`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

/** Returns the raw fetch Response so the caller can stream the file body
 * straight through (routes/artworks.ts's download route) rather than
 * buffering a multi-MB .safetensors file in memory. */
export async function fetchLoraJobDownload(id: string, userId: string): Promise<Response> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/lora-jobs/${encodeURIComponent(id)}/download?userId=${encodeURIComponent(userId)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: `asset-service returned ${res.status}` }));
    throw new AssetServiceError(res.status, body);
  }
  return res;
}

export type BotAction = "ALLOW" | "BLOCK" | "LOG_ONLY";

export interface ArtworkBotPolicy {
  defaultAction: BotAction;
  groupPolicies: Record<string, BotAction>;
  botOverrides: Record<string, BotAction>;
}

/** Thin pass-throughs to asset-service's own GET/PUT /:id/bot-policy (the
 * durable counterpart to delivery-gateway's in-memory BotPolicyStore cache
 * -- see that Rust module's own doc, and asset-service schema.ts's
 * botPolicies table doc, for why the system of record lives here). */
export async function getBotPolicy(id: string): Promise<ArtworkBotPolicy> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/bot-policy`);
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export async function setBotPolicy(id: string, policy: ArtworkBotPolicy): Promise<ArtworkBotPolicy> {
  const res = await fetch(`${env.ASSET_SERVICE_URL}/artworks/${encodeURIComponent(id)}/bot-policy`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(policy),
  });
  const body = await res.json();
  if (!res.ok) {
    throw new AssetServiceError(res.status, body);
  }
  return body;
}

export interface BotAccessLogEntry {
  id: number;
  artworkId: string;
  botName: string;
  botGroup: string;
  action: BotAction;
  clientIp: string;
  userAgent: string;
  responseStatus: number;
  timestamp: string;
}

export async function getBotAccessLogs(artworkId: string, limit = 100): Promise<BotAccessLogEntry[]> {
  const res = await fetch(
    `${env.ASSET_SERVICE_URL}/artworks/bot-access-logs?artworkId=${encodeURIComponent(artworkId)}&limit=${limit}`,
  );
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
