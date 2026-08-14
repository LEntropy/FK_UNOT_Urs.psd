import { api, ApiError } from "./client";

export interface RemeasureResult {
  styleDriftScore: number | null;
  styleSimilarityToOriginal: number | null;
  perceptualPsnrDb: number | null;
  perceptualRmse: number | null;
}

export const remeasureProtection = (artworkId: string) =>
  api.post<RemeasureResult>(`/artworks/${artworkId}/remeasure-protection`);

/** Per-architecture (SD1.5/SDXL) result of the Test Lab's real-LoRA-training
 * score -- see apps/protection-svc/docker/strongprotect/protection_score.py's
 * module doc for the mechanism. Sample images are base64-encoded PNGs,
 * meant to be turned into `data:image/png;base64,...` src URLs directly. */
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

/** Kicks off the real-LoRA-training score job -- several minutes on RunPod
 * Serverless, unlike remeasureProtection above. Returns immediately with a
 * jobId; callers poll getScoreProtectionJob. */
export const createScoreProtectionJob = (artworkId: string) =>
  api.post<{ jobId: string }>(`/artworks/${artworkId}/score-protection`);

export const getScoreProtectionJob = (artworkId: string, jobId: string) =>
  api.get<ScoreProtectionJob>(`/artworks/${artworkId}/score-protection/${jobId}`);

/** The persisted last real result for this artwork, if any -- survives a
 * closed/reloaded tab, unlike the jobId a live run tracks in React state.
 * Resolves to null (not a thrown ApiError) when nothing has been run yet,
 * since that's the common, expected state for most artworks. */
export async function getStoredScoreProtectionResult(
  artworkId: string,
): Promise<(ScoreProtectionJob & { checkedAt: number }) | null> {
  try {
    return await api.get<ScoreProtectionJob & { checkedAt: number }>(`/artworks/${artworkId}/score-protection-result`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}
