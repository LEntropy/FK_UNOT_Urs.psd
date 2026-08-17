import { env } from "../env.js";
import { withRetry } from "../lib/retry.js";

/** Matches apps/protection-svc/INTEGRATION.md's job contract exactly. */
export interface ProtectRequest {
  imageUri: string;
  protectionProfile: string;
  eot?: boolean;
  styleTargetUri?: string;
  title: string;
  creatorId: string;
  allowAiTraining: boolean;
  watermarkPayloadHex?: string;
  size?: number;
  // Independent of protectionProfile (which stays one of the three
  // style_cloak presets, used as the fallback tier if this fails) --
  // protection-svc's own ProtectRequest keeps these as two separate
  // fields, not a fourth protectionProfile enum value, since strong
  // protection is a different mechanism entirely (dual-arch RunPod
  // Serverless attack), not another style_cloak preset.
  strongProtection?: boolean;
  // Advanced-options upload feature (2026-08-08, redesigned 2026-08-15) --
  // ignored unless strongProtection is also true. undefined means run at
  // clean_protect.py's own CLEAN_FULL preset values, not "no protection."
  // See protection-svc's server.py ProtectRequest for the matching field
  // name.
  strongProtectionEpsilon?: number;
}

export interface VariantResult {
  name: string;
  width: number;
  height: number;
  scaleVsSource: number;
  protectionStatus: string;
  // Real per-variant file path under protection-svc's own out/<jobId>/
  // tree (protection_out volume, mounted at the same absolute path in
  // delivery-gateway too -- see docker-compose.yml's comment on that
  // mount). Optional only for backward compatibility with jobs run before
  // orchestrate.py started reporting this -- orchestration.ts falls back
  // to protectedImageUri when absent.
  path?: string;
}

export interface ProtectJob {
  jobId: string;
  status: "queued" | "processing" | "completed" | "failed";
  protectedImageUri?: string;
  perceptualHash?: string;
  metadataHash?: string;
  appliedPreset?: string;
  // What actually ran, distinct from the strongProtection request flag --
  // orchestrate.py's dual-arch attack can fail and fall back to plain
  // style_cloak rather than fail the whole job. See schema.ts's
  // usedStrongProtection column doc for why this matters downstream.
  usedStrongProtection?: boolean;
  eotUsed?: boolean;
  size?: number;
  sizeValidated?: boolean;
  watermarkPayloadHex?: string;
  variants?: VariantResult[];
  processingTimeMs?: number;
  error?: string;
  // Real per-upload measurements (orchestrate.py's compute_protection_metrics)
  // -- null/absent when protection-svc skipped the measurement (USE_REMOTE_GPU,
  // or it errored), not the same as a measured value of 0.
  styleDriftScore?: number | null;
  styleSimilarityToOriginal?: number | null;
  perceptualPsnrDb?: number | null;
}

export interface SuggestedTag {
  tag: string;
  score: number;
}

/**
 * Fast, synchronous unlike createProtectJob/pollProtectJob -- a single CLIP
 * forward pass (see protection-svc/server.py's /suggest-tags and
 * ml-engine/src/tag_suggest.py), meant to be called directly from an HTTP
 * request handler (routes/artworks.ts's own suggest-tags proxy) while the
 * user is still on the upload-preview screen, not from background
 * orchestration.
 */
export async function suggestTags(imageUri: string, topK = 10): Promise<SuggestedTag[]> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/suggest-tags`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageUri, topK }),
    });
    if (!res.ok) {
      throw new Error(`protection-svc POST /suggest-tags failed: ${res.status} ${await res.text()}`);
    }
    const body = (await res.json()) as { tags: SuggestedTag[] };
    return body.tags;
  });
}

export interface RemeasureResult {
  styleDriftScore: number | null;
  styleSimilarityToOriginal: number | null;
  perceptualPsnrDb: number | null;
  perceptualRmse: number | null;
}

/**
 * Live counterpart to the styleDriftScore/etc stored on the artwork row at
 * upload time (protect()'s own one-shot measurement) -- backs the Test
 * Lab's "재실행" button. Synchronous, unlike createProtectJob/pollProtectJob:
 * server.py's /remeasure is a few VGG19 forward passes plus one SSH round
 * trip under USE_REMOTE_GPU, not a cloak/training job.
 */
export async function remeasureProtection(
  originalImageUri: string,
  cloakedImageUri: string,
): Promise<RemeasureResult> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/remeasure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ originalImageUri, cloakedImageUri }),
    });
    if (!res.ok) {
      throw new Error(`protection-svc POST /remeasure failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

/**
 * Synchronous, like suggestTags above -- protection-svc's /original-preview
 * is pure PIL (no GPU), see ml-engine/src/original_preview.py's module doc.
 * Returns a local file path (protection-svc's own filesystem, same trust
 * boundary as every other imageUri in this PoC) that routes/artworks.ts
 * copies into this service's own storage before writing the asset_versions
 * row.
 */
export async function createOriginalPreview(
  imageUri: string,
  watermarkPayloadHex: string,
): Promise<{ previewUri: string; width: number; height: number }> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/original-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageUri, watermarkPayloadHex }),
    });
    if (!res.ok) {
      throw new Error(`protection-svc POST /original-preview failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

export async function createProtectJob(req: ProtectRequest): Promise<{ jobId: string; status: string }> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/protect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (res.status !== 202) {
      throw new Error(`protection-svc POST /protect failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

/** Cancel-upload feature (2026-08-14) -- best-effort, not retried through
 * withRetry like the others above: a cancel request should fire once and
 * return quickly, not spend several retry rounds against a service that
 * might be mid-restart itself. Swallows its own failure (routes/artworks.ts's
 * cancel route treats this as fire-and-forget) since the artwork's own
 * status change is what the caller actually depends on, not this call
 * succeeding. */
export async function cancelProtectJob(jobId: string): Promise<void> {
  try {
    await fetch(`${env.PROTECTION_SVC_URL}/protect/${jobId}/cancel`, { method: "POST" });
  } catch {
    // best-effort, see this function's own doc
  }
}

export async function getProtectJob(jobId: string): Promise<ProtectJob> {
  // Retried here, not just once at the top of pollProtectJob's loop -- a
  // single dropped poll request over a multi-minute-to-hours job shouldn't
  // throw the whole job away; the next poll a few seconds later would have
  // succeeded anyway, this just doesn't wait for it.
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/protect/${jobId}`);
    if (!res.ok) {
      throw new Error(`protection-svc GET /protect/${jobId} failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

/**
 * Polls until the job reaches completed/failed. protection-svc's own jobs
 * can take from ~1 minute to hours (see ml-engine/README.md's size/EOT
 * timing notes) -- this is meant to be called from asset-service's own
 * background orchestration (routes/artworks.ts), never from inside an HTTP
 * request handler that a caller is waiting on.
 */
export async function pollProtectJob(
  jobId: string,
  // 2026-08-14: was 30min, shorter than protection-svc's own worst-case
  // ceiling for a real strong_protection job -- clean_protect.py's four-
  // stage RunPod Serverless chain can legitimately run up to the RunPod
  // endpoint's executionTimeoutMs (3000000ms/50min) or remote_gpu.py's own
  // client-side poll timeout (3300s/55min), and a real production run on
  // the actual Illustrious-XL checkpoint has been observed taking well
  // over 30min for the SDXL stage alone. This outer poll must stay
  // comfortably above BOTH of those inner ceilings, the same layered-
  // timeout principle already applied there (each ceiling here must
  // exceed the one it wraps) -- otherwise this is the one that always
  // fires first and fails a real, still-succeeding job, exactly what
  // happened live (errorMessage: "...did not complete within 1800000ms").
  { intervalMs = 3000, timeoutMs = 70 * 60 * 1000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<ProtectJob> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const job = await getProtectJob(jobId);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`protection-svc job ${jobId} did not complete within ${timeoutMs}ms`);
}

export interface ScoreProtectionRequest {
  originalImageUri: string;
  protectedImageUri: string;
  prompt: string;
}

/** Per-architecture (SD1.5/SDXL) result of the Test Lab's real-LoRA-training
 * score -- matches protection_score.py's return shape exactly. Sample
 * images are base64-encoded PNGs, meant to be turned into data: URLs
 * client-side (apps/web's TestLabPage), never persisted server-side. */
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

/**
 * Test Lab's on-demand real-LoRA-training protection score (see
 * protection-svc/docker/strongprotect/protection_score.py's module doc).
 * Job-based like createProtectJob, not synchronous like remeasureProtection
 * -- this trains four LoRAs on RunPod Serverless, the slowest job class
 * this project exposes.
 */
export async function createScoreProtectionJob(req: ScoreProtectionRequest): Promise<{ jobId: string; status: string }> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/score-protection`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (res.status !== 202) {
      throw new Error(`protection-svc POST /score-protection failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

export async function getScoreProtectionJob(jobId: string): Promise<ScoreProtectionJob> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/score-protection/${jobId}`);
    if (!res.ok) {
      throw new Error(`protection-svc GET /score-protection/${jobId} failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

export interface LoraGenerationRequest {
  imageUri: string;
  prompt: string;
  seed?: number;
  trainSteps?: number;
}

export interface LoraGenerationJob {
  status: "queued" | "processing" | "completed" | "failed";
  outputPath?: string;
  contentPrompt?: string;
  error?: string;
}

/**
 * Coin-system feature (2026-08-10): kicks off training a real, downloadable
 * SD1.5 LoRA on a single artwork image (see ml-engine/src/lora_generate.py's
 * module doc). Job-based like createScoreProtectionJob, not synchronous.
 */
export async function createLoraJob(req: LoraGenerationRequest): Promise<{ jobId: string; status: string }> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/lora-jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (res.status !== 202) {
      throw new Error(`protection-svc POST /lora-jobs failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

export async function getLoraJob(jobId: string): Promise<LoraGenerationJob> {
  return withRetry(async () => {
    const res = await fetch(`${env.PROTECTION_SVC_URL}/lora-jobs/${jobId}`);
    if (!res.ok) {
      throw new Error(`protection-svc GET /lora-jobs/${jobId} failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  });
}

/** Polls until the job reaches completed/failed -- same shape as
 * pollScoreProtectionJob below, used from routes/loraJobs.ts's background
 * (fire-and-forget) job runner. */
export async function pollLoraJob(
  jobId: string,
  { intervalMs = 5000, timeoutMs = 30 * 60 * 1000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<LoraGenerationJob> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const job = await getLoraJob(jobId);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`protection-svc job ${jobId} did not complete within ${timeoutMs}ms`);
}

/**
 * Fire-and-forget cleanup helper, not a user-facing poll -- see
 * routes/artworks.ts's /:id/score-protection handler. That route returns
 * the jobId to the caller immediately (this job runs many minutes on
 * RunPod), but the decrypted original image temp file it handed to
 * protection-svc as originalImageUri must stay on disk until protection-svc
 * has actually finished reading it. This polls the same job in the
 * background (uncoupled from any HTTP response) purely to know when it's
 * safe to delete that temp file.
 */
export async function pollScoreProtectionJob(
  jobId: string,
  { intervalMs = 5000, timeoutMs = 30 * 60 * 1000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<ScoreProtectionJob> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const job = await getScoreProtectionJob(jobId);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`protection-svc job ${jobId} did not complete within ${timeoutMs}ms`);
}
