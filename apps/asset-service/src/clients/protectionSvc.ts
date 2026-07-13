import { env } from "../env.js";

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
}

export interface VariantResult {
  name: string;
  width: number;
  height: number;
  scaleVsSource: number;
  protectionStatus: string;
}

export interface ProtectJob {
  jobId: string;
  status: "queued" | "processing" | "completed" | "failed";
  protectedImageUri?: string;
  perceptualHash?: string;
  metadataHash?: string;
  appliedPreset?: string;
  eotUsed?: boolean;
  size?: number;
  sizeValidated?: boolean;
  variants?: VariantResult[];
  processingTimeMs?: number;
  error?: string;
}

export async function createProtectJob(req: ProtectRequest): Promise<{ jobId: string; status: string }> {
  const res = await fetch(`${env.PROTECTION_SVC_URL}/protect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (res.status !== 202) {
    throw new Error(`protection-svc POST /protect failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function getProtectJob(jobId: string): Promise<ProtectJob> {
  const res = await fetch(`${env.PROTECTION_SVC_URL}/protect/${jobId}`);
  if (!res.ok) {
    throw new Error(`protection-svc GET /protect/${jobId} failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
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
  { intervalMs = 3000, timeoutMs = 30 * 60 * 1000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<ProtectJob> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const job = await getProtectJob(jobId);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`protection-svc job ${jobId} did not complete within ${timeoutMs}ms`);
}
