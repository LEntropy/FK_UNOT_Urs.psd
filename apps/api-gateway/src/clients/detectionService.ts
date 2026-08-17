import { env } from "../env.js";

/**
 * Thin pass-through to detection-svc's own contract (apps/detection-svc/
 * README.md) -- same role as assetService.ts: auth + ownership live here,
 * detection-svc itself has none of its own. Backs the "테스트" tab's
 * 추적·증빙 test (scan/report a candidate URL, poll the resulting case,
 * read back the evidence bundle).
 */

export class DetectionServiceError extends Error {
  constructor(public status: number, public body: unknown) {
    super(`detection-svc request failed: ${status}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${env.DETECTION_SERVICE_URL}${path}`, init);
  const body = await res.json().catch(() => undefined);
  if (!res.ok) {
    throw new DetectionServiceError(res.status, body);
  }
  return body as T;
}

export interface DetectionCaseResponse {
  caseId: string;
  status: string;
}

export const scanArtwork = (artworkId: string) =>
  request<DetectionCaseResponse>(`/scan/${encodeURIComponent(artworkId)}`, { method: "POST" });

export const reportArtwork = (artworkId: string, suspectUrl: string) =>
  request<DetectionCaseResponse>("/reports", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ artworkId, suspectUrl }),
  });

export const reportModelLeak = (artworkId: string, suspectModelUrl: string) =>
  request<DetectionCaseResponse>("/model-leak-reports", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ artworkId, suspectModelUrl }),
  });

export interface EvidenceRecord {
  id: number;
  case_id: string;
  evidence_type: string;
  source_url: string | null;
  confidence: number | null;
  artifact_uri: string | null;
  detected_at: number;
}

export interface Case {
  id: string;
  artwork_id: string;
  status:
    | "OPEN"
    | "EVIDENCE_READY"
    | "NO_MATCH_FOUND"
    | "FAILED"
    | "NOTIFIED"
    | "RESOLVED"
    | "ESCALATED"
    | "AUTH_REQUIRED"
    | "ACCESS_DENIED";
  trigger: "scan" | "report" | "model_report";
  error_message: string | null;
  note: string | null;
  created_at: number;
  updated_at: number;
  evidence: EvidenceRecord[];
}

export const getCase = (caseId: string) => request<Case>(`/cases/${encodeURIComponent(caseId)}`);

/** RUNBOOK.md §7 steps 4-6's manual runbook progression -- server.py only
 * allows this from EVIDENCE_READY (or another manual status), never from
 * an automated-only state like OPEN/NO_MATCH_FOUND/FAILED. */
export const updateCaseStatus = (caseId: string, status: "NOTIFIED" | "RESOLVED" | "ESCALATED", note?: string) =>
  request<Case>(`/cases/${encodeURIComponent(caseId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, note }),
  });

export interface EvidenceBundle {
  originalHash: string | null;
  protectedHash: string | null;
  registeredAt: string | null;
  rightsHolder: string | null;
  watermarkDetection: {
    recoveredHex: string;
    avgConfidence: number;
    minConfidence: number;
    bitErrorRate: number;
    isMatch: boolean;
  } | null;
  c2paDetection: {
    hasManifest: boolean;
    signedByDontai: boolean;
    ownership: Record<string, unknown> | null;
    validationIssues: string[] | null;
  } | null;
  modelLeakDetection: {
    perPrompt: Array<{ prompt: string; avgBaseSimilarity: number; avgSuspectSimilarity: number; delta: number }>;
    meanDelta: number;
    stdevDelta: number;
    verdict: "SUSPECTED_LEAK" | "INCONCLUSIVE" | "NO_EVIDENCE";
    threshold: number;
  } | null;
  discoveredUrl: string | null;
  discoveredAt: number;
  phashDistance: number | null;
  screenshotPath: string | null;
  httpHeaders: Record<string, string> | null;
  onchainTransaction: {
    chain: string | null;
    registryAddress: string | null;
    txHash: string | null;
    blockNumber: number | null;
  } | null;
  signature: { signature: string; publicKeyPem: string; algorithm: string } | null;
  evidenceAnchor: { txHash: string; blockNumber: number; contentHash: string } | null;
}

export const getEvidence = (caseId: string) =>
  request<{ caseId: string; status: string; bundles: EvidenceBundle[] }>(
    `/evidence/${encodeURIComponent(caseId)}`,
  );

export interface DmcaNoticeResult {
  caseId: string;
  notices: Array<{ sourceUrl: string | null; notice: string | null; note: string | null }>;
}

/** RUNBOOK.md Step 5's DMCA template, auto-filled from the case's real
 * evidence bundles -- see dmca_notice.py's own doc for what's filled in
 * vs left as a bracketed placeholder. `notice` is null (with `note`
 * explaining why) for a model-leak bundle, which isn't a "this URL hosts
 * a copy of the work" situation a DMCA notice applies to. */
export const getDmcaNotice = (caseId: string) =>
  request<DmcaNoticeResult>(`/cases/${encodeURIComponent(caseId)}/dmca-notice`);

export interface EvidenceVerifyResult {
  caseId: string;
  manifests: Array<{
    artifactUri: string;
    valid: boolean;
    status: string;
    signature: { valid: boolean; status: string; keyId?: string } | null;
    files: Array<{ path: string; valid: boolean; sha256?: string; error?: string }>;
    sealed: boolean;
  }>;
}

/** Independent of GET /evidence/{caseId}'s bundle-level `signature` --
 * re-hashes each evidence directory's files against manifest.json right
 * now, so a "still valid" answer means the files on disk today, not just
 * at capture time. See evidence_integrity.py's module doc. */
export const verifyEvidence = (caseId: string) =>
  request<EvidenceVerifyResult>(`/evidence/${encodeURIComponent(caseId)}/verify`);

export interface VisionUsage {
  configured: boolean;
  month: string;
  used: number;
  limit: number;
  remaining: number;
}

/** Global (not per-artwork) Google Vision reverse-image-search quota --
 * see server.py's VISION_MONTHLY_LIMIT. No ownership check needed, this
 * isn't scoped to any one creator's data. */
export const getVisionUsage = () => request<VisionUsage>("/vision/usage");
