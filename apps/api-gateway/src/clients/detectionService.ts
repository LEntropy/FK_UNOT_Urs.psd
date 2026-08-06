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
  status: "OPEN" | "EVIDENCE_READY" | "NO_MATCH_FOUND" | "FAILED" | "NOTIFIED" | "RESOLVED" | "ESCALATED";
  trigger: "scan" | "report" | "model_report";
  error_message: string | null;
  note: string | null;
  created_at: number;
  updated_at: number;
  evidence: EvidenceRecord[];
}

export const getCase = (caseId: string) => request<Case>(`/cases/${encodeURIComponent(caseId)}`);

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
