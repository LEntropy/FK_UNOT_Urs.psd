import { api } from "./client";
import type { DetectionCase, EvidenceBundle } from "./types";

export const scanArtwork = (artworkId: string) =>
  api.post<{ caseId: string; status: string }>(`/artworks/${artworkId}/scan`);

export const reportArtwork = (artworkId: string, suspectUrl: string) =>
  api.post<{ caseId: string; status: string }>(`/artworks/${artworkId}/report`, { suspectUrl });

export const reportModelLeak = (artworkId: string, suspectModelUrl: string) =>
  api.post<{ caseId: string; status: string }>(`/artworks/${artworkId}/model-leak-report`, { suspectModelUrl });

export const getDetectionCase = (caseId: string) => api.get<DetectionCase>(`/detection-cases/${caseId}`);

export const getDetectionEvidence = (caseId: string) =>
  api.get<{ caseId: string; status: string; bundles: EvidenceBundle[] }>(`/detection-cases/${caseId}/evidence`);

export interface DmcaNoticeResult {
  caseId: string;
  notices: Array<{ sourceUrl: string | null; notice: string | null; note: string | null }>;
}

export const getDmcaNotice = (caseId: string) =>
  api.get<DmcaNoticeResult>(`/detection-cases/${caseId}/dmca-notice`);

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

export const verifyEvidence = (caseId: string) =>
  api.get<EvidenceVerifyResult>(`/detection-cases/${caseId}/verify`);

export const updateCaseStatus = (caseId: string, status: "NOTIFIED" | "RESOLVED" | "ESCALATED", note?: string) =>
  api.patch<DetectionCase>(`/detection-cases/${caseId}`, { status, note });

export interface VisionUsage {
  configured: boolean;
  month: string;
  used: number;
  limit: number;
  remaining: number;
}

export const getVisionUsage = () => api.get<VisionUsage>("/vision-usage");
