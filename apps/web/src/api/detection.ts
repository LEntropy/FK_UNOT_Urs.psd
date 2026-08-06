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
