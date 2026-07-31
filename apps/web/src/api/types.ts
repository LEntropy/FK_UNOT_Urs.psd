export interface AssetVersion {
  variantName: string;
  storageUri: string;
  width: number;
  height: number;
  scaleVsSource: number;
  protectionStatus: "SAFE" | "UNKNOWN" | "UNSAFE";
}

export interface Artwork {
  id: string;
  title: string;
  sourceImageUri: string;
  creatorId: string;
  ownerWalletAddress: string;
  protectionProfile: "L1_PREVIEW" | "L2_PORTFOLIO" | "L3_ANTI_TRAIN";
  allowAiTraining: boolean;
  visibility: "public" | "followers" | "private";
  status: "UPLOADED" | "PROTECTING" | "REGISTERING" | "PUBLISHED" | "FAILED";
  errorMessage: string | null;
  protectedImageUri: string | null;
  perceptualHash: string | null;
  metadataHash: string | null;
  // Real, per-upload measurements (protection-svc's evaluate.py) -- null
  // when protection-svc skipped the measurement, not a claim of zero effect.
  styleDriftScore: number | null;
  styleSimilarityToOriginal: number | null;
  perceptualPsnrDb: number | null;
  publishedAt: string | null;
  createdAt: string;
  updatedAt: string;
  assetVersions: AssetVersion[];
  ownershipRecords: Array<{
    txHash: string;
    chain: string;
    registryAddress: string;
    registeredAt: string;
  }>;
}

export interface Comment {
  id: string;
  artworkId: string;
  userId: string;
  body: string;
  createdAt: string;
}

export interface Report {
  id: string;
  reporterId: string;
  artworkId: string;
  reason: string;
  status: "PENDING" | "RESOLVED" | "DISMISSED";
  createdAt: string;
}

export type FeedType = "latest" | "popular" | "following";

export interface DetectionCase {
  id: string;
  artwork_id: string;
  status: "OPEN" | "EVIDENCE_READY" | "NO_MATCH_FOUND" | "FAILED" | "NOTIFIED" | "RESOLVED" | "ESCALATED";
  trigger: "scan" | "report";
  error_message: string | null;
  note: string | null;
  created_at: number;
  updated_at: number;
  evidence: Array<{
    id: number;
    case_id: string;
    evidence_type: string;
    source_url: string | null;
    confidence: number | null;
    artifact_uri: string | null;
    detected_at: number;
  }>;
}

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
}
