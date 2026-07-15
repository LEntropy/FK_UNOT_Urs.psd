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
