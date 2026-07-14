export interface Artwork {
  id: string;
  title: string;
  sourceImageUri: string;
  creatorId: string;
  ownerWalletAddress: string;
  protectionProfile: "L1_PREVIEW" | "L2_PORTFOLIO" | "L3_ANTI_TRAIN";
  allowAiTraining: boolean;
  status: "UPLOADED" | "PROTECTING" | "REGISTERING" | "PUBLISHED" | "FAILED";
  errorMessage: string | null;
  protectedImageUri: string | null;
  perceptualHash: string | null;
  metadataHash: string | null;
  createdAt: string;
  updatedAt: string;
  ownershipRecords: Array<{
    txHash: string;
    chain: string;
    registryAddress: string;
    registeredAt: string;
  }>;
}
