import { api } from "./client";

export interface RemeasureResult {
  styleDriftScore: number | null;
  styleSimilarityToOriginal: number | null;
  perceptualPsnrDb: number | null;
  perceptualRmse: number | null;
}

export const remeasureProtection = (artworkId: string) =>
  api.post<RemeasureResult>(`/artworks/${artworkId}/remeasure-protection`);
