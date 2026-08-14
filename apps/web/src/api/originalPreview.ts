import { api } from "./client";

// 2026-08-14 redesign: the coin-unlock button is available on every
// published artwork by default -- no more separate creator "enable" step.
// This is now purely an opt-OUT for a creator who never wants their
// original unlockable at any coin price.
export const setOriginalPreviewBlocked = (artworkId: string, blocked: boolean) =>
  api.put<{ originalPreviewBlocked: boolean }>(`/artworks/${artworkId}/original-preview-blocked`, { blocked });

export const unlockOriginalPreview = (artworkId: string) =>
  api.post<{ originalPreviewUnlockedByViewer: boolean }>(`/artworks/${artworkId}/original-preview/unlock`);
