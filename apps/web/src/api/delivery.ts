import { api } from "./client";

export const getRenderUrl = (artworkId: string, variant: "logged_in" | "thumbnail" | "original" = "logged_in") =>
  api.get<{ url: string }>(`/artworks/${artworkId}/render-url?variant=${variant}`);

/** Batched counterpart used by feed/gallery grids -- one request for every
 * thumbnail on screen instead of one per card (see api-gateway's own route
 * doc for why that mattered). Missing/failed entries come back as null. */
export const getRenderUrlsBatch = (ids: string[], variant: "logged_in" | "thumbnail" = "thumbnail") =>
  api.post<Record<string, string | null>>("/artworks/render-urls", { ids, variant });
