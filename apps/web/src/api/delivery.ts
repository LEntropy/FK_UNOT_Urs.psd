import { api } from "./client";

export const getRenderUrl = (artworkId: string, variant: "logged_in" | "thumbnail" = "logged_in") =>
  api.get<{ url: string }>(`/artworks/${artworkId}/render-url?variant=${variant}`);
