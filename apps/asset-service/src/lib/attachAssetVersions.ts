import { inArray } from "drizzle-orm";
import type { Db } from "../db/client.js";
import { assetVersions } from "../db/schema.js";

/**
 * GET /artworks and GET /feed both used to return bare artwork rows with
 * no assetVersions -- only GET /artworks/:id joined them in. That's fine
 * for a detail page, but leaves list/feed views with no way to know
 * whether an artwork has a renderable image at all, so apps/web's
 * FeedPage and GalleryPage could never show a thumbnail (ArtworkImage
 * needs assetVersions.length > 0 to even ask delivery-gateway for a
 * signed URL). One batched query instead of an assetVersions lookup per
 * row (which is what a naive per-artwork query in a .map() would do).
 */
export function attachAssetVersions<T extends { id: string }>(
  db: Db,
  rows: T[],
): Array<T & { assetVersions: (typeof assetVersions.$inferSelect)[] }> {
  if (rows.length === 0) return [];

  const ids = rows.map((r) => r.id);
  const versions = db.select().from(assetVersions).where(inArray(assetVersions.artworkId, ids)).all();

  const byArtworkId = new Map<string, (typeof assetVersions.$inferSelect)[]>();
  for (const v of versions) {
    const list = byArtworkId.get(v.artworkId);
    if (list) list.push(v);
    else byArtworkId.set(v.artworkId, [v]);
  }

  return rows.map((r) => ({ ...r, assetVersions: byArtworkId.get(r.id) ?? [] }));
}
