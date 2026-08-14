/**
 * One-off admin cleanup (2026-08-14 incident follow-up): deletes the 40
 * artworks affected by the earlier disk-cleanup file-loss incident, per
 * the user's explicit decision to delete rather than recover them.
 * Bypasses the normal POST /:id/cancel HTTP route (creator-authenticated,
 * one artwork's own creator only) since this cleanup spans many different
 * creatorIds -- this is an operator action taken with the user's direct
 * authorization, not a per-user self-service delete.
 *
 * Mirrors routes/artworks.ts's own /:id/cancel delete branch (same four
 * tables), plus community-engagement tables that route doesn't touch
 * (likes/bookmarks/comments/reports) so nothing orphaned is left pointing
 * at a since-deleted artwork_id.
 *
 * Usage: npx tsx scripts/delete-affected-artworks.ts ast_xxx ast_yyy ...
 */
import { eq } from "drizzle-orm";
import { createDb } from "../src/db/client.js";
import {
  artworks,
  assetVersions,
  ownershipRecords,
  originalPreviewUnlocks,
  botPolicies,
  likes,
  bookmarks,
  comments,
  reports,
  scoreProtectionResults,
  pendingScoreProtectionJobs,
} from "../src/db/schema.js";
import { getObjectStorage } from "../src/storage/objectStorage.js";
import { env } from "../src/env.js";

async function deleteOne(db: ReturnType<typeof createDb>, artworkId: string): Promise<void> {
  const artwork = db.select().from(artworks).where(eq(artworks.id, artworkId)).get();
  if (!artwork) {
    console.log(`[delete] ${artworkId}: no such artwork, skipping`);
    return;
  }

  db.delete(assetVersions).where(eq(assetVersions.artworkId, artworkId)).run();
  db.delete(ownershipRecords).where(eq(ownershipRecords.artworkId, artworkId)).run();
  db.delete(originalPreviewUnlocks).where(eq(originalPreviewUnlocks.artworkId, artworkId)).run();
  db.delete(botPolicies).where(eq(botPolicies.artworkId, artworkId)).run();
  db.delete(likes).where(eq(likes.artworkId, artworkId)).run();
  db.delete(bookmarks).where(eq(bookmarks.artworkId, artworkId)).run();
  db.delete(comments).where(eq(comments.artworkId, artworkId)).run();
  db.delete(reports).where(eq(reports.artworkId, artworkId)).run();
  db.delete(scoreProtectionResults).where(eq(scoreProtectionResults.artworkId, artworkId)).run();
  db.delete(pendingScoreProtectionJobs).where(eq(pendingScoreProtectionJobs.artworkId, artworkId)).run();
  db.delete(artworks).where(eq(artworks.id, artworkId)).run();

  if (artwork.encryptedImagePath) {
    try {
      await getObjectStorage().delete(artwork.encryptedImagePath);
    } catch {
      // best-effort -- a leftover encrypted blob isn't worth failing this cleanup over
    }
  }

  console.log(`[delete] ${artworkId} (${artwork.title}): deleted`);
}

async function main() {
  const ids = process.argv.slice(2);
  if (ids.length === 0) {
    console.error("usage: npx tsx scripts/delete-affected-artworks.ts ast_xxx [ast_yyy ...]");
    process.exit(1);
  }

  const db = createDb(env.DATABASE_URL);
  for (const id of ids) {
    await deleteOne(db, id);
  }
}

main();
