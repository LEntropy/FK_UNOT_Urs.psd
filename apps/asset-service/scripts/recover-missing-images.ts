/**
 * One-off recovery script (2026-08-14 incident): a disk-cleanup pass
 * earlier this session deleted old `protection-svc/out/job_<id>` directories
 * (anything older than 7 days), believing them to be disposable job scratch space.
 * For artworks published before storage moved off that path, those
 * directories were actually the PERMANENT location `protectedImageUri`
 * and every `asset_versions.storageUri` pointed at -- deleting them left
 * 40 published artworks with no servable image (delivery-gateway's own
 * fallback logic correctly found *a* variant row, but the file it pointed
 * to no longer existed on disk: "variant file missing on disk").
 *
 * This re-derives a fresh protected image + variants from each affected
 * artwork's still-intact encrypted original (only run against artworks
 * confirmed to have one -- see the accompanying investigation) using the
 * exact same protectionProfile/watermarkPayloadHex/title/creatorId the
 * artwork was originally uploaded with, then overwrites the artwork's
 * image-related columns and asset_versions rows with the new result.
 *
 * Deliberately does NOT touch status/ownershipRecords -- these artworks
 * are already PUBLISHED with an on-chain registration. Re-running
 * protection is not guaranteed to reproduce the exact original
 * perceptualHash (most of this project's cloak mechanisms aren't
 * bit-for-bit deterministic across runs), so the on-chain record's own
 * contentHash will no longer match this new image's perceptualHash for
 * these specific artworks. That mismatch is an unavoidable, honestly-
 * accepted consequence of the underlying file loss, not something this
 * script tries to paper over by re-registering (which would either spend
 * more gas for a hash the chain already has under a different value, or
 * require deciding whether to treat the old on-chain record as void --
 * neither is this script's call to make silently).
 *
 * Usage (run from apps/asset-service, same environment/.env the live
 * service itself runs under -- needs a real KMS connection to unwrap each
 * artwork's DEK):
 *   npx tsx scripts/recover-missing-images.ts ast_xxx ast_yyy ...
 */
import { eq } from "drizzle-orm";
import { createDb } from "../src/db/client.js";
import { artworks, assetVersions } from "../src/db/schema.js";
import { env } from "../src/env.js";
import { decryptToTempFile, cleanupTempFile } from "../src/crypto/imageEncryption.js";
import { createProtectJob, pollProtectJob } from "../src/clients/protectionSvc.js";

async function recoverOne(db: ReturnType<typeof createDb>, artworkId: string): Promise<void> {
  const artwork = db.select().from(artworks).where(eq(artworks.id, artworkId)).get();
  if (!artwork) {
    console.log(`[recover] ${artworkId}: no such artwork, skipping`);
    return;
  }

  let decryptedTempPath: string | undefined;
  try {
    console.log(`[recover] ${artworkId} (${artwork.title}): decrypting original...`);
    decryptedTempPath = await decryptToTempFile(
      {
        encryptedImagePath: artwork.encryptedImagePath,
        encryptedDekBase64: artwork.encryptedDekBase64,
        encryptionIv: artwork.encryptionIv,
        encryptionAuthTag: artwork.encryptionAuthTag,
      },
      artworkId,
    );

    // None of the affected artworks used STRONG_PROTECTION (confirmed by
    // the investigation this script follows up on) -- no RunPod Serverless
    // cost involved here, same style_cloak path every L1/L2/L3 upload uses.
    console.log(`[recover] ${artworkId}: submitting protect job (${artwork.protectionProfile})...`);
    const { jobId } = await createProtectJob({
      imageUri: decryptedTempPath,
      protectionProfile: artwork.protectionProfile,
      title: artwork.title,
      creatorId: artwork.creatorId,
      allowAiTraining: artwork.allowAiTraining,
      watermarkPayloadHex: artwork.watermarkPayloadHex,
      strongProtection: false,
    });

    const job = await pollProtectJob(jobId, { timeoutMs: 20 * 60 * 1000 });
    if (job.status === "failed" || !job.perceptualHash || !job.metadataHash || !job.protectedImageUri) {
      console.error(`[recover] ${artworkId}: protect job failed -- ${job.error ?? "no error message"}`);
      return;
    }

    db.update(artworks)
      .set({
        protectedImageUri: job.protectedImageUri,
        perceptualHash: job.perceptualHash,
        metadataHash: job.metadataHash,
        styleDriftScore: job.styleDriftScore ?? null,
        styleSimilarityToOriginal: job.styleSimilarityToOriginal ?? null,
        perceptualPsnrDb: job.perceptualPsnrDb ?? null,
        usedStrongProtection: job.usedStrongProtection ?? false,
        updatedAt: new Date(),
      })
      .where(eq(artworks.id, artworkId))
      .run();

    // Old rows point at files that no longer exist -- replace outright
    // rather than append, so nothing stale lingers alongside the new set.
    db.delete(assetVersions).where(eq(assetVersions.artworkId, artworkId)).run();
    for (const variant of job.variants ?? []) {
      db.insert(assetVersions)
        .values({
          artworkId,
          variantName: variant.name,
          storageUri: variant.path ?? job.protectedImageUri,
          width: variant.width,
          height: variant.height,
          scaleVsSource: variant.scaleVsSource,
          protectionStatus: variant.protectionStatus,
        })
        .run();
    }

    console.log(`[recover] ${artworkId}: done, ${(job.variants ?? []).length} variant(s) restored`);
  } catch (err) {
    console.error(`[recover] ${artworkId}: ERROR -- ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    if (decryptedTempPath) cleanupTempFile(decryptedTempPath);
  }
}

async function main() {
  const ids = process.argv.slice(2);
  if (ids.length === 0) {
    console.error("usage: npx tsx scripts/recover-missing-images.ts ast_xxx [ast_yyy ...]");
    process.exit(1);
  }

  const db = createDb(env.DATABASE_URL);
  for (const id of ids) {
    await recoverOne(db, id);
  }
}

main();
