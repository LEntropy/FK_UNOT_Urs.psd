import { eq, inArray } from "drizzle-orm";
import type { Db } from "./db/client.js";
import { artworks, assetVersions, ownershipRecords } from "./db/schema.js";
import { createProtectJob, pollProtectJob } from "./clients/protectionSvc.js";
import { registerAsset, verifyAsset, AlreadyRegisteredError } from "./clients/blockchainSvc.js";
import { decryptToTempFile, cleanupTempFile } from "./crypto/imageEncryption.js";
import { env } from "./env.js";

/**
 * The upload pipeline (PROJECT_DESIGN.md §1-1), implemented as the state
 * machine the `artworks.status` column tracks:
 *
 *   UPLOADED -> PROTECTING -> REGISTERING -> PUBLISHED
 *                    \-> FAILED (from either step, with errorMessage set)
 *
 * Called fire-and-forget from routes/artworks.ts's POST handler (not
 * awaited before responding) -- same reasoning as protection-svc's own
 * job design: this can run from ~1 minute to hours depending on protection
 * profile/size (ml-engine/README.md), so it cannot live inside a
 * request/response cycle. GET /artworks/:id is how callers observe
 * progress, same shape as protection-svc's GET /protect/:jobId pattern.
 */
export async function runUploadPipeline(db: Db, artworkId: string): Promise<void> {
  const artwork = db.select().from(artworks).where(eq(artworks.id, artworkId)).get();
  if (!artwork) throw new Error(`runUploadPipeline: no artwork ${artworkId}`);

  let decryptedTempPath: string | undefined;

  try {
    await setStatus(db, artworkId, "PROTECTING");

    // Decrypted only for the duration of this job -- protection-svc needs
    // a real local file path (INTEGRATION.md's imageUri contract), not
    // bytes in memory. Needs the live KMS server (unwrapKey), unlike the
    // encrypt-at-upload-time path in routes/artworks.ts.
    decryptedTempPath = await decryptToTempFile(
      {
        encryptedImagePath: artwork.encryptedImagePath,
        encryptedDekBase64: artwork.encryptedDekBase64,
        encryptionIv: artwork.encryptionIv,
        encryptionAuthTag: artwork.encryptionAuthTag,
      },
      artworkId,
    );

    const { jobId } = await createProtectJob({
      imageUri: decryptedTempPath,
      protectionProfile: artwork.protectionProfile,
      title: artwork.title,
      creatorId: artwork.creatorId,
      allowAiTraining: artwork.allowAiTraining,
      watermarkPayloadHex: artwork.watermarkPayloadHex,
    });

    db.update(artworks).set({ protectJobId: jobId, updatedAt: new Date() }).where(eq(artworks.id, artworkId)).run();

    const job = await pollProtectJob(jobId);

    if (job.status === "failed" || !job.perceptualHash || !job.metadataHash || !job.protectedImageUri) {
      await setStatus(db, artworkId, "FAILED", job.error ?? "protection-svc job failed without an error message");
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
        updatedAt: new Date(),
      })
      .where(eq(artworks.id, artworkId))
      .run();

    for (const variant of job.variants ?? []) {
      db.insert(assetVersions)
        .values({
          artworkId,
          variantName: variant.name,
          storageUri: job.protectedImageUri, // rust-core variants aren't uploaded anywhere separate yet -- see README
          width: variant.width,
          height: variant.height,
          scaleVsSource: variant.scaleVsSource,
          protectionStatus: variant.protectionStatus,
        })
        .run();
    }

    await runRegistrationStep(db, artworkId, artwork.ownerWalletAddress, job.perceptualHash, job.metadataHash, artwork.allowAiTraining);
  } catch (err) {
    await setStatus(db, artworkId, "FAILED", err instanceof Error ? err.message : String(err));
  } finally {
    // Decrypted plaintext only ever exists for the duration of this job --
    // clean it up regardless of success/failure, not just on the happy path.
    if (decryptedTempPath) cleanupTempFile(decryptedTempPath);
  }
}

/**
 * Extracted from runUploadPipeline so recoverInterruptedUploads() can
 * resume an artwork stuck at REGISTERING without re-running protection --
 * by the time a row reaches REGISTERING, perceptualHash/metadataHash are
 * already persisted on it (set just before this is first called), so
 * there's nothing to recompute, only the on-chain call to retry.
 */
async function runRegistrationStep(
  db: Db,
  artworkId: string,
  ownerWalletAddress: string,
  perceptualHash: string,
  metadataHash: string,
  allowAiTraining: boolean,
): Promise<void> {
  await setStatus(db, artworkId, "REGISTERING");

  try {
    const registration = await registerAsset({
      ownerAddress: ownerWalletAddress,
      perceptualHash,
      metadataHash,
      doNotTrain: !allowAiTraining,
    });

    db.insert(ownershipRecords)
      .values({
        artworkId,
        ownerWallet: registration.ownerAddress,
        contentHash: registration.contentHash,
        chain: env.CHAIN_NAME,
        registryAddress: env.REGISTRY_ADDRESS,
        txHash: registration.txHash,
        blockNumber: registration.blockNumber,
        registeredAt: new Date(),
      })
      .run();

    await setStatus(db, artworkId, "PUBLISHED");
  } catch (err) {
    if (err instanceof AlreadyRegisteredError) {
      // apps/blockchain-svc/INTEGRATION.md's documented 409 handling,
      // actually implemented here: re-verify on-chain ownership. Same
      // owner => idempotent (e.g. a retried request), not a failure.
      // Different owner => a genuine hash collision, worth flagging
      // rather than silently overwriting.
      const onChain = await verifyAsset(err.contentHash);
      if (onChain.exists && onChain.owner?.toLowerCase() === ownerWalletAddress.toLowerCase()) {
        db.insert(ownershipRecords)
          .values({
            artworkId,
            ownerWallet: onChain.owner,
            contentHash: err.contentHash,
            chain: env.CHAIN_NAME,
            registryAddress: env.REGISTRY_ADDRESS,
            txHash: "(pre-existing registration, no new tx)",
            blockNumber: 0,
            registeredAt: onChain.timestamp ? new Date(onChain.timestamp * 1000) : new Date(),
          })
          .run();
        await setStatus(db, artworkId, "PUBLISHED");
      } else {
        await setStatus(
          db,
          artworkId,
          "FAILED",
          `content hash collision: ${err.contentHash} is already registered to a different owner (${onChain.owner}), not this artwork's owner (${ownerWalletAddress})`,
        );
      }
    } else {
      throw err;
    }
  }
}

/**
 * Called once at asset-service startup (index.ts) -- the resumability fix.
 * Any artwork left at PROTECTING or REGISTERING when this process starts
 * was, by definition, interrupted by a previous process's death (a fresh
 * process has issued no POST /artworks yet, so nothing can legitimately be
 * mid-flight already).
 *
 * REGISTERING is safely resumable: protection already finished and its
 * hashes are already persisted on the row, so this just retries the
 * on-chain call (registerAsset/verifyAsset are already idempotent-safe,
 * see runRegistrationStep's AlreadyRegisteredError handling).
 *
 * PROTECTING is not resumed, only failed cleanly: there's no reliable way
 * to know whether protection-svc's job for it is still running, already
 * finished, or itself died (protection-svc's own restart-recovery,
 * jobs_db.py, would have already marked it failed if protection-svc
 * itself restarted -- but asset-service has no way to distinguish that
 * from "protection-svc is still fine and still working on it" without
 * risking a duplicate protect job). Marking it failed with an honest
 * message and asking for a re-upload is the safe choice over silently
 * leaving it stuck forever or guessing.
 */
export async function recoverInterruptedUploads(db: Db): Promise<void> {
  const stuck = db.select().from(artworks).where(inArray(artworks.status, ["PROTECTING", "REGISTERING"])).all();

  for (const artwork of stuck) {
    if (artwork.status === "REGISTERING" && artwork.perceptualHash && artwork.metadataHash) {
      console.log(`[recovery] resuming REGISTERING for ${artwork.id} after a restart`);
      // Awaited (unlike runUploadPipeline's own fire-and-forget dispatch
      // from the request handler) -- this runs once at startup, not inside
      // an HTTP request cycle, so there's no latency budget forcing this
      // to return quickly. Awaiting sequentially also means startup
      // finishes only once every stuck artwork has actually been resolved
      // one way or the other, not left racing in the background.
      await runRegistrationStep(
        db,
        artwork.id,
        artwork.ownerWalletAddress,
        artwork.perceptualHash,
        artwork.metadataHash,
        artwork.allowAiTraining,
      ).catch((err) => setStatus(db, artwork.id, "FAILED", err instanceof Error ? err.message : String(err)));
    } else {
      console.log(`[recovery] marking ${artwork.id} failed -- interrupted mid-${artwork.status} by a restart`);
      await setStatus(
        db,
        artwork.id,
        "FAILED",
        `interrupted by an asset-service restart while ${artwork.status.toLowerCase()} -- please re-upload`,
      );
    }
  }
}

async function setStatus(db: Db, artworkId: string, status: string, errorMessage?: string): Promise<void> {
  const now = new Date();
  db.update(artworks)
    .set({
      status,
      errorMessage: errorMessage ?? null,
      updatedAt: now,
      // Set once -- a later status churn (there isn't one today, but
      // nothing stops a future re-protect flow) shouldn't bump this back
      // to the top of the "latest" feed.
      ...(status === "PUBLISHED" ? { publishedAt: now } : {}),
    })
    .where(eq(artworks.id, artworkId))
    .run();
}
