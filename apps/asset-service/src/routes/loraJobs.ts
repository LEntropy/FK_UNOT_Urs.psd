import { randomUUID } from "node:crypto";
import { Router } from "express";
import { eq } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import { artworks, loraGenerationJobs } from "../db/schema.js";
import { decryptToTempFile, cleanupTempFile } from "../crypto/imageEncryption.js";
import { createLoraJob, pollLoraJob } from "../clients/protectionSvc.js";
import { COIN_COSTS, InsufficientCoinsError, spendCoins } from "../coins.js";

/**
 * Coin-system feature (2026-08-10): spend coins to train a real,
 * downloadable SD1.5 LoRA on one of the user's own artworks (see
 * protection-svc's ml-engine/src/lora_generate.py for the mechanism).
 * Mounted at root, not under /artworks -- these rows aren't scoped to a
 * single artwork's own sub-resources the way remeasure-protection/
 * score-protection are (source_artwork_id is just a foreign reference, not
 * an owning path segment), and asset-service has no auth of its own either
 * way (api-gateway verifies the JWT and checks sourceArtworkId ownership
 * before ever calling here, same trust boundary as every other route in
 * this service).
 */
export function loraJobsRouter(db: Db): Router {
  const router = Router();

  const createLoraJobSchema = z.object({
    sourceArtworkId: z.string().min(1),
    userId: z.string().min(1),
  });

  router.post("/lora-jobs", async (req, res) => {
    const parsed = createLoraJobSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const { sourceArtworkId, userId } = parsed.data;

    const artwork = db.select().from(artworks).where(eq(artworks.id, sourceArtworkId)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${sourceArtworkId}` });
    }

    const id = `lora_${randomUUID().replace(/-/g, "").slice(0, 16)}`;

    try {
      spendCoins(db, userId, COIN_COSTS.LORA_GENERATION, "lora_generation_spend", sourceArtworkId);
    } catch (err) {
      if (err instanceof InsufficientCoinsError) {
        return res.status(402).json({ error: "insufficient_coins", required: err.required, balance: err.balance });
      }
      throw err;
    }

    const now = new Date();
    db.insert(loraGenerationJobs)
      .values({
        id,
        userId,
        sourceArtworkId,
        status: "QUEUED",
        coinCost: COIN_COSTS.LORA_GENERATION,
        createdAt: now,
        updatedAt: now,
      })
      .run();

    let decryptedTempPath: string;
    try {
      decryptedTempPath = await decryptToTempFile(
        {
          encryptedImagePath: artwork.encryptedImagePath,
          encryptedDekBase64: artwork.encryptedDekBase64,
          encryptionIv: artwork.encryptionIv,
          encryptionAuthTag: artwork.encryptionAuthTag,
        },
        artwork.id,
      );
      const { jobId: protectionJobId } = await createLoraJob({ imageUri: decryptedTempPath, prompt: artwork.title });

      db.update(loraGenerationJobs).set({ status: "RUNNING", updatedAt: new Date() }).where(eq(loraGenerationJobs.id, id)).run();

      // Fire-and-forget: same pattern as routes/artworks.ts's own
      // score-protection handler -- the decrypted temp file must outlive
      // this HTTP response (protection-svc's background job reads it well
      // after this request returns), so cleanup happens once the poll
      // resolves, not in a `finally` here.
      void pollLoraJob(protectionJobId)
        .then((job) => {
          if (job.status === "failed" || !job.outputPath) {
            db.update(loraGenerationJobs)
              .set({ status: "FAILED", errorMessage: job.error ?? "protection-svc job failed without an error message", updatedAt: new Date() })
              .where(eq(loraGenerationJobs.id, id))
              .run();
          } else {
            db.update(loraGenerationJobs)
              .set({ status: "COMPLETED", resultPath: job.outputPath, updatedAt: new Date() })
              .where(eq(loraGenerationJobs.id, id))
              .run();
          }
        })
        .catch((err) => {
          db.update(loraGenerationJobs)
            .set({ status: "FAILED", errorMessage: err instanceof Error ? err.message : String(err), updatedAt: new Date() })
            .where(eq(loraGenerationJobs.id, id))
            .run();
        })
        .finally(() => cleanupTempFile(decryptedTempPath));

      res.status(202).json({ id });
    } catch (err) {
      db.update(loraGenerationJobs)
        .set({ status: "FAILED", errorMessage: err instanceof Error ? err.message : String(err), updatedAt: new Date() })
        .where(eq(loraGenerationJobs.id, id))
        .run();
      res.status(502).json({ error: `lora generation failed to start: ${err instanceof Error ? err.message : String(err)}` });
    }
  });

  router.get("/lora-jobs/:id", (req, res) => {
    const userId = typeof req.query.userId === "string" ? req.query.userId : undefined;
    const job = db.select().from(loraGenerationJobs).where(eq(loraGenerationJobs.id, req.params.id)).get();
    if (!job) {
      return res.status(404).json({ error: `no lora job ${req.params.id}` });
    }
    if (!userId || userId !== job.userId) {
      return res.status(403).json({ error: "not this job's owner" });
    }
    res.json(job);
  });

  router.get("/lora-jobs/:id/download", (req, res) => {
    const userId = typeof req.query.userId === "string" ? req.query.userId : undefined;
    const job = db.select().from(loraGenerationJobs).where(eq(loraGenerationJobs.id, req.params.id)).get();
    if (!job) {
      return res.status(404).json({ error: `no lora job ${req.params.id}` });
    }
    if (!userId || userId !== job.userId) {
      return res.status(403).json({ error: "not this job's owner" });
    }
    if (job.status !== "COMPLETED" || !job.resultPath) {
      return res.status(400).json({ error: "job is not completed yet" });
    }
    res.download(job.resultPath, `${job.id}.safetensors`);
  });

  return router;
}
