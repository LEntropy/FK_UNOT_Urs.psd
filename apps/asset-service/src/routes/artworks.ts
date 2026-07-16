import { randomBytes, randomUUID } from "node:crypto";
import { Router } from "express";
import { eq } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import { artworks, assetVersions, ownershipRecords } from "../db/schema.js";
import { runUploadPipeline } from "../orchestration.js";
import { encryptImageAtRest } from "../crypto/imageEncryption.js";

const createArtworkSchema = z.object({
  title: z.string().min(1),
  sourceImageUri: z.string().min(1),
  creatorId: z.string().min(1),
  ownerWalletAddress: z.string().regex(/^0x[0-9a-fA-F]{40}$/, "must be a 20-byte hex address"),
  protectionProfile: z.enum(["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"]).default("L3_ANTI_TRAIN"),
  allowAiTraining: z.boolean().default(false),
});

export function artworksRouter(db: Db): Router {
  const router = Router();

  router.post("/", async (req, res) => {
    const parsed = createArtworkSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }

    const id = `ast_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const now = new Date();
    // Per-artwork, generated once here (not by protection-svc) so it's
    // stable and known before the protect job even starts -- detection-svc
    // needs to read the same value back later via GET /artworks/:id.
    const watermarkPayloadHex = randomBytes(8).toString("hex");

    // Envelope-encrypts and deletes the plaintext upload -- client-side
    // only (wrapKey), no live KMS server needed for this step (see
    // src/crypto/imageEncryption.ts). Done before the row exists so a
    // request that can't even read its own upload never creates one.
    let encrypted;
    try {
      encrypted = await encryptImageAtRest(parsed.data.sourceImageUri, id);
    } catch (err) {
      return res.status(400).json({
        error: `could not read sourceImageUri ${JSON.stringify(parsed.data.sourceImageUri)}: ${err instanceof Error ? err.message : String(err)}`,
      });
    }

    db.insert(artworks)
      .values({
        id,
        title: parsed.data.title,
        sourceImageUri: parsed.data.sourceImageUri,
        creatorId: parsed.data.creatorId,
        ownerWalletAddress: parsed.data.ownerWalletAddress,
        protectionProfile: parsed.data.protectionProfile,
        allowAiTraining: parsed.data.allowAiTraining,
        watermarkPayloadHex,
        encryptedImagePath: encrypted.encryptedImagePath,
        encryptedDekBase64: encrypted.encryptedDekBase64,
        encryptionIv: encrypted.encryptionIv,
        encryptionAuthTag: encrypted.encryptionAuthTag,
        status: "UPLOADED",
        createdAt: now,
        updatedAt: now,
      })
      .run();

    // Fire-and-forget: see orchestration.ts's module doc for why this isn't
    // awaited here. Errors inside are caught and recorded as status=FAILED
    // on the row itself, not thrown here.
    void runUploadPipeline(db, id);

    res.status(202).json({ id, status: "UPLOADED" });
  });

  router.get("/", (req, res) => {
    const creatorId = typeof req.query.creatorId === "string" ? req.query.creatorId : undefined;

    const rows = creatorId
      ? db.select().from(artworks).where(eq(artworks.creatorId, creatorId)).all()
      : db.select().from(artworks).all();

    res.json(rows);
  });

  router.get("/:id", (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }

    const versions = db.select().from(assetVersions).where(eq(assetVersions.artworkId, artwork.id)).all();
    const ownership = db.select().from(ownershipRecords).where(eq(ownershipRecords.artworkId, artwork.id)).all();

    res.json({ ...artwork, assetVersions: versions, ownershipRecords: ownership });
  });

  return router;
}
