import { randomBytes, randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { Router } from "express";
import multer from "multer";
import { eq } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import { artworks, assetVersions, ownershipRecords } from "../db/schema.js";
import { runUploadPipeline } from "../orchestration.js";
import { encryptImageAtRest } from "../crypto/imageEncryption.js";
import { env } from "../env.js";
import { attachAssetVersions } from "../lib/attachAssetVersions.js";

const createArtworkSchema = z.object({
  title: z.string().min(1),
  // Optional here even though the row always ends up with one -- a real
  // browser upload (multipart, req.file below) supplies image bytes
  // instead of a path; a server-side/script caller (existing tests,
  // detection-svc smoke checks, curl in this README) still supplies a
  // path this process can read directly. The route handler below picks
  // whichever one the request actually gave it.
  sourceImageUri: z.string().min(1).optional(),
  creatorId: z.string().min(1),
  ownerWalletAddress: z.string().regex(/^0x[0-9a-fA-F]{40}$/, "must be a 20-byte hex address"),
  protectionProfile: z.enum(["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"]).default("L3_ANTI_TRAIN"),
  // multipart/form-data fields arrive as strings, never real booleans, so
  // this needs to accept both shapes -- a real boolean from a JSON body,
  // or "true"/"false" from a multipart field. Deliberately NOT
  // z.coerce.boolean(): that's just Boolean(value) under the hood, and
  // Boolean("false") is true (any non-empty string is truthy in JS) --
  // a real bug, not a hypothetical one, caught live when a genuine
  // S3_USE_SSL=false env value hit the identical mistake in env.ts.
  allowAiTraining: z
    .union([z.boolean(), z.enum(["true", "false"])])
    .default(false)
    .transform((v) => v === true || v === "true"),
});

// multer's own disk storage, not os.tmpdir() -- same reasoning as
// DECRYPT_TEMP_DIR (src/crypto/imageEncryption.ts, env.ts's doc comment):
// this process's own ./data volume, not wherever the OS temp dir happens
// to live. encryptImageAtRest() deletes this file right after encrypting
// it either way, so nothing further needs to clean it up.
const upload = multer({
  storage: multer.diskStorage({
    // multer's diskStorage does NOT create this directory itself (unlike
    // encryptImageAtRest's own mkdirSync elsewhere in this codebase) --
    // it just fails if it's missing, so create it once up front.
    destination: (_req, _file, cb) => {
      const dir = resolve(env.UPLOAD_TEMP_DIR);
      mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename: (_req, file, cb) => cb(null, `upload-${randomUUID()}-${file.originalname}`),
  }),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB -- generous for a single artwork image, not unbounded
});

export function artworksRouter(db: Db): Router {
  const router = Router();

  router.post("/", upload.single("image"), async (req, res) => {
    const parsed = createArtworkSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    // A real browser upload (req.file, from multer) always wins over a
    // caller-given path when both are somehow present -- the uploaded
    // bytes are the actual thing the user picked; a stale sourceImageUri
    // field alongside it would be surprising to honor instead.
    const sourceImageUri = req.file?.path ?? parsed.data.sourceImageUri;
    if (!sourceImageUri) {
      return res.status(400).json({ error: "either upload an image file or provide sourceImageUri" });
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
      encrypted = await encryptImageAtRest(sourceImageUri, id);
    } catch (err) {
      return res.status(400).json({
        error: `could not read uploaded image (${JSON.stringify(sourceImageUri)}): ${err instanceof Error ? err.message : String(err)}`,
      });
    }

    db.insert(artworks)
      .values({
        id,
        title: parsed.data.title,
        // The real filesystem path for either case (multer's temp path, or
        // a caller-given local path) is meaningless to keep -- both get
        // deleted by encryptImageAtRest right above. Store something a
        // human reading the row later can actually make sense of.
        sourceImageUri: req.file ? `upload:${req.file.originalname}` : parsed.data.sourceImageUri!,
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

    // Without this, GalleryPage/FeedPage have no way to know whether an
    // artwork has a renderable image yet -- ArtworkImage needs
    // assetVersions.length > 0 before it'll even ask delivery-gateway for
    // a signed URL. GET /artworks/:id already joined this in; the list
    // route never did.
    res.json(attachAssetVersions(db, rows));
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
