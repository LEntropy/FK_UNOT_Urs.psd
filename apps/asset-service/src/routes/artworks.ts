import { randomBytes, randomUUID } from "node:crypto";
import { mkdirSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import { Router } from "express";
import multer from "multer";
import { and, desc, eq } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import {
  artworks,
  assetVersions,
  ownershipRecords,
  originalPreviewUnlocks,
  scoreProtectionResults,
  pendingScoreProtectionJobs,
  botPolicies,
  botAccessLogs,
} from "../db/schema.js";
import { runUploadPipeline, persistScoreProtectionResult } from "../orchestration.js";
import { encryptImageAtRest, decryptToTempFile, cleanupTempFile } from "../crypto/imageEncryption.js";
import { getObjectStorage } from "../storage/objectStorage.js";
import {
  suggestTags,
  remeasureProtection,
  createScoreProtectionJob,
  getScoreProtectionJob,
  createOriginalPreview,
  cancelProtectJob,
} from "../clients/protectionSvc.js";
import { env } from "../env.js";
import { attachAssetVersions } from "../lib/attachAssetVersions.js";
import { logComplianceAudit } from "../lib/auditLog.js";
import { COIN_COSTS, InsufficientCoinsError, spendCoins } from "../coins.js";

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
  // STRONG_PROTECTION dispatches to protection-svc's dual-arch (SD1.5+SDXL)
  // RunPod Serverless attack (orchestration.ts) instead of style_cloak --
  // the only tier with a real, statistically-validated protection effect
  // (n=30-plus-replication, PHASE4_SCOPING.md §6). Real GPU cost/time per
  // upload (minutes, not the ~seconds-to-a-minute the other three tiers
  // take), which is why it's a fourth, separately-priced-in-spirit option
  // rather than folded into L3_ANTI_TRAIN's existing meaning.
  protectionProfile: z.enum(["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN", "STRONG_PROTECTION"]).default("L3_ANTI_TRAIN"),
  // Advanced-options upload feature (2026-08-08) -- opt-in, only
  // meaningful when protectionProfile is STRONG_PROTECTION (silently
  // ignored otherwise, same as protection-svc's own handling). Bounds
  // (0, 0.5] match hybrid_protect.py's own real range: 0 would be no
  // protection at all, values much above what this project has ever
  // measured (0.3) risk the exact "unrecognizable" failure this feature
  // exists to let a user step back from, not just forward into.
  // z.coerce.number() (not a union like allowAiTraining above) is safe
  // here specifically because these are never boolean-shaped -- no
  // "coerces falsy-looking strings to true" trap to avoid.
  strongProtectionLatentEpsilon: z.coerce.number().min(0).max(0.5).optional(),
  strongProtectionPixelEpsilon: z.coerce.number().min(0).max(0.5).optional(),
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
  // User-confirmed tags (see suggest-tags route below for where the
  // frontend's initial suggestions come from) -- a real array from a JSON
  // body, or a JSON-encoded string from a multipart field (multipart has
  // no native array type, same reasoning as allowAiTraining's string
  // variant above). Invalid JSON in the multipart case is treated as "no
  // tags" rather than a 400 -- a malformed tags field shouldn't block an
  // otherwise-valid upload for a nice-to-have feature.
  tags: z
    .union([z.array(z.string()), z.string()])
    .default([])
    .transform((v) => {
      if (Array.isArray(v)) return v;
      try {
        const parsed = JSON.parse(v);
        return Array.isArray(parsed) ? parsed.filter((t): t is string => typeof t === "string") : [];
      } catch {
        return [];
      }
    }),
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

// Mirrors delivery-gateway's BotAction/BotGroup (src/bot_policy.rs,
// src/crawlers.rs) exactly -- SCREAMING_SNAKE_CASE actions, snake_case
// group keys -- since this JSON round-trips straight to/from that Rust
// process's own serde(rename_all) shapes with no translation layer.
const botActionSchema = z.enum(["ALLOW", "BLOCK", "LOG_ONLY"]);
const botPolicyBodySchema = z.object({
  defaultAction: botActionSchema,
  groupPolicies: z.record(z.string(), botActionSchema).default({}),
  botOverrides: z.record(z.string(), botActionSchema).default({}),
});

const botAccessLogBodySchema = z.object({
  artworkId: z.string().min(1),
  botName: z.string().min(1),
  botGroup: z.string().min(1),
  action: botActionSchema,
  clientIp: z.string().min(1),
  userAgent: z.string(),
  responseStatus: z.number().int(),
  // Unix seconds (matches bot_policy.rs's BotAccessLog.timestamp, a
  // SystemTime::now().duration_since(UNIX_EPOCH).as_secs()) -- converted to
  // a Date only here at the DB boundary, not carried as one over the wire.
  timestampUnixSeconds: z.number().int(),
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

    // Coin gate, before any encryption/DB work -- STRONG_PROTECTION is the
    // one tier that costs coins (coins.ts's COIN_COSTS). Checked this early
    // so an unaffordable upload never touches encryptImageAtRest or leaves
    // an artwork row behind; the multer temp file is cleaned up the same
    // way suggest-tags's route does below.
    if (parsed.data.protectionProfile === "STRONG_PROTECTION") {
      try {
        spendCoins(db, parsed.data.creatorId, COIN_COSTS.STRONG_PROTECTION, "strong_protection_spend", id);
      } catch (err) {
        if (req.file) {
          try {
            unlinkSync(req.file.path);
          } catch {
            // already gone -- not worth failing the error response over.
          }
        }
        if (err instanceof InsufficientCoinsError) {
          return res.status(402).json({ error: "insufficient_coins", required: err.required, balance: err.balance });
        }
        throw err;
      }
    }
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
        strongProtectionLatentEpsilon: parsed.data.strongProtectionLatentEpsilon ?? null,
        strongProtectionPixelEpsilon: parsed.data.strongProtectionPixelEpsilon ?? null,
        allowAiTraining: parsed.data.allowAiTraining,
        tags: JSON.stringify(parsed.data.tags),
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

    // Compliance audit trail (schema.ts's complianceAuditLogs doc) -- the
    // creator's declared AI-training consent (allowAiTraining -> the
    // on-chain doNotTrain flag, see orchestrate.py) and the protection
    // tier they chose, at the moment of upload. Never blocks or fails the
    // upload itself; a logging bug here shouldn't turn into a lost artwork.
    try {
      logComplianceAudit(db, {
        userWallet: parsed.data.ownerWalletAddress,
        actionType: "artwork_upload_consent",
        targetArtworkId: id,
        ipAddress: req.ip ?? null,
        userAgent: req.headers["user-agent"] ?? null,
        payload: { allowAiTraining: parsed.data.allowAiTraining, protectionProfile: parsed.data.protectionProfile },
      });
    } catch {
      // best-effort, see this call's own comment
    }

    // Fire-and-forget: see orchestration.ts's module doc for why this isn't
    // awaited here. Errors inside are caught and recorded as status=FAILED
    // on the row itself, not thrown here.
    void runUploadPipeline(db, id);

    res.status(202).json({ id, status: "UPLOADED" });
  });

  // Upload-preview step (PixAI-style image-to-tag): the frontend calls this
  // right after the user picks a file, before the real POST / -- shows
  // suggested tags to review/edit, then the real upload resubmits the same
  // file plus whatever tags list the user confirmed. A second real upload
  // of the same bytes (not a shared temp-file handoff) is the deliberate,
  // simpler tradeoff here -- see this route's own temp-file cleanup below
  // for why that's fine: nothing about this request persists past the
  // response.
  router.post("/suggest-tags", upload.single("image"), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "upload an image file" });
    }
    try {
      const tags = await suggestTags(req.file.path);
      res.json({ tags });
    } catch (err) {
      res.status(502).json({
        error: `protection-svc suggest-tags failed: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      // Best-effort cleanup -- this file was only ever needed for the CLIP
      // pass above, unlike the real upload path where encryptImageAtRest
      // itself deletes the plaintext once it's done with it. Synchronous
      // (not fire-and-forget) so the file is reliably gone by the time
      // this request finishes, not racing the response.
      try {
        unlinkSync(req.file.path);
      } catch {
        // already gone, or some other non-critical cleanup failure -- not
        // worth failing an otherwise-successful (or already-failed) request over.
      }
    }
  });

  router.get("/", (req, res) => {
    const creatorId = typeof req.query.creatorId === "string" ? req.query.creatorId : undefined;
    const tag = typeof req.query.tag === "string" && req.query.tag.trim() ? req.query.tag.trim() : undefined;
    // Profile-page feature (2026-08-14): a caller looking at someone ELSE's
    // artworks (not their own gallery) must only ever see what that creator
    // has actually published publicly -- unlike the plain creatorId filter
    // above, which api-gateway's own GET /artworks route only ever uses for
    // "the authenticated caller's own gallery" (drafts/private/failed rows
    // included, deliberately). api-gateway's separate GET /artworks/by-
    // creator/:id route always sets this, and never lets a caller turn it
    // off for someone else's id -- see that route's own doc.
    //
    // Tag search (2026-08-14) forces the same publicOnly restriction even
    // when the caller didn't ask for it -- a global "search by tag" is
    // discovery across every creator's work, and must never become a way
    // to enumerate someone else's private/draft rows by guessing tags.
    const publicOnly = req.query.publicOnly === "true" || (tag !== undefined && creatorId === undefined);

    const rows =
      creatorId && publicOnly
        ? db
            .select()
            .from(artworks)
            .where(and(eq(artworks.creatorId, creatorId), eq(artworks.visibility, "public"), eq(artworks.status, "PUBLISHED")))
            .orderBy(desc(artworks.publishedAt))
            .all()
        : publicOnly
          ? db
              .select()
              .from(artworks)
              .where(and(eq(artworks.visibility, "public"), eq(artworks.status, "PUBLISHED")))
              .orderBy(desc(artworks.publishedAt))
              .all()
          : creatorId
            ? db.select().from(artworks).where(eq(artworks.creatorId, creatorId)).all()
            : db.select().from(artworks).all();

    // Tags are stored as a JSON-encoded string column; matched here in JS
    // rather than a SQL LIKE (which would also false-positive on
    // substrings across tag boundaries, e.g. "art" matching "cart").
    const matchesTag = (row: (typeof rows)[number]): boolean => {
      if (!tag) return true;
      const rowTags: string[] = JSON.parse(row.tags);
      return rowTags.some((t) => t.toLowerCase() === tag.toLowerCase());
    };

    // Without this, GalleryPage/FeedPage have no way to know whether an
    // artwork has a renderable image yet -- ArtworkImage needs
    // assetVersions.length > 0 before it'll even ask delivery-gateway for
    // a signed URL. GET /artworks/:id already joined this in; the list
    // route never did.
    const withVersions = attachAssetVersions(db, rows.filter(matchesTag));
    res.json(withVersions.map((row) => ({ ...row, tags: JSON.parse(row.tags) })));
  });

  // Bot access log ingest + query (registered before GET "/:id" -- both
  // are single-segment paths, and Express's "/:id" would otherwise swallow
  // "/bot-access-logs" as if it were an artwork id). delivery-gateway's
  // render_asset handler POSTs one row here per classified-bot request
  // (best-effort, fire-and-forget -- see delivery-gateway/src/lib.rs's own
  // doc), on top of its own fast in-memory tail; this is the durable
  // history a creator can still see after that process restarts or its
  // ring buffer rolls over.
  router.post("/bot-access-logs", (req, res) => {
    const parsed = botAccessLogBodySchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.message });
    }
    db.insert(botAccessLogs)
      .values({
        artworkId: parsed.data.artworkId,
        botName: parsed.data.botName,
        botGroup: parsed.data.botGroup,
        action: parsed.data.action,
        clientIp: parsed.data.clientIp,
        userAgent: parsed.data.userAgent,
        responseStatus: parsed.data.responseStatus,
        timestamp: new Date(parsed.data.timestampUnixSeconds * 1000),
      })
      .run();
    res.status(201).json({ status: "recorded" });
  });

  router.get("/bot-access-logs", (req, res) => {
    const artworkId = typeof req.query.artworkId === "string" ? req.query.artworkId : undefined;
    const limit = Math.min(Number(req.query.limit) || 100, 500);
    const rows = (
      artworkId
        ? db.select().from(botAccessLogs).where(eq(botAccessLogs.artworkId, artworkId))
        : db.select().from(botAccessLogs)
    )
      .orderBy(desc(botAccessLogs.id))
      .limit(limit)
      .all();
    res.json(rows);
  });

  // Bulk read for delivery-gateway's startup cache hydration (main.rs) --
  // only rows with an explicit policy ever set (no row = the default
  // applies, same reasoning as GET /:id/bot-policy's own default-on-miss
  // response). Registered before GET "/:id" for the same single-segment-
  // path-ordering reason as /bot-access-logs above.
  router.get("/bot-policies", (req, res) => {
    const rows = db.select().from(botPolicies).all();
    res.json(
      rows.map((row) => ({
        artworkId: row.artworkId,
        defaultAction: row.defaultAction,
        groupPolicies: JSON.parse(row.groupPoliciesJson),
        botOverrides: JSON.parse(row.botOverridesJson),
      })),
    );
  });

  router.get("/:id", (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }

    const versions = db.select().from(assetVersions).where(eq(assetVersions.artworkId, artwork.id)).all();
    const ownership = db.select().from(ownershipRecords).where(eq(ownershipRecords.artworkId, artwork.id)).all();

    // 2026-08-14 redesign: availability is now "creator hasn't blocked it",
    // not "creator has pre-generated it" -- generation happens lazily on
    // first unlock (see the unlock route below). versions.some(...) is no
    // longer how this is decided; kept the field name for frontend compat.
    const originalPreviewAvailable = !artwork.originalPreviewBlocked;
    const viewerId = typeof req.query.userId === "string" ? req.query.userId : undefined;
    const originalPreviewUnlockedByViewer =
      !!viewerId &&
      (viewerId === artwork.creatorId ||
        !!db
          .select()
          .from(originalPreviewUnlocks)
          .where(and(eq(originalPreviewUnlocks.userId, viewerId), eq(originalPreviewUnlocks.artworkId, artwork.id)))
          .get());

    res.json({
      ...artwork,
      tags: JSON.parse(artwork.tags),
      assetVersions: versions,
      ownershipRecords: ownership,
      originalPreviewAvailable,
      originalPreviewUnlockedByViewer,
    });
  });

  // Cancel-upload feature (2026-08-14). Creator-only (api-gateway checks
  // creatorId before proxying here, same trust boundary as remeasure-
  // protection below). Two different behaviors depending on where the
  // artwork actually is, both reachable through the same button per the
  // user's own request ("업로드가 되고 나서도 사용 할 수 있도록"):
  //   - still in flight (UPLOADED/PROTECTING/REGISTERING): best-effort
  //     asks protection-svc to cancel the underlying job (cancelProtectJob,
  //     itself best-effort against RunPod -- see that chain's own docs),
  //     then marks the artwork FAILED with an honest "cancelled by
  //     creator" message. orchestration.ts's own poll loop, if still
  //     running, will see the job already terminal on its next poll and
  //     stop -- no separate signal needed there.
  //   - already terminal (PUBLISHED/FAILED): there's no job left to
  //     cancel, so this becomes a real delete -- removes the artwork and
  //     its dependent rows outright. Best-effort storage cleanup (a
  //     leftover encrypted blob isn't worth failing the request over, same
  //     reasoning as this file's other best-effort cleanup steps).
  router.post("/:id/cancel", async (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }

    if (artwork.status === "UPLOADED" || artwork.status === "PROTECTING" || artwork.status === "REGISTERING") {
      if (artwork.protectJobId) {
        await cancelProtectJob(artwork.protectJobId);
      }
      db.update(artworks)
        .set({ status: "FAILED", errorMessage: "cancelled by creator", updatedAt: new Date() })
        .where(eq(artworks.id, artwork.id))
        .run();
      return res.json({ id: artwork.id, status: "FAILED", cancelled: true });
    }

    db.delete(assetVersions).where(eq(assetVersions.artworkId, artwork.id)).run();
    db.delete(ownershipRecords).where(eq(ownershipRecords.artworkId, artwork.id)).run();
    db.delete(originalPreviewUnlocks).where(eq(originalPreviewUnlocks.artworkId, artwork.id)).run();
    db.delete(botPolicies).where(eq(botPolicies.artworkId, artwork.id)).run();
    db.delete(artworks).where(eq(artworks.id, artwork.id)).run();
    try {
      await getObjectStorage().delete(artwork.encryptedImagePath);
    } catch {
      // best-effort, see this route's own doc
    }
    res.json({ id: artwork.id, deleted: true });
  });

  // Lazily generates (or refreshes) the downscaled+watermarked derivative
  // via protection-svc's own /original-preview (pure PIL, synchronous, see
  // that module's doc) and upserts it as an asset_versions row. Called from
  // the unlock route below the first time any viewer actually pays for it
  // -- 2026-08-14 redesign removed the old separate creator "generate in
  // advance" step (POST /:id/original-preview) since nothing needed the
  // derivative to exist before the first real unlock.
  async function ensureOriginalPreviewGenerated(artwork: typeof artworks.$inferSelect): Promise<void> {
    const existing = db
      .select()
      .from(assetVersions)
      .where(and(eq(assetVersions.artworkId, artwork.id), eq(assetVersions.variantName, "original_preview")))
      .get();
    if (existing) return;

    let decryptedTempPath: string | undefined;
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
      const { previewUri, width, height } = await createOriginalPreview(decryptedTempPath, artwork.watermarkPayloadHex);
      db.insert(assetVersions)
        .values({
          artworkId: artwork.id,
          variantName: "original_preview",
          storageUri: previewUri,
          width,
          height,
          scaleVsSource: 1,
          // Not a protection tier -- this variant is a deliberately
          // near-original derivative, gated by the coin unlock below, not
          // by protection strength (rust-core's SAFE/UNKNOWN/UNSAFE
          // vocabulary is about the OTHER variants' AI-training
          // resistance, which this one has none of by design).
          protectionStatus: "UNSAFE",
        })
        .run();
    } finally {
      if (decryptedTempPath) cleanupTempFile(decryptedTempPath);
    }
  }

  const originalPreviewBlockedSchema = z.object({ blocked: z.boolean() });

  // Creator-only opt-OUT (api-gateway checks creatorId before proxying
  // here, same trust boundary as remeasure-protection below). Replaces the
  // old opt-IN toggle -- see schema.ts's originalPreviewBlocked doc for why
  // the polarity flipped. Setting blocked=true stops new unlocks (the
  // route below 403s) but doesn't revoke unlocks a viewer already paid
  // for, same as a creator raising a price doesn't refund past buyers.
  router.put("/:id/original-preview-blocked", (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }
    const parsed = originalPreviewBlockedSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    db.update(artworks).set({ originalPreviewBlocked: parsed.data.blocked }).where(eq(artworks.id, artwork.id)).run();
    try {
      logComplianceAudit(db, {
        userWallet: artwork.ownerWalletAddress,
        actionType: "original_preview_blocked_changed",
        targetArtworkId: artwork.id,
        ipAddress: req.ip ?? null,
        userAgent: req.headers["user-agent"] ?? null,
        payload: { blocked: parsed.data.blocked },
      });
    } catch {
      // best-effort, see logComplianceAudit call sites' shared reasoning
    }
    res.status(200).json({ originalPreviewBlocked: parsed.data.blocked });
  });

  const unlockOriginalPreviewSchema = z.object({ userId: z.string().min(1) });

  // Viewer-side unlock -- creator gets it free (their own artwork), anyone
  // else spends COIN_COSTS.ORIGINAL_PREVIEW_UNLOCK once for a permanent
  // per-(userId, artworkId) grant (originalPreviewUnlocks table). No
  // creatorId check here (unlike the toggle route above) -- this is
  // deliberately callable by any authenticated user, that's the whole
  // point of "anyone logged in can view once they've unlocked it".
  // 2026-08-14: no longer requires a pre-existing derivative -- generates
  // it on demand (ensureOriginalPreviewGenerated) the first time anyone
  // actually unlocks, instead of requiring the creator to opt in first.
  router.post("/:id/original-preview/unlock", async (req, res) => {
    const parsed = unlockOriginalPreviewSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const { userId } = parsed.data;

    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }
    if (artwork.originalPreviewBlocked) {
      return res.status(403).json({ error: "creator has disabled original-preview access for this artwork" });
    }

    const alreadyUnlocked =
      userId === artwork.creatorId ||
      !!db
        .select()
        .from(originalPreviewUnlocks)
        .where(and(eq(originalPreviewUnlocks.userId, userId), eq(originalPreviewUnlocks.artworkId, artwork.id)))
        .get();

    if (!alreadyUnlocked) {
      try {
        spendCoins(db, userId, COIN_COSTS.ORIGINAL_PREVIEW_UNLOCK, "original_preview_unlock", artwork.id);
      } catch (err) {
        if (err instanceof InsufficientCoinsError) {
          return res.status(402).json({ error: "insufficient_coins", required: err.required, balance: err.balance });
        }
        throw err;
      }
      db.insert(originalPreviewUnlocks).values({ userId, artworkId: artwork.id, unlockedAt: new Date() }).run();
    }

    try {
      await ensureOriginalPreviewGenerated(artwork);
    } catch (err) {
      // The coin spend / unlock row above already committed -- a
      // generation failure here (e.g. protection-svc unreachable) shouldn't
      // silently pretend the viewer never paid. Surface it as a real error;
      // the next unlock-route call for this same viewer will retry
      // generation for free (alreadyUnlocked short-circuits the spend) since
      // ensureOriginalPreviewGenerated is naturally idempotent/retriable.
      return res
        .status(502)
        .json({ error: `original-preview generation failed: ${err instanceof Error ? err.message : String(err)}` });
    }

    res.status(200).json({ originalPreviewUnlockedByViewer: true });
  });

  // Test Lab's "재실행" button (apps/web's TestLabPage) -- a live re-run of
  // the styleDriftScore/etc measurement stored on the row at upload time,
  // not a rewrite of that stored value (this never UPDATEs the artworks
  // table). Compares the real original (decrypted on demand -- it's
  // envelope-encrypted at rest, see crypto/imageEncryption.ts) against the
  // artwork's actual *published* image (protectedImageUri: the final
  // watermarked+C2PA-signed file, not protection-svc's own intermediate
  // pre-watermark cloaked.png) -- arguably the more honest comparison
  // anyway, since that published file is what would actually get
  // redistributed/trained on, not an internal intermediate nobody else
  // ever sees.
  router.post("/:id/remeasure-protection", async (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }
    if (!artwork.protectedImageUri) {
      return res.status(400).json({ error: "artwork has no protected image yet" });
    }

    let decryptedTempPath: string | undefined;
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
      const metrics = await remeasureProtection(decryptedTempPath, artwork.protectedImageUri);
      res.json(metrics);
    } catch (err) {
      res.status(502).json({ error: `re-measurement failed: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      if (decryptedTempPath) cleanupTempFile(decryptedTempPath);
    }
  });

  // Test Lab's real-LoRA-training protection score -- gated to
  // usedStrongProtection artworks only (the one tier this project has
  // actually validated by training a real LoRA against; every other tier
  // only has the proxy VGG19-feature-distance measurement remeasure-
  // protection above surfaces). Unlike remeasure-protection, this is
  // job-based, not synchronous: protection-svc trains four LoRAs on
  // RunPod Serverless, minutes not milliseconds (see protection_score.py's
  // module doc). Returns the jobId immediately -- callers poll GET
  // /score-protection-jobs/:jobId, same shape as detection-svc's
  // case-based polling the web UI already uses for scan/report.
  router.post("/:id/score-protection", async (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }
    if (!artwork.usedStrongProtection) {
      return res
        .status(400)
        .json({ error: "score-protection is only available for artworks strong_protection actually ran on" });
    }
    if (!artwork.protectedImageUri) {
      return res.status(400).json({ error: "artwork has no protected image yet" });
    }

    let decryptedTempPath: string | undefined;
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
      // Tags the creator actually chose at upload time describe the
      // artwork's real content far better than the freeform title (a
      // title like "무제" or a pun tells protection_score.py's LoRA
      // nothing to generate toward). Sent empty when there are no tags --
      // protection_score.py falls back to its own auto-caption in that
      // case, same as it always used to, rather than risking the title's
      // known content-collapse bug.
      const uploadTags: string[] = JSON.parse(artwork.tags);
      const scorePrompt = uploadTags.join(", ");

      const { jobId } = await createScoreProtectionJob({
        originalImageUri: decryptedTempPath,
        protectedImageUri: artwork.protectedImageUri,
        prompt: scorePrompt,
      });

      // Recorded BEFORE the fire-and-forget poll starts -- this row is
      // what lets recoverPendingScoreProtectionJobs (orchestration.ts)
      // pick this job back up if asset-service restarts before the poll
      // below finishes (a real incident this fixed: a real ~11-minute
      // RunPod job completed with a real result that nothing ever
      // recorded, because the in-memory promise tracking it had been
      // killed by an unrelated restart hours earlier).
      db.insert(pendingScoreProtectionJobs)
        .values({ artworkId: artwork.id, jobId, submittedAt: new Date() })
        .onConflictDoUpdate({
          target: pendingScoreProtectionJobs.artworkId,
          set: { jobId, submittedAt: new Date() },
        })
        .run();

      // Fire-and-forget: the decrypted temp file must outlive this HTTP
      // response (protection-svc's background job reads it well after
      // this request returns), so cleanup can't happen in a `finally`
      // here the way remeasure-protection's synchronous flow does above.
      // persistScoreProtectionResult (orchestration.ts) does the actual
      // polling + durable persistence + pendingScoreProtectionJobs cleanup;
      // this route's only remaining job is the temp file.
      void persistScoreProtectionResult(db, artwork.id, jobId).finally(() => cleanupTempFile(decryptedTempPath!));

      res.status(202).json({ jobId });
    } catch (err) {
      if (decryptedTempPath) cleanupTempFile(decryptedTempPath);
      res.status(502).json({ error: `score-protection failed: ${err instanceof Error ? err.message : String(err)}` });
    }
  });

  // The persisted counterpart to POST /:id/score-protection above --
  // returns the last real result for this artwork regardless of whether
  // the caller ever saw it live (tab closed mid-job, reloaded, came back
  // days later). 404 means "no completed run yet", not an error.
  router.get("/:id/score-protection-result", (req, res) => {
    const stored = db
      .select()
      .from(scoreProtectionResults)
      .where(eq(scoreProtectionResults.artworkId, req.params.id))
      .get();
    if (!stored) {
      return res.status(404).json({ error: `no stored score-protection result for ${req.params.id}` });
    }
    res.json({ ...JSON.parse(stored.resultJson), checkedAt: stored.checkedAt });
  });

  // Per-artwork ALLOW/BLOCK/LOG_ONLY bot policy -- system of record for
  // delivery-gateway's BotPolicyStore cache (src/bot_policy.rs). Creator-
  // only mutation (api-gateway checks creatorId before proxying here, same
  // trust boundary as /original-preview above); GET has no such check since
  // delivery-gateway itself needs to read this to hydrate its cache.
  router.get("/:id/bot-policy", (req, res) => {
    const row = db.select().from(botPolicies).where(eq(botPolicies.artworkId, req.params.id)).get();
    if (!row) {
      // No row = no explicit policy ever set -- mirrors delivery-gateway's
      // own ArtworkBotPolicy::default() (bot_policy.rs) exactly, so a
      // caller sees the same effective policy delivery-gateway would
      // apply, not a 404 for what's actually a well-defined default.
      return res.json({
        defaultAction: "LOG_ONLY",
        groupPolicies: { ai_training: "BLOCK", search_engine: "ALLOW", social_preview: "ALLOW" },
        botOverrides: {},
      });
    }
    res.json({
      defaultAction: row.defaultAction,
      groupPolicies: JSON.parse(row.groupPoliciesJson),
      botOverrides: JSON.parse(row.botOverridesJson),
    });
  });

  router.put("/:id/bot-policy", (req, res) => {
    const artwork = db.select().from(artworks).where(eq(artworks.id, req.params.id)).get();
    if (!artwork) {
      return res.status(404).json({ error: `no artwork ${req.params.id}` });
    }
    const parsed = botPolicyBodySchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.message });
    }
    const now = new Date();
    db.insert(botPolicies)
      .values({
        artworkId: req.params.id,
        defaultAction: parsed.data.defaultAction,
        groupPoliciesJson: JSON.stringify(parsed.data.groupPolicies),
        botOverridesJson: JSON.stringify(parsed.data.botOverrides),
        updatedAt: now,
      })
      .onConflictDoUpdate({
        target: botPolicies.artworkId,
        set: {
          defaultAction: parsed.data.defaultAction,
          groupPoliciesJson: JSON.stringify(parsed.data.groupPolicies),
          botOverridesJson: JSON.stringify(parsed.data.botOverrides),
          updatedAt: now,
        },
      })
      .run();
    try {
      logComplianceAudit(db, {
        userWallet: artwork.ownerWalletAddress,
        actionType: "bot_policy_updated",
        targetArtworkId: artwork.id,
        ipAddress: req.ip ?? null,
        userAgent: req.headers["user-agent"] ?? null,
        payload: parsed.data,
      });
    } catch {
      // best-effort, see logComplianceAudit call sites' shared reasoning
    }
    res.json(parsed.data);
  });

  router.get("/score-protection-jobs/:jobId", async (req, res) => {
    try {
      const job = await getScoreProtectionJob(req.params.jobId);
      res.json(job);
    } catch (err) {
      res
        .status(502)
        .json({ error: `score-protection job lookup failed: ${err instanceof Error ? err.message : String(err)}` });
    }
  });

  return router;
}
