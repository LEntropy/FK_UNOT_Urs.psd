import { Router } from "express";
import multer from "multer";
import { z } from "zod";
import { requireAuth } from "../middleware/requireAuth.js";
import {
  createArtwork,
  createArtworkWithFile,
  getArtwork,
  listArtworks,
  listPublicArtworksByCreator,
  searchArtworksByTag,
  getPlatformStats,
  suggestTags,
  remeasureProtection,
  createScoreProtectionJob,
  getScoreProtectionJob,
  getStoredScoreProtectionResult,
  setOriginalPreviewBlocked,
  unlockOriginalPreview,
  cancelArtwork,
  getBotPolicy,
  setBotPolicy,
  getBotAccessLogs,
  AssetServiceError,
} from "../clients/assetService.js";
import { signRenderUrl } from "../clients/deliveryGateway.js";

const createArtworkSchema = z.object({
  title: z.string().min(1),
  sourceImageUri: z.string().min(1).optional(),
  protectionProfile: z.enum(["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN", "STRONG_PROTECTION"]).optional(),
  // Advanced-options upload feature (2026-08-08, redesigned 2026-08-15) --
  // see asset-service's identical field for the bounds reasoning. Passed
  // through as-is; asset-service does the real validation.
  strongProtectionEpsilon: z.coerce.number().min(0.01).max(0.3).optional(),
  // Not z.coerce.boolean() -- Boolean("false") is true in JS, so a real
  // "false" multipart field would coerce to true. See asset-service's
  // identical fix (routes/artworks.ts) for the live bug this was caught
  // from (S3_USE_SSL=false silently read as true).
  allowAiTraining: z
    .union([z.boolean(), z.enum(["true", "false"])])
    .optional()
    .transform((v) => (v === undefined ? undefined : v === true || v === "true")),
  // Real array from a JSON body, or a JSON-encoded string from a multipart
  // field -- same reasoning as allowAiTraining's string variant above.
  // Passed through as-is to asset-service, which does its own parsing/
  // validation (this gateway's job is auth + identity injection, not
  // reshaping the payload -- see this file's module doc).
  tags: z
    .union([z.array(z.string()), z.string()])
    .optional()
    .transform((v) => {
      if (v === undefined || Array.isArray(v)) return v;
      try {
        const parsed = JSON.parse(v);
        return Array.isArray(parsed) ? parsed.filter((t): t is string => typeof t === "string") : undefined;
      } catch {
        return undefined;
      }
    }),
});

// Memory storage, not disk -- api-gateway only holds the bytes long enough
// to re-POST them to asset-service (createArtworkWithFile below), never
// writes them to its own filesystem at all.
const upload = multer({ limits: { fileSize: 50 * 1024 * 1024 } });

/**
 * Thin authenticated proxy in front of asset-service (which has no auth of
 * its own -- PROJECT_DESIGN.md §2's "인증·라우팅" role for api-gateway).
 * The frontend only ever talks to this router; creatorId/ownerWalletAddress
 * are taken from the verified JWT, never from the request body, so a
 * caller can't upload artwork as someone else.
 */
export function artworksRouter(): Router {
  const router = Router();
  router.use(requireAuth);

  router.post("/", upload.single("image"), async (req, res) => {
    const parsed = createArtworkSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    if (!req.file && !parsed.data.sourceImageUri) {
      return res.status(400).json({ error: "either upload an image file or provide sourceImageUri" });
    }

    try {
      const result = req.file
        ? await createArtworkWithFile(
            {
              title: parsed.data.title,
              protectionProfile: parsed.data.protectionProfile,
              strongProtectionEpsilon: parsed.data.strongProtectionEpsilon,
              allowAiTraining: parsed.data.allowAiTraining,
              tags: parsed.data.tags,
              file: req.file,
            },
            req.user!.sub,
            req.user!.walletAddress,
          )
        : await createArtwork(
            {
              title: parsed.data.title,
              sourceImageUri: parsed.data.sourceImageUri!,
              protectionProfile: parsed.data.protectionProfile,
              strongProtectionEpsilon: parsed.data.strongProtectionEpsilon,
              allowAiTraining: parsed.data.allowAiTraining,
              tags: parsed.data.tags,
            },
            req.user!.sub,
            req.user!.walletAddress,
          );
      res.status(202).json(result);
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Upload-preview step (PixAI-style image-to-tag) -- see asset-service's
  // own suggest-tags route doc for the full flow. requireAuth still
  // applies (router.use above): this needs a logged-in user the same way
  // every other artwork action does, even though nothing gets persisted.
  router.post("/suggest-tags", upload.single("image"), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "upload an image file" });
    }
    try {
      const tags = await suggestTags(req.file);
      res.json({ tags });
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  router.get("/", async (req, res) => {
    try {
      res.json(await listArtworks(req.user!.sub));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Profile-page feature (2026-08-14) -- registered before GET "/:id" for
  // the usual reason (both are single-vs-two-segment paths that could
  // otherwise collide; "/by-creator" itself never matches "/:id" since
  // that's a different path shape, but kept in this position to group with
  // the other list route above). Any authenticated user can view any
  // creator's public gallery -- see listPublicArtworksByCreator's own doc
  // for why this can never leak drafts/private/failed rows.
  router.get("/by-creator/:creatorId", async (req, res) => {
    try {
      res.json(await listPublicArtworksByCreator(req.params.creatorId));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Tag-search feature (2026-08-14) -- registered before GET "/:id" for the
  // same reason as "/by-creator" above ("/search" would otherwise be
  // swallowed as an artwork id). Query-param-driven rather than a path
  // segment since a tag can contain characters path segments don't like.
  router.get("/search", async (req, res) => {
    const tag = typeof req.query.tag === "string" ? req.query.tag.trim() : "";
    if (!tag) {
      return res.status(400).json({ error: "tag query param is required" });
    }
    try {
      res.json(await searchArtworksByTag(tag));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Same "/:id" collision reasoning as "/search" and "/by-creator" above.
  router.get("/stats", async (_req, res) => {
    try {
      res.json(await getPlatformStats());
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  router.get("/:id", async (req, res) => {
    try {
      res.json(await getArtwork(req.params.id, req.user!.sub));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Cancel-upload feature (2026-08-14) -- creator-only, same check pattern
  // as every other creator-only route here. Works regardless of the
  // artwork's current status (asset-service's own route decides whether
  // that means "cancel the in-flight job" or "delete outright" -- see its
  // doc), matching the user's own requirement that the button stay usable
  // even after the upload has already finished.
  router.post("/:id/cancel", async (req, res) => {
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.json(await cancelArtwork(req.params.id));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Creator-only opt-OUT (2026-08-14 redesign, replaces the old enable/
  // disable pair) -- see clients/assetService.ts's setOriginalPreviewBlocked
  // doc. Same creator-check pattern as remeasure-protection below.
  const originalPreviewBlockedSchema = z.object({ blocked: z.boolean() });
  router.put("/:id/original-preview-blocked", async (req, res) => {
    const parsed = originalPreviewBlockedSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.json(await setOriginalPreviewBlocked(req.params.id, parsed.data.blocked));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Viewer-side unlock -- no creator check (deliberately callable by any
  // authenticated user, see asset-service's own route doc); identity comes
  // from the verified JWT, never the request body.
  router.post("/:id/original-preview/unlock", async (req, res) => {
    try {
      res.json(await unlockOriginalPreview(req.params.id, req.user!.sub));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Test Lab's "재실행" button -- a real SSH round trip to the GPU PC per
  // call (remote_measure_existing_images), so gated to the artwork's own
  // creator like the detection routes' scan/report are, unlike plain GET
  // /:id which any authenticated user can view (community feed browsing).
  router.post("/:id/remeasure-protection", async (req, res) => {
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.json(await remeasureProtection(req.params.id));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Test Lab's real-LoRA-training protection score -- same creator-only
  // gating as remeasure-protection above (this triggers a real, several-
  // minutes RunPod Serverless job, not free to spam). Job-based: this
  // route only kicks the job off and returns its jobId; the poll route
  // below is what the frontend actually repeats.
  router.post("/:id/score-protection", async (req, res) => {
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.status(202).json(await createScoreProtectionJob(req.params.id));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // Re-checks creator ownership on every poll, not just at job creation --
  // cheap (one extra getArtwork call) and closes the gap where a jobId
  // could otherwise be guessed/replayed by a different authenticated user
  // to read another creator's score results.
  router.get("/:id/score-protection/:jobId", async (req, res) => {
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.json(await getScoreProtectionJob(req.params.jobId));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  // The persisted counterpart to score-protection above -- returns the
  // last real result whenever the creator wants it, not just while the
  // job that produced it is still tracked in their own browser tab.
  router.get("/:id/score-protection-result", async (req, res) => {
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      const stored = await getStoredScoreProtectionResult(req.params.id);
      if (!stored) {
        return res.status(404).json({ error: `no stored score-protection result for ${req.params.id}` });
      }
      res.json(stored);
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  const batchRenderUrlSchema = z.object({
    ids: z.array(z.string().min(1)).min(1).max(100),
    variant: z.enum(["logged_in", "thumbnail"]).default("thumbnail"),
  });

  // Feed/gallery performance fix (2026-08-14): a grid of N thumbnails used
  // to make N separate GET /:id/render-url round trips, one per
  // <ArtworkImage>, before a single <img> could even start loading -- on
  // top of the N actual image fetches that follow, this routinely blew
  // past the browser's ~6-connections-per-origin limit and serialized
  // most of the grid behind that queue. One batched call here (still N
  // signRenderUrl calls server-to-server, but in parallel, and it's a pure
  // local HMAC computation on delivery-gateway's side, not a DB/network
  // call -- see clients/deliveryGateway.ts's own doc) cuts the browser
  // side down to a single request. Deliberately doesn't support
  // variant="original" -- that path needs a per-artwork asset-service
  // unlock check (see the single-artwork route below), which would defeat
  // the "no per-id asset-service round trip" point of batching; the
  // original-preview image is only ever rendered one at a time (artwork
  // detail page), where the existing single-artwork route already suffices.
  router.post("/render-urls", async (req, res) => {
    const parsed = batchRenderUrlSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    const entries = await Promise.all(
      parsed.data.ids.map(async (id) => {
        try {
          return [id, await signRenderUrl(id, parsed.data.variant)] as const;
        } catch {
          // One artwork's signing failure (e.g. it has no variants yet)
          // shouldn't 502 the whole batch -- the frontend already handles
          // a missing url per-card (same fallback ArtworkImage shows when
          // hasVariants is false).
          return [id, null] as const;
        }
      }),
    );
    res.json(Object.fromEntries(entries));
  });

  const renderVariantQuery = z.object({ variant: z.enum(["logged_in", "thumbnail", "original"]).default("logged_in") });

  // Every caller of this web app is authenticated (ProtectedRoute wraps
  // the whole gallery/feed/detail UI) -- there's no "anonymous browsing"
  // path in this app yet, so this always signs as "logged_in"/"thumbnail"/
  // "original", never "anonymous". A future public-browsing feature would
  // need its own unauthenticated route that signs "anonymous" instead.
  router.get("/:id/render-url", async (req, res) => {
    const parsed = renderVariantQuery.safeParse(req.query);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    try {
      // "original" needs an extra check delivery-gateway itself doesn't do
      // (it just serves whatever variant asset-service reports) -- the
      // per-viewer unlock gate is asset-service's own
      // originalPreviewUnlockedByViewer flag, re-checked here on every
      // request rather than trusted from an earlier call.
      if (parsed.data.variant === "original") {
        const artwork = await getArtwork(req.params.id, req.user!.sub);
        if (!artwork.originalPreviewUnlockedByViewer) {
          return res.status(403).json({ error: "original preview not unlocked for this viewer" });
        }
      }
      const viewer = parsed.data.variant === "original" ? "original_preview" : parsed.data.variant;
      const url = await signRenderUrl(req.params.id, viewer);
      res.json({ url });
    } catch (err) {
      if (err instanceof AssetServiceError) {
        return res.status(err.status).json(err.body);
      }
      res.status(502).json({ error: "delivery-gateway unreachable" });
    }
  });

  // Per-artwork bot ALLOW/BLOCK/LOG_ONLY policy (motection-changes handoff
  // feature, 2026-08-14 DB-backed follow-up -- see asset-service schema.ts's
  // botPolicies doc). GET has no creator check (matches asset-service's own
  // reasoning: nothing sensitive in reading back the current policy); PUT
  // is creator-only, same pattern as original-preview-blocked above.
  const botActionSchema = z.enum(["ALLOW", "BLOCK", "LOG_ONLY"]);
  const botPolicyBodySchema = z.object({
    defaultAction: botActionSchema,
    groupPolicies: z.record(z.string(), botActionSchema).default({}),
    botOverrides: z.record(z.string(), botActionSchema).default({}),
  });

  router.get("/:id/bot-policy", async (req, res) => {
    try {
      res.json(await getBotPolicy(req.params.id));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  router.put("/:id/bot-policy", async (req, res) => {
    const parsed = botPolicyBodySchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.json(await setBotPolicy(req.params.id, parsed.data));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  router.get("/:id/bot-access-logs", async (req, res) => {
    try {
      const artwork = await getArtwork(req.params.id);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      const limit = typeof req.query.limit === "string" ? Number(req.query.limit) : undefined;
      res.json(await getBotAccessLogs(req.params.id, limit));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  return router;
}

function forwardAssetServiceError(err: unknown, res: import("express").Response) {
  if (err instanceof AssetServiceError) {
    return res.status(err.status).json(err.body);
  }
  res.status(502).json({ error: "asset-service unreachable" });
}
