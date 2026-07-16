import { Router } from "express";
import multer from "multer";
import { z } from "zod";
import { requireAuth } from "../middleware/requireAuth.js";
import {
  createArtwork,
  createArtworkWithFile,
  getArtwork,
  listArtworks,
  AssetServiceError,
} from "../clients/assetService.js";
import { signRenderUrl } from "../clients/deliveryGateway.js";

const createArtworkSchema = z.object({
  title: z.string().min(1),
  sourceImageUri: z.string().min(1).optional(),
  protectionProfile: z.enum(["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"]).optional(),
  // Not z.coerce.boolean() -- Boolean("false") is true in JS, so a real
  // "false" multipart field would coerce to true. See asset-service's
  // identical fix (routes/artworks.ts) for the live bug this was caught
  // from (S3_USE_SSL=false silently read as true).
  allowAiTraining: z
    .union([z.boolean(), z.enum(["true", "false"])])
    .optional()
    .transform((v) => (v === undefined ? undefined : v === true || v === "true")),
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
            { title: parsed.data.title, protectionProfile: parsed.data.protectionProfile, allowAiTraining: parsed.data.allowAiTraining, file: req.file },
            req.user!.sub,
            req.user!.walletAddress,
          )
        : await createArtwork(
            { title: parsed.data.title, sourceImageUri: parsed.data.sourceImageUri!, protectionProfile: parsed.data.protectionProfile, allowAiTraining: parsed.data.allowAiTraining },
            req.user!.sub,
            req.user!.walletAddress,
          );
      res.status(202).json(result);
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

  router.get("/:id", async (req, res) => {
    try {
      res.json(await getArtwork(req.params.id));
    } catch (err) {
      forwardAssetServiceError(err, res);
    }
  });

  const renderVariantQuery = z.object({ variant: z.enum(["logged_in", "thumbnail"]).default("logged_in") });

  // Every caller of this web app is authenticated (ProtectedRoute wraps
  // the whole gallery/feed/detail UI) -- there's no "anonymous browsing"
  // path in this app yet, so this always signs as "logged_in"/"thumbnail",
  // never "anonymous". A future public-browsing feature would need its
  // own unauthenticated route that signs "anonymous" instead.
  router.get("/:id/render-url", async (req, res) => {
    const parsed = renderVariantQuery.safeParse(req.query);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    try {
      const url = await signRenderUrl(req.params.id, parsed.data.variant);
      res.json({ url });
    } catch {
      res.status(502).json({ error: "delivery-gateway unreachable" });
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
