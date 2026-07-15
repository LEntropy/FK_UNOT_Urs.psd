import { Router } from "express";
import { z } from "zod";
import { requireAuth } from "../middleware/requireAuth.js";
import { createArtwork, getArtwork, listArtworks, AssetServiceError } from "../clients/assetService.js";
import { signRenderUrl } from "../clients/deliveryGateway.js";

const createArtworkSchema = z.object({
  title: z.string().min(1),
  sourceImageUri: z.string().min(1),
  protectionProfile: z.enum(["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"]).optional(),
  allowAiTraining: z.boolean().optional(),
});

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

  router.post("/", async (req, res) => {
    const parsed = createArtworkSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }

    try {
      const result = await createArtwork(parsed.data, req.user!.sub, req.user!.walletAddress);
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
