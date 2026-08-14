import { Readable } from "node:stream";
import { Router } from "express";
import { requireAuth } from "../middleware/requireAuth.js";
import { createLoraJob, getLoraJob, fetchLoraJobDownload, getArtwork, AssetServiceError } from "../clients/assetService.js";

/** Coin-system feature (2026-08-10) -- thin authenticated proxy to
 * asset-service's /lora-jobs routes, same identity-from-JWT pattern as
 * artworksRouter/coinsRouter. */
export function loraJobsRouter(): Router {
  const router = Router();
  router.use(requireAuth);

  router.post("/lora-jobs", async (req, res) => {
    const sourceArtworkId = typeof req.body?.sourceArtworkId === "string" ? req.body.sourceArtworkId : undefined;
    if (!sourceArtworkId) {
      return res.status(400).json({ error: "sourceArtworkId required" });
    }
    try {
      const artwork = await getArtwork(sourceArtworkId);
      if (artwork.creatorId !== req.user!.sub) {
        return res.status(403).json({ error: "not this artwork's creator" });
      }
      res.status(202).json(await createLoraJob(sourceArtworkId, req.user!.sub));
    } catch (err) {
      if (err instanceof AssetServiceError) {
        return res.status(err.status).json(err.body);
      }
      res.status(502).json({ error: "asset-service unreachable" });
    }
  });

  router.get("/lora-jobs/:id", async (req, res) => {
    try {
      res.json(await getLoraJob(req.params.id, req.user!.sub));
    } catch (err) {
      if (err instanceof AssetServiceError) {
        return res.status(err.status).json(err.body);
      }
      res.status(502).json({ error: "asset-service unreachable" });
    }
  });

  router.get("/lora-jobs/:id/download", async (req, res) => {
    try {
      const upstream = await fetchLoraJobDownload(req.params.id, req.user!.sub);
      res.setHeader("Content-Type", upstream.headers.get("content-type") ?? "application/octet-stream");
      const disposition = upstream.headers.get("content-disposition");
      if (disposition) res.setHeader("Content-Disposition", disposition);
      // Streams the .safetensors body straight through -- this can be a
      // real multi-MB file, no reason to buffer it in this process first.
      Readable.fromWeb(upstream.body as import("stream/web").ReadableStream).pipe(res);
    } catch (err) {
      if (err instanceof AssetServiceError) {
        return res.status(err.status).json(err.body);
      }
      res.status(502).json({ error: "asset-service unreachable" });
    }
  });

  return router;
}
