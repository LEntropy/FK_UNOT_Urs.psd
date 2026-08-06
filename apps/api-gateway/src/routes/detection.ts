import { Router } from "express";
import { z } from "zod";
import { requireAuth } from "../middleware/requireAuth.js";
import { getArtwork, AssetServiceError } from "../clients/assetService.js";
import { scanArtwork, reportArtwork, reportModelLeak, getCase, getEvidence, DetectionServiceError } from "../clients/detectionService.js";

/**
 * Authenticated proxy in front of detection-svc (which has no auth of its
 * own, same trust boundary as asset-service -- see artworks.ts's module
 * doc). Backs the "테스트" tab's 추적·증빙 test: only an artwork's own
 * creator can trigger a scan/report or read back its evidence, since a
 * bundle carries the rights holder's wallet address and other case detail
 * that isn't otherwise public. Ownership is checked by fetching the
 * artwork/case from the owning service on every call rather than trusting
 * a client-supplied claim -- this router has no database of its own.
 */
export function detectionRouter(): Router {
  const router = Router();
  router.use(requireAuth);

  async function assertOwnsArtwork(artworkId: string, userId: string): Promise<{ ok: true } | { ok: false; status: number; body: unknown }> {
    try {
      const artwork = await getArtwork(artworkId);
      if (artwork.creatorId !== userId) {
        return { ok: false, status: 403, body: { error: "not this artwork's creator" } };
      }
      return { ok: true };
    } catch (err) {
      if (err instanceof AssetServiceError) return { ok: false, status: err.status, body: err.body };
      return { ok: false, status: 502, body: { error: "asset-service unreachable" } };
    }
  }

  router.post("/artworks/:id/scan", async (req, res) => {
    const owns = await assertOwnsArtwork(req.params.id, req.user!.sub);
    if (!owns.ok) return res.status(owns.status).json(owns.body);

    try {
      res.status(202).json(await scanArtwork(req.params.id));
    } catch (err) {
      forwardDetectionError(err, res);
    }
  });

  const reportSchema = z.object({ suspectUrl: z.string().url() });

  router.post("/artworks/:id/report", async (req, res) => {
    const parsed = reportSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    const owns = await assertOwnsArtwork(req.params.id, req.user!.sub);
    if (!owns.ok) return res.status(owns.status).json(owns.body);

    try {
      res.status(202).json(await reportArtwork(req.params.id, parsed.data.suspectUrl));
    } catch (err) {
      forwardDetectionError(err, res);
    }
  });

  const modelLeakReportSchema = z.object({ suspectModelUrl: z.string().url() });

  router.post("/artworks/:id/model-leak-report", async (req, res) => {
    const parsed = modelLeakReportSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    const owns = await assertOwnsArtwork(req.params.id, req.user!.sub);
    if (!owns.ok) return res.status(owns.status).json(owns.body);

    try {
      res.status(202).json(await reportModelLeak(req.params.id, parsed.data.suspectModelUrl));
    } catch (err) {
      forwardDetectionError(err, res);
    }
  });

  // A case_id doesn't carry the caller's identity, so both of these look up
  // the case first (to learn its artwork_id) and re-check ownership the
  // same way the scan/report routes above do -- otherwise a guessed/leaked
  // case_id would leak another creator's evidence bundle (wallet address,
  // discovered URL, etc).
  router.get("/detection-cases/:caseId", async (req, res) => {
    try {
      const detectionCase = await getCase(req.params.caseId);
      const owns = await assertOwnsArtwork(detectionCase.artwork_id, req.user!.sub);
      if (!owns.ok) return res.status(owns.status).json(owns.body);
      res.json(detectionCase);
    } catch (err) {
      forwardDetectionError(err, res);
    }
  });

  router.get("/detection-cases/:caseId/evidence", async (req, res) => {
    try {
      const detectionCase = await getCase(req.params.caseId);
      const owns = await assertOwnsArtwork(detectionCase.artwork_id, req.user!.sub);
      if (!owns.ok) return res.status(owns.status).json(owns.body);
      res.json(await getEvidence(req.params.caseId));
    } catch (err) {
      forwardDetectionError(err, res);
    }
  });

  return router;
}

function forwardDetectionError(err: unknown, res: import("express").Response) {
  if (err instanceof DetectionServiceError) {
    return res.status(err.status).json(err.body);
  }
  res.status(502).json({ error: "detection-svc unreachable" });
}
