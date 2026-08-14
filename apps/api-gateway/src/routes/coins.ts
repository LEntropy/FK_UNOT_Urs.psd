import { Router } from "express";
import { requireAuth } from "../middleware/requireAuth.js";
import { getCoinBalance, AssetServiceError } from "../clients/assetService.js";

/** Thin authenticated proxy to asset-service's coin ledger -- same pattern
 * as artworksRouter (identity comes from the verified JWT, never the
 * request). */
export function coinsRouter(): Router {
  const router = Router();
  router.use(requireAuth);

  router.get("/coins/balance", async (req, res) => {
    try {
      res.json(await getCoinBalance(req.user!.sub));
    } catch (err) {
      if (err instanceof AssetServiceError) {
        return res.status(err.status).json(err.body);
      }
      res.status(502).json({ error: "asset-service unreachable" });
    }
  });

  return router;
}
