import { Router } from "express";
import { signEvidencePayload } from "../evidenceSigning.js";

/**
 * Internal-only endpoint (same trust boundary as delivery-gateway's own
 * /internal/sign -- network-level, not app-level auth, per that service's
 * README) -- detection-svc calls this to get a real signature for the
 * "내부 서명" field in its evidence bundles (PROJECT_DESIGN.md §3-7),
 * replacing the previously-always-null placeholder (see
 * apps/detection-svc/src/evidence_bundle.py's former docstring).
 */
export function internalRouter() {
  const router = Router();

  router.post("/internal/sign-evidence", async (req, res) => {
    const payload = req.body;
    if (payload === undefined || payload === null || typeof payload !== "object") {
      return res.status(400).json({ error: "body must be a JSON object (the evidence bundle minus its signature field)" });
    }
    try {
      const { signature, publicKeyPem } = await signEvidencePayload(payload);
      res.json({ signature, publicKeyPem, algorithm: "ed25519" });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : "signing failed" });
    }
  });

  return router;
}
