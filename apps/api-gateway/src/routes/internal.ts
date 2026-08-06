import { Router } from "express";
import { z } from "zod";
import { eq } from "drizzle-orm";
import type { Db } from "../db/client.js";
import { users } from "../db/schema.js";
import { signEvidencePayload } from "../evidenceSigning.js";
import { sendEvidenceReadyEmail } from "../notify.js";

/**
 * Internal-only endpoints (same trust boundary as delivery-gateway's own
 * /internal/sign -- network-level, not app-level auth, per that service's
 * README) -- detection-svc calls these for two things it can't do itself:
 * getting a real signature for evidence bundles' "내부 서명" field
 * (/internal/sign-evidence, PROJECT_DESIGN.md §3-7), and emailing a
 * creator once a case reaches EVIDENCE_READY (/internal/notify-evidence-ready
 * -- api-gateway owns the users table/email, detection-svc never has it).
 */
export function internalRouter(db: Db) {
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

  const notifySchema = z.object({
    creatorId: z.string(),
    artworkTitle: z.string(),
    caseId: z.string(),
    evidenceType: z.enum(["copy", "model_leak"]),
  });

  router.post("/internal/notify-evidence-ready", async (req, res) => {
    const parsed = notifySchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    const user = db.select().from(users).where(eq(users.id, parsed.data.creatorId)).get();
    if (!user) {
      // Not a client error -- detection-svc has no way to know a creator
      // was deleted/never existed before asking. Same "sent: false, still
      // 200" shape as SMTP-not-configured, so this best-effort caller
      // never needs a try/except for this specific case either.
      return res.json({ sent: false });
    }

    const sent = await sendEvidenceReadyEmail({
      toEmail: user.email,
      artworkTitle: parsed.data.artworkTitle,
      caseId: parsed.data.caseId,
      evidenceType: parsed.data.evidenceType,
    });
    res.json({ sent });
  });

  return router;
}
