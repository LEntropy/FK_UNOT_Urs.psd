import { Router } from "express";
import { eq } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import { users } from "../db/schema.js";
import { requireAuth } from "../middleware/requireAuth.js";

// Settings-page profile edit (2026-08-14). displayName/bio are the only
// fields a user can change about themselves here -- handle/email/wallet
// stay fixed (handle is a unique identity anchor other users link/@-
// mention by, email/wallet are account-recovery/custody concerns, not
// profile presentation). 160 chars mirrors X's bio cap, a familiar length
// rather than an arbitrary one.
const updateMeSchema = z.object({
  displayName: z.string().trim().min(1).max(50).optional(),
  bio: z.string().trim().max(160).optional(),
});

export function meRouter(db: Db): Router {
  const router = Router();

  router.get("/", requireAuth, (req, res) => {
    const user = db.select().from(users).where(eq(users.id, req.user!.sub)).get();
    if (!user) {
      return res.status(404).json({ error: "user not found" });
    }
    const { passwordHash, encryptedWalletKey, ...safe } = user;
    res.json(safe);
  });

  router.patch("/", requireAuth, (req, res) => {
    const parsed = updateMeSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    if (Object.keys(parsed.data).length === 0) {
      return res.status(400).json({ error: "no fields to update" });
    }

    db.update(users).set(parsed.data).where(eq(users.id, req.user!.sub)).run();

    const user = db.select().from(users).where(eq(users.id, req.user!.sub)).get()!;
    const { passwordHash, encryptedWalletKey, ...safe } = user;
    res.json(safe);
  });

  return router;
}
