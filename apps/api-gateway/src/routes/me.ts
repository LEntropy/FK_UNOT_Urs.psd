import { Router } from "express";
import { eq } from "drizzle-orm";
import type { Db } from "../db/client.js";
import { users } from "../db/schema.js";
import { requireAuth } from "../middleware/requireAuth.js";

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

  return router;
}
