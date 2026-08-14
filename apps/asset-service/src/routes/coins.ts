import { Router } from "express";
import { desc, eq } from "drizzle-orm";
import type { Db } from "../db/client.js";
import { coinTransactions } from "../db/schema.js";
import { getOrCreateBalance } from "../coins.js";

export function coinsRouter(db: Db): Router {
  const router = Router();

  // No auth of its own (same trust boundary as artworksRouter -- api-gateway
  // is where the JWT gets verified and userId gets attached). Lazily grants
  // the signup bonus on first touch, see coins.ts's module doc.
  router.get("/coins/balance", (req, res) => {
    const userId = typeof req.query.userId === "string" ? req.query.userId : undefined;
    if (!userId) {
      return res.status(400).json({ error: "userId query param required" });
    }

    const balance = getOrCreateBalance(db, userId);
    const recentTransactions = db
      .select()
      .from(coinTransactions)
      .where(eq(coinTransactions.userId, userId))
      .orderBy(desc(coinTransactions.createdAt))
      .limit(20)
      .all();

    res.json({ balance, recentTransactions });
  });

  return router;
}
