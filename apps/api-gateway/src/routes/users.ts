import { Router } from "express";
import { inArray, eq } from "drizzle-orm";
import type { Db } from "../db/client.js";
import { users } from "../db/schema.js";
import { requireAuth } from "../middleware/requireAuth.js";

/**
 * Public-facing profile lookups (2026-08-14, profile-page feature) --
 * distinct from routes/me.ts (which is always "the caller's own account").
 * Same requireAuth trust boundary as the rest of this app (App.tsx wraps
 * every page but /login|/signup|/oauth-callback|/terms in ProtectedRoute,
 * so there's no anonymous-browsing path yet to serve public profiles to
 * anyway) -- see communityRouter's own module doc for that reasoning.
 *
 * Deliberately thin: this only returns display identity (handle,
 * displayName, avatarUri, bio, role) -- never email/wallet/password hash,
 * regardless of whether the caller is looking at their own id or someone
 * else's (unlike GET /me, which is allowed to return more about yourself).
 */
function toPublicProfile(user: typeof users.$inferSelect) {
  return {
    id: user.id,
    handle: user.handle,
    displayName: user.displayName,
    avatarUri: user.avatarUri,
    bio: user.bio,
    role: user.role,
  };
}

export function usersRouter(db: Db): Router {
  const router = Router();
  router.use(requireAuth);

  // Batch lookup, e.g. to resolve a page of feed cards' or a follow list's
  // creatorIds into display names in one round trip instead of one request
  // per card. Silently drops ids that don't resolve to a user rather than
  // 404ing the whole batch -- a stale/deleted id in the list shouldn't sink
  // the other 19 that are still valid.
  router.get("/", (req, res) => {
    const idsParam = req.query.ids;
    const ids = typeof idsParam === "string" ? idsParam.split(",").filter(Boolean) : [];
    if (ids.length === 0) {
      return res.status(400).json({ error: "ids query param is required (comma-separated)" });
    }
    const rows = db.select().from(users).where(inArray(users.id, ids)).all();
    res.json(rows.map(toPublicProfile));
  });

  router.get("/:id", (req, res) => {
    const user = db.select().from(users).where(eq(users.id, req.params.id)).get();
    if (!user) {
      return res.status(404).json({ error: `no user ${req.params.id}` });
    }
    res.json(toPublicProfile(user));
  });

  return router;
}
