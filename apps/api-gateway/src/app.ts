import cors from "cors";
import express from "express";
import type { Db } from "./db/client.js";
import { authRouter } from "./routes/auth.js";
import { oauthRouter } from "./routes/oauth.js";
import { meRouter } from "./routes/me.js";
import { artworksRouter } from "./routes/artworks.js";
import { communityRouter } from "./routes/community.js";
import { internalRouter } from "./routes/internal.js";
import { detectionRouter } from "./routes/detection.js";

export function createApp(db: Db) {
  const app = express();
  // apps/web runs on a different origin (Vite dev server / static server)
  // than this gateway -- this is the only browser-facing service in the
  // stack, so it's the only one that needs CORS.
  app.use(cors());
  app.use(express.json());

  app.get("/health", (_req, res) => res.json({ status: "ok" }));
  app.use("/auth", authRouter(db));
  app.use("/auth", oauthRouter(db)); // /auth/google, /auth/kakao (+ /callback)
  app.use("/me", meRouter(db));
  app.use("/artworks", artworksRouter());
  // internalRouter has no auth of its own (network-level trust boundary,
  // see its own doc) -- must be mounted before detectionRouter/
  // communityRouter below, both of which apply requireAuth via router.use()
  // with no path prefix. That runs unconditionally for *every* request that
  // reaches the router, matched route or not, so mounting either of them
  // first would 401 /internal/sign-evidence before it ever reached its own
  // handler (a real bug, caught by evidenceSigning.test.ts failing once
  // detectionRouter was briefly mounted ahead of this).
  app.use(internalRouter(db));
  // detectionRouter owns /artworks/:id/scan, /artworks/:id/report, and
  // /detection-cases/:id/... -- mounted at root (like communityRouter
  // below) since it adds sub-paths under /artworks rather than owning the
  // whole prefix.
  app.use(detectionRouter());
  // communityRouter registers its own /artworks/:id/..., /users/:id/...,
  // /feed, /me/..., /collections, /moderation sub-paths -- mounted at root
  // since it owns multiple top-level prefixes, not just one (same reason as
  // asset-service's own community router). Must stay last among the
  // no-path-prefix routers for the same reason internalRouter had to come
  // first: its own requireAuth would otherwise swallow anything mounted
  // after it.
  app.use(communityRouter());

  return app;
}
