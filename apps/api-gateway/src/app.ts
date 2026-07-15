import cors from "cors";
import express from "express";
import type { Db } from "./db/client.js";
import { authRouter } from "./routes/auth.js";
import { oauthRouter } from "./routes/oauth.js";
import { meRouter } from "./routes/me.js";
import { artworksRouter } from "./routes/artworks.js";
import { communityRouter } from "./routes/community.js";

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
  // communityRouter registers its own /artworks/:id/..., /users/:id/...,
  // /feed, /me/..., /collections, /moderation sub-paths -- mounted at root
  // since it owns multiple top-level prefixes, not just one (same reason as
  // asset-service's own community router).
  app.use(communityRouter());

  return app;
}
