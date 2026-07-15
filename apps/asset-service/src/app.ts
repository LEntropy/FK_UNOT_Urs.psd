import express from "express";
import type { Db } from "./db/client.js";
import { artworksRouter } from "./routes/artworks.js";
import { communityRouter } from "./routes/community.js";

export function createApp(db: Db) {
  const app = express();
  app.use(express.json());

  app.get("/health", (_req, res) => res.json({ status: "ok" }));
  app.use("/artworks", artworksRouter(db));
  // communityRouter registers its own /artworks/:id/... and /users/:id/...
  // sub-paths, /feed, /collections, /moderation -- mounted at root since it
  // owns multiple top-level prefixes, not just one.
  app.use(communityRouter(db));

  return app;
}
