import cors from "cors";
import express from "express";
import type { Db } from "./db/client.js";
import { authRouter } from "./routes/auth.js";
import { meRouter } from "./routes/me.js";
import { artworksRouter } from "./routes/artworks.js";

export function createApp(db: Db) {
  const app = express();
  // apps/web runs on a different origin (Vite dev server / static server)
  // than this gateway -- this is the only browser-facing service in the
  // stack, so it's the only one that needs CORS.
  app.use(cors());
  app.use(express.json());

  app.get("/health", (_req, res) => res.json({ status: "ok" }));
  app.use("/auth", authRouter(db));
  app.use("/me", meRouter(db));
  app.use("/artworks", artworksRouter());

  return app;
}
