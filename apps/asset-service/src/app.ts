import express from "express";
import type { Db } from "./db/client.js";
import { artworksRouter } from "./routes/artworks.js";

export function createApp(db: Db) {
  const app = express();
  app.use(express.json());

  app.get("/health", (_req, res) => res.json({ status: "ok" }));
  app.use("/artworks", artworksRouter(db));

  return app;
}
