import express from "express";
import { registerRouter } from "./routes/register.js";
import { verifyRouter } from "./routes/verify.js";
import { getRelayerBalance } from "./relayerBalance.js";

export function createApp() {
  const app = express();
  app.use(express.json());

  app.get("/health", (_req, res) => res.json({ status: "ok" }));
  // Queryable relayer balance (src/relayerBalance.ts) -- separate from
  // /health on purpose: it makes a live RPC call, so it shouldn't be on
  // the hot path of a healthcheck polled every few seconds (see
  // docker-compose.yml's healthcheck intervals elsewhere in this repo).
  app.get("/relayer/balance", async (_req, res) => {
    try {
      res.json(await getRelayerBalance());
    } catch (err) {
      res.status(502).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });
  app.use("/assets/register", registerRouter);
  app.use("/assets/verify", verifyRouter);

  return app;
}
