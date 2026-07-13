import express from "express";
import { registerRouter } from "./routes/register.js";
import { verifyRouter } from "./routes/verify.js";

export function createApp() {
  const app = express();
  app.use(express.json());

  app.get("/health", (_req, res) => res.json({ status: "ok" }));
  app.use("/assets/register", registerRouter);
  app.use("/assets/verify", verifyRouter);

  return app;
}
