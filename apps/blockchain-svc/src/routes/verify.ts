import { Router } from "express";
import { isHexString, keccak256, toUtf8Bytes } from "ethers";
import { registry } from "../contract.js";

export const verifyRouter = Router();

/**
 * GET /assets/verify/:hashOrContent
 * If the path segment is a 0x-prefixed 32-byte hex string, it's used as the
 * content hash directly. Otherwise it's treated as raw content and hashed
 * with keccak256 first (convenient for quick curl/browser testing).
 */
verifyRouter.get("/:hashOrContent", async (req, res) => {
  const raw = req.params.hashOrContent;
  const contentHash = isHexString(raw, 32) ? raw : keccak256(toUtf8Bytes(raw));

  try {
    const [exists, owner, timestamp, doNotTrain] = await registry.verify(contentHash);

    res.json({
      contentHash,
      exists,
      owner: exists ? owner : null,
      timestamp: exists ? Number(timestamp) : null,
      doNotTrain: exists ? doNotTrain : null,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown error";
    res.status(502).json({ error: "on-chain lookup failed", detail: message });
  }
});
