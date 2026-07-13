import { Router } from "express";
import { isAddress, isHexString, keccak256, toUtf8Bytes } from "ethers";
import { z } from "zod";
import { registry } from "../contract.js";
import { computeContentHash } from "../hash.js";

const hash32 = () => z.string().refine((v) => isHexString(v, 32), "must be a 0x-prefixed 32-byte hex string");

const bodySchema = z
  .object({
    ownerAddress: z.string().refine(isAddress, "not a valid address"),
    doNotTrain: z.boolean().default(false),
    // Production path: protection-svc has already computed both hashes —
    // this recomputes contentHash with the canonical formula (hash.ts).
    perceptualHash: hash32().optional(),
    metadataHash: hash32().optional(),
    // Escape hatch: pass a precomputed 32-byte content hash directly.
    contentHash: hash32().optional(),
    // Convenience for manual/curl testing only — hashes an arbitrary string.
    // Do not use this from protection-svc/asset-service; it does not follow
    // the pHash‖metadataHash formula other services expect.
    content: z.string().min(1).optional(),
  })
  .refine((v) => v.contentHash || v.content || (v.perceptualHash && v.metadataHash), {
    message: "provide contentHash, content, or both perceptualHash and metadataHash",
  });

export const registerRouter = Router();

registerRouter.post("/", async (req, res) => {
  const parsed = bodySchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }

  const { ownerAddress, doNotTrain } = parsed.data;
  const contentHash =
    parsed.data.contentHash
    ?? (parsed.data.perceptualHash && parsed.data.metadataHash
      ? computeContentHash(parsed.data.perceptualHash, parsed.data.metadataHash)
      : keccak256(toUtf8Bytes(parsed.data.content!)));

  try {
    const tx = await registry.registerFor(ownerAddress, contentHash, doNotTrain);
    const receipt = await tx.wait();

    res.status(201).json({
      contentHash,
      ownerAddress,
      doNotTrain,
      txHash: receipt.hash,
      blockNumber: receipt.blockNumber,
    });
  } catch (err) {
    const errorData = (err as { data?: string; error?: { data?: string } })?.data
      ?? (err as { error?: { data?: string } })?.error?.data;
    const revertName = errorData ? registry.interface.parseError(errorData)?.name : undefined;

    if (revertName === "AlreadyRegistered") {
      return res.status(409).json({ error: "content hash already registered", contentHash });
    }
    const message = err instanceof Error ? err.message : "unknown error";
    res.status(502).json({ error: "on-chain registration failed", detail: message });
  }
});
