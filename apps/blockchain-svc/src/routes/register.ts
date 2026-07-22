import { Router } from "express";
import { AbiCoder, isAddress, isHexString, keccak256, randomBytes, toUtf8Bytes } from "ethers";
import { z } from "zod";
import { provider, registry, relayerWallet } from "../contract.js";
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

// Matches OwnershipRegistry.sol's registerFor commitHash formula exactly:
// keccak256(abi.encode(contentHash, doNotTrain, owner, msg.sender, nonce)).
// abi.encode (not encodePacked) on both sides so there's no ambiguity about
// how the fixed-size fields are laid out.
const abiCoder = AbiCoder.defaultAbiCoder();
function computeCommitHash(contentHash: string, doNotTrain: boolean, ownerAddress: string, nonce: string): string {
  const encoded = abiCoder.encode(
    ["bytes32", "bool", "address", "address", "bytes32"],
    [contentHash, doNotTrain, ownerAddress, relayerWallet.address, nonce],
  );
  return keccak256(encoded);
}

// Polls for the commit to reach MIN_COMMIT_AGE instead of a fixed sleep --
// block time varies (faster on local anvil in tests, ~2s on real Amoy), and
// a fixed sleep would either be wastefully long or occasionally too short.
async function waitForCommitToMature(commitBlockNumber: number): Promise<void> {
  const minCommitAge = Number(await registry.MIN_COMMIT_AGE());
  const readyAtBlock = commitBlockNumber + minCommitAge;
  while ((await provider.getBlockNumber()) < readyAtBlock) {
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

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
    // Step 1/2 (commit): only the hash goes on-chain here, hiding
    // contentHash from the mempool until reveal -- see contract.ts's
    // REGISTRY_ABI comment for why this exists.
    const nonce = `0x${Buffer.from(randomBytes(32)).toString("hex")}`;
    const commitHash = computeCommitHash(contentHash, doNotTrain, ownerAddress, nonce);
    const commitTx = await registry.commit(commitHash);
    const commitReceipt = await commitTx.wait();

    await waitForCommitToMature(commitReceipt.blockNumber);

    // Step 2/2 (reveal): the real registration, now safe to broadcast in
    // plaintext since any copy-cat committing upon seeing this transaction
    // in the mempool can't reveal until after this one is already mined.
    const tx = await registry.registerFor(ownerAddress, contentHash, doNotTrain, nonce);
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
    // Real crash found live: some provider errors (e.g. "insufficient
    // funds" from an RPC node) carry a *present but empty* `data: "0x"`
    // field rather than omitting it -- `errorData ? ... : undefined` still
    // treated that as "parse this," and parseError() throws (not returns
    // undefined) on anything shorter than a 4-byte selector, crashing the
    // whole process since this catch block itself wasn't the one guarding
    // against it. `errorData.length > 2` requires at least one real byte
    // beyond the "0x" prefix before attempting to parse.
    let revertName: string | undefined;
    if (errorData && errorData.length > 2) {
      try {
        revertName = registry.interface.parseError(errorData)?.name;
      } catch {
        revertName = undefined; // not one of this contract's own custom errors
      }
    }

    if (revertName === "AlreadyRegistered") {
      return res.status(409).json({ error: "content hash already registered", contentHash });
    }
    const message = err instanceof Error ? err.message : "unknown error";
    res.status(502).json({ error: "on-chain registration failed", detail: message });
  }
});
