import { concat, isHexString, keccak256 } from "ethers";

/**
 * Canonical on-chain content hash formula (PROJECT_DESIGN.md §5-1):
 *
 *   contentHash = keccak256(perceptualHash ‖ metadataHash)
 *
 * protection-svc computes `perceptualHash` (pHash of the protected/public
 * image variant) and `metadataHash` (hash of title/creator/license fields),
 * then calls this to get the exact value blockchain-svc's /assets/register
 * and OwnershipRegistry.verify() expect. Any other service that needs to
 * recompute this hash (for verification, dispute resolution, etc.) MUST use
 * the same byte-concatenation order — a mismatch here silently produces a
 * different token, not an error.
 *
 * Both inputs must already be 32-byte hex hashes (0x + 64 hex chars) —
 * e.g. the output of a pHash algorithm padded/hashed down to 32 bytes, and
 * keccak256 of the canonical JSON metadata string, respectively.
 */
export function computeContentHash(perceptualHash: string, metadataHash: string): string {
  if (!isHexString(perceptualHash, 32)) {
    throw new Error("perceptualHash must be a 0x-prefixed 32-byte hex string");
  }
  if (!isHexString(metadataHash, 32)) {
    throw new Error("metadataHash must be a 0x-prefixed 32-byte hex string");
  }
  return keccak256(concat([perceptualHash, metadataHash]));
}
