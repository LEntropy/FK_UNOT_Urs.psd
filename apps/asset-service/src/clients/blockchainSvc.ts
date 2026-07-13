import { env } from "../env.js";

/** Matches apps/blockchain-svc/INTEGRATION.md's contract exactly. */
export interface RegisterAssetRequest {
  ownerAddress: string;
  perceptualHash: string;
  metadataHash: string;
  doNotTrain: boolean;
}

export interface RegisterAssetResult {
  contentHash: string;
  ownerAddress: string;
  doNotTrain: boolean;
  txHash: string;
  blockNumber: number;
}

export interface VerifyAssetResult {
  contentHash: string;
  exists: boolean;
  owner: string | null;
  timestamp: number | null;
  doNotTrain: boolean | null;
}

/**
 * 409 (already registered) is a distinct, expected outcome per
 * blockchain-svc/INTEGRATION.md -- not necessarily an error. Callers should
 * check `alreadyRegistered` and decide (idempotent success if the owner
 * matches, a real conflict to flag otherwise) rather than treating this as
 * a thrown exception.
 */
export class AlreadyRegisteredError extends Error {
  constructor(public contentHash: string) {
    super(`content hash ${contentHash} already registered`);
  }
}

export async function registerAsset(req: RegisterAssetRequest): Promise<RegisterAssetResult> {
  const res = await fetch(`${env.BLOCKCHAIN_SVC_URL}/assets/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (res.status === 409) {
    const body = await res.json();
    throw new AlreadyRegisteredError(body.contentHash);
  }
  if (res.status !== 201) {
    throw new Error(`blockchain-svc POST /assets/register failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

export async function verifyAsset(contentHash: string): Promise<VerifyAssetResult> {
  const res = await fetch(`${env.BLOCKCHAIN_SVC_URL}/assets/verify/${contentHash}`);
  if (!res.ok) {
    throw new Error(`blockchain-svc GET /assets/verify failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}
