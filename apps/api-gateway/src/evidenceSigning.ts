import { createPrivateKey, createPublicKey, sign as cryptoSign, type KeyObject } from "node:crypto";
import { unwrapKey } from "@dontai/kms-adapter";
import { env } from "./env.js";

/**
 * Evidence-bundle signing (PROJECT_DESIGN.md §3-7's "내부 서명" field). Same
 * envelope-encryption pattern as blockchain-svc/src/contract.ts's relayer
 * key -- the real KMS C server only implements envelope-key decrypt, no
 * Sign() RPC, so this resolves a dedicated Ed25519 private key (either
 * KMS-wrapped ciphertext, decrypted once here, or a plaintext dev key) and
 * signs with it locally. Cached at module scope so the KMS round-trip (or
 * PEM parse) only happens once per process, not once per evidence bundle.
 */
let cachedKeyPromise: Promise<KeyObject> | null = null;

async function resolveSigningKey(): Promise<KeyObject> {
  if (env.EVIDENCE_SIGNING_PRIVATE_KEY) {
    const pem = Buffer.from(env.EVIDENCE_SIGNING_PRIVATE_KEY, "base64").toString("utf8");
    return createPrivateKey({ key: pem, format: "pem" });
  }

  const plainKeyDer = await unwrapKey({
    host: env.KMS_HOST,
    port: env.KMS_PORT,
    caCertPath: env.KMS_CA_CERT_PATH,
    requesterOrg: env.KMS_ORG,
    fileOrg: env.KMS_ORG,
    keyId: env.KMS_KEY_ID,
    encKey: Buffer.from(env.EVIDENCE_SIGNING_ENCRYPTED_KEY!, "base64"),
  });
  // Wrapped key is the raw PKCS8 DER bytes of the Ed25519 private key (see
  // scripts/generate-evidence-signing-key.mjs, which produces exactly this
  // format before wrapping) -- no PEM headers to strip, unlike the relayer
  // key's raw-hex-private-key convention.
  return createPrivateKey({ key: plainKeyDer, format: "der", type: "pkcs8" });
}

function getSigningKey(): Promise<KeyObject> {
  if (!cachedKeyPromise) {
    cachedKeyPromise = resolveSigningKey();
  }
  return cachedKeyPromise;
}

/**
 * Canonicalizes an evidence bundle's fields (sorted keys, stable
 * stringification) and signs the UTF-8 bytes with Ed25519. detection-svc
 * sends every field except `signature` itself (which doesn't exist yet at
 * signing time) -- see routes/internal.ts's /internal/sign-evidence.
 */
export async function signEvidencePayload(payload: unknown): Promise<{ signature: string; publicKeyPem: string }> {
  const key = await getSigningKey();
  const canonical = canonicalize(payload);
  const signature = cryptoSign(null, Buffer.from(canonical, "utf8"), key); // null digest: Ed25519 signs the message directly
  const publicKeyPem = createPublicKey(key).export({ type: "spki", format: "pem" }).toString();
  return { signature: signature.toString("base64"), publicKeyPem };
}

// Sorted-keys JSON stringify so the same logical bundle always signs to the
// same bytes regardless of the caller's own key insertion order (JSON.stringify
// alone is order-dependent and would make signatures fragile to unrelated
// refactors on the Python side).
function canonicalize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalize((value as Record<string, unknown>)[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
