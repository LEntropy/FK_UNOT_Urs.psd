import { randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { unwrapKey, wrapKey, KmsProtocolError } from "../src/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_KEY_PATH = path.join(here, "fixtures", "teamA1_key_v1_pub.pem");
const CA_CERT_PATH = path.join(here, "fixtures", "kms_ca.crt");

const KMS_HOST = process.env.KMS_HOST ?? "Philosophyz.iptime.org";
const KMS_PORT = Number(process.env.KMS_PORT ?? 8443);

// The live KMS server; policy.conf grants "teamA/teamA1" org access to its own "key_v1".
describe.skipIf(process.env.SKIP_KMS_INTEGRATION === "1")("kms-adapter roundtrip (live server)", () => {
  it("wraps a random key client-side and unwraps it back via the KMS server", async () => {
    const plainKey = randomBytes(32); // e.g. a per-artwork DEK

    const wrapped = wrapKey(PUBLIC_KEY_PATH, plainKey);
    expect(wrapped.length).toBeGreaterThan(0);
    expect(wrapped.equals(plainKey)).toBe(false);

    const unwrapped = await unwrapKey({
      host: KMS_HOST,
      port: KMS_PORT,
      caCertPath: CA_CERT_PATH,
      requesterOrg: "teamA/teamA1",
      fileOrg: "teamA/teamA1",
      keyId: "key_v1",
      encKey: wrapped,
    });

    expect(unwrapped.equals(plainKey)).toBe(true);
  });

  it("rejects a requester without access to the target org", async () => {
    const plainKey = randomBytes(32);
    const wrapped = wrapKey(PUBLIC_KEY_PATH, plainKey);

    await expect(
      unwrapKey({
        host: KMS_HOST,
        port: KMS_PORT,
        caCertPath: CA_CERT_PATH,
        requesterOrg: "teamB",
        fileOrg: "teamA/teamA1",
        keyId: "key_v1",
        encKey: wrapped,
      }),
    ).rejects.toThrow(KmsProtocolError);
  });
});
