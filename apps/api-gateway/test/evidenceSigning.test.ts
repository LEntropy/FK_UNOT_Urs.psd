import { verify as cryptoVerify } from "node:crypto";
import { describe, expect, it } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

// Uses the real EVIDENCE_SIGNING_PRIVATE_KEY dev key from .env (loaded via
// dotenv/config in src/env.ts) -- real Ed25519 signing, not mocked. KMS's
// own unwrapKey round-trip is already proven separately by
// infra/kms-adapter's roundtrip.test.ts; this only needs to prove
// signEvidencePayload/the route produce a verifiable signature and are
// wired together correctly, which doesn't need the live KMS server.

describe("POST /internal/sign-evidence", () => {
  it("400s on a non-object body", async () => {
    const app = createApp(createTestDb());
    // Explicit content-type + a JSON-encoded string literal so express.json()
    // actually parses req.body as the string "just a string", not {} --
    // supertest's .send() with a plain string arg (no content-type override)
    // defaults to a content-type body-parser won't touch as JSON at all.
    const res = await request(app).post("/internal/sign-evidence").set("Content-Type", "application/json").send('"just a string"');
    expect(res.status).toBe(400);
  });

  it("returns a signature that verifies against the returned public key, over the exact content sent", async () => {
    const app = createApp(createTestDb());
    // Flat (non-nested) on purpose: JSON.stringify's array-replacer form
    // used below to reconstruct the expected canonical bytes only applies
    // its key allowlist at every nesting level, so it isn't a faithful
    // stand-in for evidenceSigning.ts's own recursive canonicalize() for
    // nested objects -- keeping this fixture flat avoids that mismatch
    // rather than reimplementing that function a second time here.
    const bundle = {
      originalHash: "abc123",
      protectedHash: "abc123",
      rightsHolder: "0xDEADBEEF",
      isMatch: true,
    };

    const res = await request(app).post("/internal/sign-evidence").send(bundle);

    expect(res.status).toBe(200);
    expect(res.body.algorithm).toBe("ed25519");
    expect(typeof res.body.signature).toBe("string");
    expect(res.body.publicKeyPem).toContain("BEGIN PUBLIC KEY");

    // Verifies the signature is real (not a stub) and actually covers the
    // bundle's content, not some unrelated fixed string.
    const canonical = JSON.stringify(bundle, Object.keys(bundle).sort());
    const isValid = cryptoVerify(null, Buffer.from(canonical, "utf8"), res.body.publicKeyPem, Buffer.from(res.body.signature, "base64"));
    expect(isValid).toBe(true);
  });

  it("produces a different signature for different content signed with the same key", async () => {
    const app = createApp(createTestDb());
    const res1 = await request(app).post("/internal/sign-evidence").send({ a: 1 });
    const res2 = await request(app).post("/internal/sign-evidence").send({ a: 2 });

    expect(res1.body.signature).not.toBe(res2.body.signature);
    expect(res1.body.publicKeyPem).toBe(res2.body.publicKeyPem); // same key, cached across requests
  });

  it("produces the same signature for the same content regardless of key insertion order", async () => {
    const app = createApp(createTestDb());
    const res1 = await request(app).post("/internal/sign-evidence").send({ a: 1, b: 2 });
    const res2 = await request(app).post("/internal/sign-evidence").send({ b: 2, a: 1 });

    expect(res1.body.signature).toBe(res2.body.signature);
  });
});
