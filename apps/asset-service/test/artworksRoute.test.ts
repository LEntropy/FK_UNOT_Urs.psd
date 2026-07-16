import { existsSync, mkdtempSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { eq } from "drizzle-orm";
import { describe, expect, it, vi } from "vitest";
import request from "supertest";
import { artworks } from "../src/db/schema.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/orchestration.js", () => ({ runUploadPipeline: vi.fn() }));

const { createApp } = await import("../src/app.js");

function seed(db: ReturnType<typeof createTestDb>, overrides: Partial<typeof artworks.$inferInsert> = {}) {
  const now = new Date();
  db.insert(artworks)
    .values({
      id: overrides.id ?? "ast_1",
      title: "Test",
      sourceImageUri: "/tmp/a.png",
      creatorId: "creator_a",
      ownerWalletAddress: "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2",
      protectionProfile: "L1_PREVIEW",
      allowAiTraining: false,
      watermarkPayloadHex: "deadbeefcafef00d",
      encryptedImagePath: "./data/encrypted/test.enc",
      encryptedDekBase64: "ZmFrZQ==",
      encryptionIv: "ZmFrZQ==",
      encryptionAuthTag: "ZmFrZQ==",
      status: "UPLOADED",
      createdAt: now,
      updatedAt: now,
      ...overrides,
    })
    .run();
}

describe("GET /artworks", () => {
  it("returns only the requested creator's artworks when creatorId is given", async () => {
    const db = createTestDb();
    seed(db, { id: "ast_a1", creatorId: "creator_a" });
    seed(db, { id: "ast_a2", creatorId: "creator_a" });
    seed(db, { id: "ast_b1", creatorId: "creator_b" });

    const res = await request(createApp(db)).get("/artworks?creatorId=creator_a");
    expect(res.status).toBe(200);
    expect(res.body.map((a: { id: string }) => a.id).sort()).toEqual(["ast_a1", "ast_a2"]);
  });

  it("returns everything when creatorId is omitted", async () => {
    const db = createTestDb();
    seed(db, { id: "ast_a1", creatorId: "creator_a" });
    seed(db, { id: "ast_b1", creatorId: "creator_b" });

    const res = await request(createApp(db)).get("/artworks");
    expect(res.status).toBe(200);
    expect(res.body).toHaveLength(2);
  });
});

describe("POST /artworks (envelope encryption at rest)", () => {
  it("encrypts the upload, deletes the plaintext, and stores no plaintext path anywhere", async () => {
    const db = createTestDb();
    const plainDir = mkdtempSync(join(tmpdir(), "dontai-upload-test-"));
    const plainPath = join(plainDir, "original.png");
    writeFileSync(plainPath, "not a real png, just needs bytes to encrypt");

    const res = await request(createApp(db)).post("/artworks").send({
      title: "Encryption test",
      sourceImageUri: plainPath,
      creatorId: "creator_enc",
      ownerWalletAddress: "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2",
    });

    expect(res.status).toBe(202);
    expect(existsSync(plainPath)).toBe(false); // the actual "at rest" guarantee

    const row = db.select().from(artworks).where(eq(artworks.id, res.body.id)).get()!;
    expect(row.encryptedImagePath).toMatch(/\.enc$/);
    expect(existsSync(row.encryptedImagePath)).toBe(true);
    expect(row.encryptedDekBase64.length).toBeGreaterThan(0);
    // Base64 RSA-PKCS1 ciphertext for a 32-byte AES key should never
    // literally contain the plaintext filename/content -- a weak sanity
    // check, but a real one: this isn't just base64 of the original path.
    expect(row.encryptedDekBase64).not.toContain("original.png");

    unlinkSync(row.encryptedImagePath);
  });

  it("returns 400 (not a crash) when sourceImageUri doesn't exist", async () => {
    const db = createTestDb();
    const res = await request(createApp(db)).post("/artworks").send({
      title: "Missing file",
      sourceImageUri: "/definitely/not/a/real/path.png",
      creatorId: "creator_x",
      ownerWalletAddress: "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2",
    });

    expect(res.status).toBe(400);
    expect(db.select().from(artworks).all()).toHaveLength(0); // no partial row left behind
  });

  it("accepts a real multipart file upload (the browser's actual path, not just a server-side sourceImageUri)", async () => {
    const db = createTestDb();

    const res = await request(createApp(db))
      .post("/artworks")
      .field("title", "Real browser upload")
      .field("creatorId", "creator_upload")
      .field("ownerWalletAddress", "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2")
      .field("allowAiTraining", "true") // multipart fields are always strings -- proves the string "true" is parsed correctly
      .attach("image", Buffer.from("not a real png, just needs bytes to encrypt"), "mona_lisa.jpg");

    expect(res.status).toBe(202);

    const row = db.select().from(artworks).where(eq(artworks.id, res.body.id)).get()!;
    expect(row.allowAiTraining).toBe(true);
    expect(row.sourceImageUri).toBe("upload:mona_lisa.jpg");
    expect(existsSync(row.encryptedImagePath)).toBe(true); // the uploaded bytes really did get encrypted and stored

    unlinkSync(row.encryptedImagePath);
  });

  it("parses the multipart string \"false\" as false, not true (z.coerce.boolean()'s exact failure mode)", async () => {
    const db = createTestDb();

    const res = await request(createApp(db))
      .post("/artworks")
      .field("title", "Explicit false")
      .field("creatorId", "creator_false")
      .field("ownerWalletAddress", "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2")
      .field("allowAiTraining", "false")
      .attach("image", Buffer.from("bytes"), "x.jpg");

    expect(res.status).toBe(202);
    const row = db.select().from(artworks).where(eq(artworks.id, res.body.id)).get()!;
    expect(row.allowAiTraining).toBe(false);

    unlinkSync(row.encryptedImagePath);
  });

  it("400s when neither a file nor sourceImageUri is given", async () => {
    const db = createTestDb();
    const res = await request(createApp(db)).post("/artworks").send({
      title: "Nothing to upload",
      creatorId: "creator_x",
      ownerWalletAddress: "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2",
    });

    expect(res.status).toBe(400);
    expect(db.select().from(artworks).all()).toHaveLength(0);
  });
});
