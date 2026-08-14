import { existsSync, mkdtempSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { eq } from "drizzle-orm";
import { beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { artworks, assetVersions } from "../src/db/schema.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/orchestration.js", () => ({
  runUploadPipeline: vi.fn(),
  // Fire-and-forget in the real route (routes/artworks.ts's own note) --
  // defaults to an already-resolved promise so tests that don't care about
  // the cleanup path don't need to stub this themselves.
  persistScoreProtectionResult: vi.fn(() => Promise.resolve()),
}));
vi.mock("../src/clients/protectionSvc.js", () => ({
  suggestTags: vi.fn(),
  remeasureProtection: vi.fn(),
  createScoreProtectionJob: vi.fn(),
  getScoreProtectionJob: vi.fn(),
  // Fire-and-forget in the real route (see routes/artworks.ts's own note) --
  // defaults to an already-resolved promise so tests that don't care about
  // the cleanup path don't need to stub this themselves just to avoid a
  // synchronous "cannot read .catch of undefined" from the route's
  // `void pollScoreProtectionJob(jobId).catch(...).finally(...)` call.
  pollScoreProtectionJob: vi.fn(() => Promise.resolve({ status: "completed" })),
}));
// encryptImageAtRest stays real (existing upload tests exercise it directly,
// no live KMS server needed for wrapKey -- client-side only). Only
// decryptToTempFile/cleanupTempFile are mocked: decryptToTempFile makes a
// real unwrapKey network call to a live KMS server, which isn't available
// in this test run.
vi.mock("../src/crypto/imageEncryption.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/crypto/imageEncryption.js")>();
  return { ...actual, decryptToTempFile: vi.fn(), cleanupTempFile: vi.fn() };
});

const { createApp } = await import("../src/app.js");
const { suggestTags, remeasureProtection, createScoreProtectionJob, getScoreProtectionJob } =
  await import("../src/clients/protectionSvc.js");
const { decryptToTempFile, cleanupTempFile } = await import("../src/crypto/imageEncryption.js");
const { persistScoreProtectionResult } = await import("../src/orchestration.js");

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

  it("includes assetVersions per row -- GalleryPage/FeedPage can't render a thumbnail without this", async () => {
    const db = createTestDb();
    seed(db, { id: "ast_with_image", status: "PUBLISHED" });
    seed(db, { id: "ast_no_image", status: "PROTECTING" });
    db.insert(assetVersions)
      .values({
        artworkId: "ast_with_image",
        variantName: "grid_thumbnail_150",
        storageUri: "/tmp/thumb.png",
        width: 150,
        height: 150,
        scaleVsSource: 0.5,
        protectionStatus: "SAFE",
      })
      .run();

    const res = await request(createApp(db)).get("/artworks");
    expect(res.status).toBe(200);
    const withImage = res.body.find((a: { id: string }) => a.id === "ast_with_image");
    const noImage = res.body.find((a: { id: string }) => a.id === "ast_no_image");
    expect(withImage.assetVersions).toHaveLength(1);
    expect(withImage.assetVersions[0].variantName).toBe("grid_thumbnail_150");
    expect(noImage.assetVersions).toEqual([]);
  });

  it("returns tags as a real array, not the raw JSON-encoded column value", async () => {
    const db = createTestDb();
    seed(db, { id: "ast_tagged", tags: JSON.stringify(["가을", "수채화"]) });

    const res = await request(createApp(db)).get("/artworks");
    expect(res.status).toBe(200);
    expect(res.body[0].tags).toEqual(["가을", "수채화"]);
  });

  it("tag search matches any artwork with that exact tag, case-insensitively", async () => {
    const db = createTestDb();
    seed(db, {
      id: "ast_match",
      tags: JSON.stringify(["가을", "수채화"]),
      visibility: "public",
      status: "PUBLISHED",
      publishedAt: new Date(),
    });
    seed(db, { id: "ast_nomatch", tags: JSON.stringify(["여름"]), visibility: "public", status: "PUBLISHED", publishedAt: new Date() });

    const res = await request(createApp(db)).get("/artworks?tag=%EC%88%98%EC%B1%84%ED%99%94"); // "수채화"
    expect(res.status).toBe(200);
    expect(res.body.map((a: { id: string }) => a.id)).toEqual(["ast_match"]);
  });

  it("tag search only ever returns public+published rows, even without publicOnly explicitly set", async () => {
    const db = createTestDb();
    seed(db, { id: "ast_draft", tags: JSON.stringify(["수채화"]), visibility: "private", status: "UPLOADED" });

    const res = await request(createApp(db)).get("/artworks?tag=%EC%88%98%EC%B1%84%ED%99%94");
    expect(res.status).toBe(200);
    expect(res.body).toEqual([]);
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

  it("stores confirmed tags from a JSON-encoded multipart field", async () => {
    const db = createTestDb();

    const res = await request(createApp(db))
      .post("/artworks")
      .field("title", "Tagged upload")
      .field("creatorId", "creator_tags")
      .field("ownerWalletAddress", "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2")
      .field("tags", JSON.stringify(["oil painting", "portrait"]))
      .attach("image", Buffer.from("bytes"), "x.jpg");

    expect(res.status).toBe(202);
    const row = db.select().from(artworks).where(eq(artworks.id, res.body.id)).get()!;
    expect(JSON.parse(row.tags)).toEqual(["oil painting", "portrait"]);

    unlinkSync(row.encryptedImagePath);
  });

  it("defaults tags to an empty array when omitted", async () => {
    const db = createTestDb();

    const res = await request(createApp(db))
      .post("/artworks")
      .field("title", "No tags")
      .field("creatorId", "creator_notags")
      .field("ownerWalletAddress", "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2")
      .attach("image", Buffer.from("bytes"), "x.jpg");

    expect(res.status).toBe(202);
    const row = db.select().from(artworks).where(eq(artworks.id, res.body.id)).get()!;
    expect(JSON.parse(row.tags)).toEqual([]);

    unlinkSync(row.encryptedImagePath);
  });

  it("treats malformed tags JSON as no tags rather than 400ing the whole upload", async () => {
    const db = createTestDb();

    const res = await request(createApp(db))
      .post("/artworks")
      .field("title", "Bad tags field")
      .field("creatorId", "creator_badtags")
      .field("ownerWalletAddress", "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2")
      .field("tags", "not valid json[")
      .attach("image", Buffer.from("bytes"), "x.jpg");

    expect(res.status).toBe(202);
    const row = db.select().from(artworks).where(eq(artworks.id, res.body.id)).get()!;
    expect(JSON.parse(row.tags)).toEqual([]);

    unlinkSync(row.encryptedImagePath);
  });
});

describe("POST /artworks/suggest-tags", () => {
  it("400s without an image file", async () => {
    const db = createTestDb();
    const res = await request(createApp(db)).post("/artworks/suggest-tags").send({});
    expect(res.status).toBe(400);
  });

  it("forwards the uploaded file to protection-svc and returns its suggested tags", async () => {
    const db = createTestDb();
    vi.mocked(suggestTags).mockResolvedValue([
      { tag: "oil painting", score: 0.31 },
      { tag: "portrait", score: 0.28 },
    ]);

    const res = await request(createApp(db))
      .post("/artworks/suggest-tags")
      .attach("image", Buffer.from("bytes"), "preview.jpg");

    expect(res.status).toBe(200);
    expect(res.body.tags).toEqual([
      { tag: "oil painting", score: 0.31 },
      { tag: "portrait", score: 0.28 },
    ]);
    // The temp file passed to protection-svc was a real, readable path.
    expect(vi.mocked(suggestTags)).toHaveBeenCalledWith(expect.stringContaining("preview.jpg"));
  });

  it("deletes the temp upload file after the request completes, success or not", async () => {
    const db = createTestDb();
    let capturedPath = "";
    vi.mocked(suggestTags).mockImplementation(async (path) => {
      capturedPath = path;
      expect(existsSync(path)).toBe(true); // still there mid-request
      return [];
    });

    await request(createApp(db)).post("/artworks/suggest-tags").attach("image", Buffer.from("bytes"), "cleanup.jpg");

    expect(existsSync(capturedPath)).toBe(false);
  });

  it("returns 502 (not a crash) when protection-svc is unreachable", async () => {
    const db = createTestDb();
    vi.mocked(suggestTags).mockRejectedValue(new Error("connect ECONNREFUSED"));

    const res = await request(createApp(db))
      .post("/artworks/suggest-tags")
      .attach("image", Buffer.from("bytes"), "fail.jpg");

    expect(res.status).toBe(502);
  });
});

describe("POST /artworks/:id/remeasure-protection", () => {
  it("404s for an unknown artwork", async () => {
    const db = createTestDb();
    const res = await request(createApp(db)).post("/artworks/ast_missing/remeasure-protection");
    expect(res.status).toBe(404);
    expect(decryptToTempFile).not.toHaveBeenCalled();
  });

  it("400s when the artwork has no protected image yet", async () => {
    const db = createTestDb();
    seed(db); // status: UPLOADED, protectedImageUri unset

    const res = await request(createApp(db)).post("/artworks/ast_1/remeasure-protection");
    expect(res.status).toBe(400);
    expect(decryptToTempFile).not.toHaveBeenCalled();
  });

  it("decrypts the original, calls protection-svc against it and the real published image, and cleans up", async () => {
    const db = createTestDb();
    seed(db, { protectedImageUri: "/out/job_x/watermarked.png" });

    vi.mocked(decryptToTempFile).mockResolvedValue("/tmp/dontai-decrypted-ast_1.png");
    vi.mocked(remeasureProtection).mockResolvedValue({
      styleDriftScore: 0.09,
      styleSimilarityToOriginal: 0.87,
      perceptualPsnrDb: 31.2,
      perceptualRmse: 0.015,
    });

    const res = await request(createApp(db)).post("/artworks/ast_1/remeasure-protection");

    expect(res.status).toBe(200);
    expect(res.body).toEqual({
      styleDriftScore: 0.09,
      styleSimilarityToOriginal: 0.87,
      perceptualPsnrDb: 31.2,
      perceptualRmse: 0.015,
    });
    expect(remeasureProtection).toHaveBeenCalledWith(
      "/tmp/dontai-decrypted-ast_1.png",
      "/out/job_x/watermarked.png",
    );
    expect(cleanupTempFile).toHaveBeenCalledWith("/tmp/dontai-decrypted-ast_1.png");
  });

  it("still cleans up the decrypted temp file when protection-svc's re-measurement fails", async () => {
    const db = createTestDb();
    seed(db, { protectedImageUri: "/out/job_x/watermarked.png" });

    vi.mocked(decryptToTempFile).mockResolvedValue("/tmp/dontai-decrypted-ast_1.png");
    vi.mocked(remeasureProtection).mockRejectedValue(new Error("GPU PC unreachable"));

    const res = await request(createApp(db)).post("/artworks/ast_1/remeasure-protection");

    expect(res.status).toBe(502);
    expect(cleanupTempFile).toHaveBeenCalledWith("/tmp/dontai-decrypted-ast_1.png");
  });
});

describe("POST /artworks/:id/score-protection", () => {
  // This file's mocks are never auto-cleared between tests (no vitest
  // config `clearMocks`, unlike some projects' default) -- the preceding
  // remeasure-protection describe block above already exercised
  // decryptToTempFile several times, which would otherwise leak into this
  // block's own "not.toHaveBeenCalled()" assertions.
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(persistScoreProtectionResult).mockResolvedValue(undefined);
  });

  it("404s for an unknown artwork", async () => {
    const db = createTestDb();
    const res = await request(createApp(db)).post("/artworks/ast_missing/score-protection");
    expect(res.status).toBe(404);
    expect(decryptToTempFile).not.toHaveBeenCalled();
  });

  it("400s when strong_protection didn't actually run on this artwork", async () => {
    const db = createTestDb();
    seed(db, { protectedImageUri: "/out/job_x/watermarked.png" }); // usedStrongProtection defaults to false

    const res = await request(createApp(db)).post("/artworks/ast_1/score-protection");
    expect(res.status).toBe(400);
    expect(decryptToTempFile).not.toHaveBeenCalled();
  });

  it("kicks off a protection-svc job and returns its jobId immediately, without waiting for it to finish", async () => {
    const db = createTestDb();
    seed(db, {
      protectedImageUri: "/out/job_x/watermarked.png",
      usedStrongProtection: true,
      title: "낙엽",
      tags: JSON.stringify(["가을", "은행나무", "수채화"]),
    });

    vi.mocked(decryptToTempFile).mockResolvedValue("/tmp/dontai-decrypted-ast_1.png");
    vi.mocked(createScoreProtectionJob).mockResolvedValue({ jobId: "scorejob_abc", status: "queued" });

    const res = await request(createApp(db)).post("/artworks/ast_1/score-protection");

    expect(res.status).toBe(202);
    expect(res.body).toEqual({ jobId: "scorejob_abc" });
    // Prompt comes from the creator's own upload-time tags, not the title
    // -- protection_score.py needs a real content description to generate
    // toward, and a freeform title (or no tags at all) risks the baseline
    // LoRA collapsing toward the CLIP floor regardless of protection.
    expect(createScoreProtectionJob).toHaveBeenCalledWith({
      originalImageUri: "/tmp/dontai-decrypted-ast_1.png",
      protectedImageUri: "/out/job_x/watermarked.png",
      prompt: "가을, 은행나무, 수채화",
    });
    // Cleanup is deferred to the background poll, not this response --
    // persistScoreProtectionResult's default mock resolves immediately
    // (pure microtasks, no real I/O), so by the time supertest's request
    // settles the fire-and-forget .finally() has already run.
    expect(persistScoreProtectionResult).toHaveBeenCalledWith(expect.anything(), "ast_1", "scorejob_abc");
    expect(cleanupTempFile).toHaveBeenCalledWith("/tmp/dontai-decrypted-ast_1.png");
  });

  it("cleans up the decrypted temp file synchronously when job creation itself fails", async () => {
    const db = createTestDb();
    seed(db, { protectedImageUri: "/out/job_x/watermarked.png", usedStrongProtection: true });

    vi.mocked(decryptToTempFile).mockResolvedValue("/tmp/dontai-decrypted-ast_1.png");
    vi.mocked(createScoreProtectionJob).mockRejectedValue(new Error("RunPod endpoint unreachable"));

    const res = await request(createApp(db)).post("/artworks/ast_1/score-protection");

    expect(res.status).toBe(502);
    expect(cleanupTempFile).toHaveBeenCalledWith("/tmp/dontai-decrypted-ast_1.png");
    expect(persistScoreProtectionResult).not.toHaveBeenCalled();
  });
});

describe("GET /artworks/score-protection-jobs/:jobId", () => {
  it("proxies protection-svc's job status", async () => {
    const db = createTestDb();
    vi.mocked(getScoreProtectionJob).mockResolvedValue({
      status: "completed",
      sd15: { baselineSimilarity: 0.9, protectedSimilarity: 0.6, delta: 0.3, verdict: "PROTECTED", baselineSamples: [], protectedSamples: [] },
      sdxl: { baselineSimilarity: 0.9, protectedSimilarity: 0.85, delta: 0.05, verdict: "WEAK", baselineSamples: [], protectedSamples: [] },
      threshold: 0.03,
    });

    const res = await request(createApp(db)).get("/artworks/score-protection-jobs/scorejob_abc");

    expect(res.status).toBe(200);
    expect(res.body.status).toBe("completed");
    expect(res.body.sd15.verdict).toBe("PROTECTED");
    expect(getScoreProtectionJob).toHaveBeenCalledWith("scorejob_abc");
  });

  it("502s when protection-svc lookup fails", async () => {
    const db = createTestDb();
    vi.mocked(getScoreProtectionJob).mockRejectedValue(new Error("no job scorejob_missing"));

    const res = await request(createApp(db)).get("/artworks/score-protection-jobs/scorejob_missing");
    expect(res.status).toBe(502);
  });
});
