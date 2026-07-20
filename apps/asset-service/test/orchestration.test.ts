import { eq } from "drizzle-orm";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { artworks } from "../src/db/schema.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/clients/protectionSvc.js", () => ({
  createProtectJob: vi.fn(),
  pollProtectJob: vi.fn(),
}));

// decryptToTempFile needs a live KMS server (unwrapKey) -- mocked here since
// this file tests orchestration logic, not the encryption round-trip itself
// (that's src/crypto/imageEncryption.ts's own concern, covered by
// test/artworksRoute.test.ts's real wrapKey() + apps/asset-service's
// eventual decrypt-path test against the live server, same pattern as
// infra/kms-adapter's own roundtrip.test.ts).
vi.mock("../src/crypto/imageEncryption.js", () => ({
  decryptToTempFile: vi.fn().mockResolvedValue("/tmp/fake-decrypted.png"),
  cleanupTempFile: vi.fn(),
}));

// Keep the real AlreadyRegisteredError class (auto-mocking it would replace
// its constructor and lose the `contentHash` field the real class sets).
vi.mock("../src/clients/blockchainSvc.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/clients/blockchainSvc.js")>();
  return { ...actual, registerAsset: vi.fn(), verifyAsset: vi.fn() };
});

// Imported after the mocks are registered, per vitest's hoisting rules.
const { createProtectJob, pollProtectJob } = await import("../src/clients/protectionSvc.js");
const { registerAsset, verifyAsset, AlreadyRegisteredError } = await import("../src/clients/blockchainSvc.js");
const { cleanupTempFile } = await import("../src/crypto/imageEncryption.js");
const { runUploadPipeline, recoverInterruptedUploads } = await import("../src/orchestration.js");

const OWNER = "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2";

function seedArtwork(db: ReturnType<typeof createTestDb>, overrides: Partial<typeof artworks.$inferInsert> = {}) {
  const now = new Date();
  const row = {
    id: "ast_test1",
    title: "Test Artwork",
    sourceImageUri: "C:/fake/path.png",
    creatorId: "creator_1",
    ownerWalletAddress: OWNER,
    protectionProfile: "L1_PREVIEW",
    allowAiTraining: false,
    watermarkPayloadHex: "deadbeefcafef00d",
    encryptedImagePath: "./data/encrypted/ast_test1.enc",
    encryptedDekBase64: "ZmFrZQ==",
    encryptionIv: "ZmFrZQ==",
    encryptionAuthTag: "ZmFrZQ==",
    status: "UPLOADED",
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
  db.insert(artworks).values(row).run();
  return row;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("runUploadPipeline", () => {
  it("goes UPLOADED -> PROTECTING -> REGISTERING -> PUBLISHED on the happy path", async () => {
    const db = createTestDb();
    seedArtwork(db);

    vi.mocked(createProtectJob).mockResolvedValue({ jobId: "job_1", status: "queued" });
    vi.mocked(pollProtectJob).mockResolvedValue({
      jobId: "job_1",
      status: "completed",
      protectedImageUri: "out/job_1/watermarked.png",
      perceptualHash: "0xaaaa",
      metadataHash: "0xbbbb",
      appliedPreset: "L1_PREVIEW",
      eotUsed: false,
      styleDriftScore: 0.1888,
      styleSimilarityToOriginal: 0.8552,
      perceptualPsnrDb: 34.5,
      variants: [
        { name: "grid_thumbnail_150", width: 150, height: 150, scaleVsSource: 0.59, protectionStatus: "SAFE" },
      ],
    });
    vi.mocked(registerAsset).mockResolvedValue({
      contentHash: "0xcccc",
      ownerAddress: OWNER,
      doNotTrain: true,
      txHash: "0xdeadbeef",
      blockNumber: 123,
    });

    await runUploadPipeline(db, "ast_test1");

    const artwork = db.select().from(artworks).where(eq(artworks.id, "ast_test1")).get()!;
    expect(artwork.status).toBe("PUBLISHED");
    expect(artwork.perceptualHash).toBe("0xaaaa");
    expect(artwork.metadataHash).toBe("0xbbbb");
    expect(artwork.protectJobId).toBe("job_1");
    expect(artwork.styleDriftScore).toBeCloseTo(0.1888);
    expect(artwork.styleSimilarityToOriginal).toBeCloseTo(0.8552);
    expect(artwork.perceptualPsnrDb).toBeCloseTo(34.5);

    expect(registerAsset).toHaveBeenCalledWith({
      ownerAddress: OWNER,
      perceptualHash: "0xaaaa",
      metadataHash: "0xbbbb",
      doNotTrain: true, // allowAiTraining: false -> doNotTrain: true
    });

    // protection-svc gets the *decrypted temp* path, never the encrypted
    // blob's path directly -- and the temp file is cleaned up afterward.
    expect(createProtectJob).toHaveBeenCalledWith(
      expect.objectContaining({ imageUri: "/tmp/fake-decrypted.png" }),
    );
    expect(cleanupTempFile).toHaveBeenCalledWith("/tmp/fake-decrypted.png");
  });

  it("marks the artwork FAILED when the protect job fails", async () => {
    const db = createTestDb();
    seedArtwork(db);

    vi.mocked(createProtectJob).mockResolvedValue({ jobId: "job_2", status: "queued" });
    vi.mocked(pollProtectJob).mockResolvedValue({ jobId: "job_2", status: "failed", error: "GPU out of memory" });

    await runUploadPipeline(db, "ast_test1");

    const artwork = db.select().from(artworks).where(eq(artworks.id, "ast_test1")).get()!;
    expect(artwork.status).toBe("FAILED");
    expect(artwork.errorMessage).toContain("GPU out of memory");
    expect(registerAsset).not.toHaveBeenCalled();
  });

  it("treats a 409 with a matching on-chain owner as idempotent success, per blockchain-svc/INTEGRATION.md", async () => {
    const db = createTestDb();
    seedArtwork(db);

    vi.mocked(createProtectJob).mockResolvedValue({ jobId: "job_3", status: "queued" });
    vi.mocked(pollProtectJob).mockResolvedValue({
      jobId: "job_3",
      status: "completed",
      protectedImageUri: "out/job_3/watermarked.png",
      perceptualHash: "0xaaaa",
      metadataHash: "0xbbbb",
      variants: [],
    });
    vi.mocked(registerAsset).mockRejectedValue(new AlreadyRegisteredError("0xcccc"));
    vi.mocked(verifyAsset).mockResolvedValue({
      contentHash: "0xcccc",
      exists: true,
      owner: OWNER, // same owner as the artwork -> idempotent
      timestamp: 1234567890,
      doNotTrain: true,
    });

    await runUploadPipeline(db, "ast_test1");

    const artwork = db.select().from(artworks).where(eq(artworks.id, "ast_test1")).get()!;
    expect(artwork.status).toBe("PUBLISHED");
    // protection-svc omitted the protection-metrics fields entirely (e.g.
    // USE_REMOTE_GPU skipped the measurement) -- must persist as null, not
    // silently coerce to 0, which would misrepresent "not measured" as "no
    // protection effect".
    expect(artwork.styleDriftScore).toBeNull();
    expect(artwork.styleSimilarityToOriginal).toBeNull();
    expect(artwork.perceptualPsnrDb).toBeNull();
  });

  it("marks the artwork FAILED (not silently overwritten) on a genuine hash collision with a different owner", async () => {
    const db = createTestDb();
    seedArtwork(db);

    vi.mocked(createProtectJob).mockResolvedValue({ jobId: "job_4", status: "queued" });
    vi.mocked(pollProtectJob).mockResolvedValue({
      jobId: "job_4",
      status: "completed",
      protectedImageUri: "out/job_4/watermarked.png",
      perceptualHash: "0xaaaa",
      metadataHash: "0xbbbb",
      variants: [],
    });
    vi.mocked(registerAsset).mockRejectedValue(new AlreadyRegisteredError("0xcccc"));
    vi.mocked(verifyAsset).mockResolvedValue({
      contentHash: "0xcccc",
      exists: true,
      owner: "0x0000000000000000000000000000000000dEaD", // different owner -> real conflict
      timestamp: 1234567890,
      doNotTrain: true,
    });

    await runUploadPipeline(db, "ast_test1");

    const artwork = db.select().from(artworks).where(eq(artworks.id, "ast_test1")).get()!;
    expect(artwork.status).toBe("FAILED");
    expect(artwork.errorMessage).toContain("collision");
  });
});

describe("recoverInterruptedUploads", () => {
  it("marks a PROTECTING artwork FAILED with an honest interrupted message, doesn't touch protection-svc", async () => {
    const db = createTestDb();
    seedArtwork(db, { id: "ast_stuck_protecting", status: "PROTECTING" });

    await recoverInterruptedUploads(db);

    const artwork = db.select().from(artworks).where(eq(artworks.id, "ast_stuck_protecting")).get()!;
    expect(artwork.status).toBe("FAILED");
    expect(artwork.errorMessage).toContain("interrupted");
    expect(artwork.errorMessage).toContain("re-upload");
    expect(createProtectJob).not.toHaveBeenCalled();
  });

  it("resumes a REGISTERING artwork by retrying only the on-chain call, not re-protecting", async () => {
    const db = createTestDb();
    seedArtwork(db, {
      id: "ast_stuck_registering",
      status: "REGISTERING",
      perceptualHash: "0xaaaa",
      metadataHash: "0xbbbb",
    });

    vi.mocked(registerAsset).mockResolvedValue({
      contentHash: "0xcccc",
      ownerAddress: OWNER,
      doNotTrain: true,
      txHash: "0xresumed",
      blockNumber: 456,
    });

    await recoverInterruptedUploads(db);

    const artwork = db.select().from(artworks).where(eq(artworks.id, "ast_stuck_registering")).get()!;
    expect(artwork.status).toBe("PUBLISHED");
    expect(registerAsset).toHaveBeenCalledWith({
      ownerAddress: OWNER,
      perceptualHash: "0xaaaa",
      metadataHash: "0xbbbb",
      doNotTrain: true,
    });
    expect(createProtectJob).not.toHaveBeenCalled(); // never re-ran protection
  });

  it("leaves UPLOADED/PUBLISHED/FAILED artworks alone -- only PROTECTING/REGISTERING are stuck states", async () => {
    const db = createTestDb();
    seedArtwork(db, { id: "ast_uploaded", status: "UPLOADED" });
    seedArtwork(db, { id: "ast_published", status: "PUBLISHED" });
    seedArtwork(db, { id: "ast_failed", status: "FAILED", errorMessage: "some earlier real failure" });

    await recoverInterruptedUploads(db);

    expect(db.select().from(artworks).where(eq(artworks.id, "ast_uploaded")).get()!.status).toBe("UPLOADED");
    expect(db.select().from(artworks).where(eq(artworks.id, "ast_published")).get()!.status).toBe("PUBLISHED");
    const failedArtwork = db.select().from(artworks).where(eq(artworks.id, "ast_failed")).get()!;
    expect(failedArtwork.status).toBe("FAILED");
    expect(failedArtwork.errorMessage).toBe("some earlier real failure"); // not overwritten
  });
});
