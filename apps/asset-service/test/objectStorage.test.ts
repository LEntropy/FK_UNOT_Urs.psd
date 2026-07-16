import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

describe("local storage backend (default, no env needed)", () => {
  it("round-trips real bytes through the real filesystem", async () => {
    process.env.STORAGE_BACKEND = "local";
    const tempDir = mkdtempSync(join(tmpdir(), "dontai-storage-test-"));
    process.env.STORAGE_LOCAL_DIR = tempDir;

    const { getObjectStorage } = await import("../src/storage/objectStorage.js?t=" + Date.now());
    const storage = getObjectStorage();

    const uri = await storage.write("ast_test.enc", Buffer.from("real ciphertext bytes"));
    expect(uri).toBe(join(tempDir, "ast_test.enc"));
    expect(existsSync(uri)).toBe(true);

    const readBack = await storage.read(uri);
    expect(readBack.toString()).toBe("real ciphertext bytes");

    await storage.delete(uri);
    expect(existsSync(uri)).toBe(false);

    rmSync(tempDir, { recursive: true, force: true });
  });
});

// Real S3-compatible integration test -- skipped unless a live endpoint is
// configured (same pattern as infra/kms-adapter's own live-server test:
// CI/local dev don't require MinIO running, but this isn't mocked when it
// is available). Run a real MinIO locally to exercise this:
//   docker run -p 9000:9000 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin minio/minio server /data
//   S3_TEST_ENDPOINT=localhost npm test
describe.skipIf(!process.env.S3_TEST_ENDPOINT)("s3 storage backend (real MinIO, S3_TEST_ENDPOINT set)", () => {
  it("round-trips real bytes through a real S3-compatible endpoint", async () => {
    process.env.STORAGE_BACKEND = "s3";
    process.env.S3_ENDPOINT = process.env.S3_TEST_ENDPOINT;
    process.env.S3_ACCESS_KEY = process.env.S3_TEST_ACCESS_KEY ?? "minioadmin";
    process.env.S3_SECRET_KEY = process.env.S3_TEST_SECRET_KEY ?? "minioadmin";
    process.env.S3_BUCKET = "dontai-storage-test";

    const { getObjectStorage } = await import("../src/storage/objectStorage.js?t=" + Date.now());
    const storage = getObjectStorage();

    const uri = await storage.write("ast_test.enc", Buffer.from("real ciphertext bytes via s3"));
    expect(uri).toBe("s3://dontai-storage-test/ast_test.enc");

    const readBack = await storage.read(uri);
    expect(readBack.toString()).toBe("real ciphertext bytes via s3");

    await storage.delete(uri);
    await expect(storage.read(uri)).rejects.toThrow();
  });
});
