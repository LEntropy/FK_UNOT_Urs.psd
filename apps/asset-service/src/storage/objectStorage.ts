import { mkdirSync, readFileSync, writeFileSync, unlinkSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { Client as MinioClient } from "minio";
import { env } from "../env.js";

/**
 * PROJECT_DESIGN.md §5-1's stated principle -- "실제 이미지... Object
 * Store... 선택" -- has been a documented gap since asset-service was
 * built (`sourceImageUri` a local file path, "same PoC-scope limit as
 * every other service"). This closes it for the highest-value target
 * first: the *encrypted* original at rest (imageEncryption.ts), which is
 * the single most sensitive asset in this whole system. Generated public
 * variants (rust-core's watermarked/thumbnail outputs, served through
 * delivery-gateway) stay on local disk for now -- they're regenerable
 * derivatives, not the thing "storage security" is actually protecting;
 * migrating those is real future work, not done here.
 *
 * `STORAGE_BACKEND=local` (default) preserves every existing local-dev
 * workflow exactly as-is -- nothing about this is a breaking change
 * unless you opt in. `STORAGE_BACKEND=s3` talks to any S3-compatible
 * endpoint (MinIO locally via docker-compose, real S3/R2/etc. in a real
 * deployment) via the `minio` client library, which despite its name
 * speaks plain S3 API and works against any S3-compatible service, not
 * just MinIO itself.
 */
export interface ObjectStorage {
  write(key: string, bytes: Buffer): Promise<string>;
  read(uri: string): Promise<Buffer>;
  delete(uri: string): Promise<void>;
}

class LocalObjectStorage implements ObjectStorage {
  constructor(private readonly baseDir: string) {}

  async write(key: string, bytes: Buffer): Promise<string> {
    const path = join(this.baseDir, key);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, bytes);
    return path;
  }

  async read(uri: string): Promise<Buffer> {
    return readFileSync(uri);
  }

  async delete(uri: string): Promise<void> {
    if (existsSync(uri)) unlinkSync(uri);
  }
}

class S3ObjectStorage implements ObjectStorage {
  private readonly client: MinioClient;
  private readonly bucket: string;

  constructor() {
    this.client = new MinioClient({
      endPoint: env.S3_ENDPOINT!,
      port: env.S3_PORT,
      useSSL: env.S3_USE_SSL,
      accessKey: env.S3_ACCESS_KEY!,
      secretKey: env.S3_SECRET_KEY!,
      region: env.S3_REGION,
    });
    this.bucket = env.S3_BUCKET!;
  }

  /** Idempotent -- safe to call before every write, cheap after the first
   * (a HEAD-equivalent bucketExists check), rather than requiring a
   * separate manual provisioning step before this service can start. */
  private async ensureBucket(): Promise<void> {
    const exists = await this.client.bucketExists(this.bucket).catch(() => false);
    if (!exists) await this.client.makeBucket(this.bucket, env.S3_REGION);
  }

  async write(key: string, bytes: Buffer): Promise<string> {
    await this.ensureBucket();
    await this.client.putObject(this.bucket, key, bytes);
    return `s3://${this.bucket}/${key}`;
  }

  async read(uri: string): Promise<Buffer> {
    const key = parseS3Uri(uri, this.bucket);
    const stream = await this.client.getObject(this.bucket, key);
    const chunks: Buffer[] = [];
    for await (const chunk of stream) chunks.push(chunk as Buffer);
    return Buffer.concat(chunks);
  }

  async delete(uri: string): Promise<void> {
    const key = parseS3Uri(uri, this.bucket);
    await this.client.removeObject(this.bucket, key);
  }
}

function parseS3Uri(uri: string, expectedBucket: string): string {
  const match = uri.match(/^s3:\/\/([^/]+)\/(.+)$/);
  if (!match) throw new Error(`not an s3:// URI: ${uri}`);
  const [, bucket, key] = match;
  if (bucket !== expectedBucket) {
    throw new Error(`URI bucket "${bucket}" does not match configured bucket "${expectedBucket}"`);
  }
  return key;
}

let cached: ObjectStorage | undefined;

export function getObjectStorage(): ObjectStorage {
  if (!cached) {
    cached = env.STORAGE_BACKEND === "s3" ? new S3ObjectStorage() : new LocalObjectStorage(env.STORAGE_LOCAL_DIR);
  }
  return cached;
}
