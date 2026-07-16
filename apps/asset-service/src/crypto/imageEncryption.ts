import { randomBytes, createCipheriv, createDecipheriv } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { wrapKey, unwrapKey } from "@dontai/kms-adapter";
import { env } from "../env.js";
import { getObjectStorage } from "../storage/objectStorage.js";

/**
 * Envelope-encrypts the original uploaded image at rest (PROJECT_DESIGN.md
 * §6: "KMS holds only the KEK; DEK ciphertext stored in DB -- app does the
 * actual image encryption, not KMS"). A fresh AES-256-GCM DEK is generated
 * per artwork, used to encrypt the image bytes, then wrapped (RSA-PKCS1,
 * client-side, no network call -- see infra/kms-adapter) against the org's
 * public key. The plaintext original is deleted once the ciphertext is
 * written -- this is the actual "at rest" guarantee, not just an extra
 * copy sitting alongside the original.
 *
 * Decrypting back (decryptToTempFile) needs the live KMS server (unwrapKey
 * is a real network call) -- only exercised right before protection-svc
 * needs a real image to process (orchestration.ts), not at upload time.
 */

export interface EncryptedImage {
  encryptedImagePath: string;
  encryptedDekBase64: string;
  encryptionIv: string;
  encryptionAuthTag: string;
}

export async function encryptImageAtRest(plainImagePath: string, artworkId: string): Promise<EncryptedImage> {
  const plainBytes = readFileSync(plainImagePath);

  const dek = randomBytes(32); // AES-256
  const iv = randomBytes(12); // GCM standard IV size
  const cipher = createCipheriv("aes-256-gcm", dek, iv);
  const ciphertext = Buffer.concat([cipher.update(plainBytes), cipher.final()]);
  const authTag = cipher.getAuthTag();

  // Local filesystem or S3-compatible object storage, per STORAGE_BACKEND
  // (src/storage/objectStorage.ts) -- this is the "at rest" storage the
  // encrypted ciphertext actually lives in; the plaintext source (read
  // above, deleted below) is a separate, always-local upload path, not
  // something this migration touches.
  const encryptedImagePath = await getObjectStorage().write(`${artworkId}.enc`, ciphertext);

  const wrappedDek = wrapKey(env.KMS_PUBLIC_KEY_PATH, dek);

  unlinkSync(plainImagePath); // the actual "at rest" guarantee -- no lingering plaintext copy

  return {
    encryptedImagePath,
    encryptedDekBase64: wrappedDek.toString("base64"),
    encryptionIv: iv.toString("base64"),
    encryptionAuthTag: authTag.toString("base64"),
  };
}

/**
 * Decrypts to a temp file for protection-svc to read (it needs a real
 * local file path, not bytes in memory -- see INTEGRATION.md's imageUri
 * contract). Caller is responsible for deleting the returned path once
 * done (orchestration.ts does this in a finally block).
 */
export async function decryptToTempFile(encrypted: EncryptedImage, artworkId: string): Promise<string> {
  const wrappedDek = Buffer.from(encrypted.encryptedDekBase64, "base64");
  const dek = await unwrapKey({
    host: env.KMS_HOST,
    port: env.KMS_PORT,
    caCertPath: env.KMS_CA_CERT_PATH,
    requesterOrg: env.KMS_ORG,
    fileOrg: env.KMS_ORG,
    keyId: env.KMS_KEY_ID,
    encKey: wrappedDek,
  });

  const ciphertext = await getObjectStorage().read(encrypted.encryptedImagePath);
  const iv = Buffer.from(encrypted.encryptionIv, "base64");
  const authTag = Buffer.from(encrypted.encryptionAuthTag, "base64");

  const decipher = createDecipheriv("aes-256-gcm", dek, iv);
  decipher.setAuthTag(authTag);
  const plainBytes = Buffer.concat([decipher.update(ciphertext), decipher.final()]);

  // env.DECRYPT_TEMP_DIR, not os.tmpdir() -- see env.ts's doc comment on why:
  // the OS temp dir can be on a different, more space-constrained volume
  // than wherever this deployment's ./data actually lives.
  mkdirSync(env.DECRYPT_TEMP_DIR, { recursive: true });
  const tempPath = join(env.DECRYPT_TEMP_DIR, `dontai-decrypted-${artworkId}${extname(encrypted.encryptedImagePath)}`);
  writeFileSync(tempPath, plainBytes);
  return tempPath;
}

export function cleanupTempFile(path: string): void {
  try {
    unlinkSync(path);
  } catch {
    // already gone / never created -- not worth failing the whole pipeline over
  }
}

function extname(path: string): string {
  // Original extension is lost once encrypted (.enc replaces it) -- default
  // to .png since that's what every upload path in this PoC produces
  // (ml-engine's cloak() always writes PNG regardless of input format).
  return ".png";
}
