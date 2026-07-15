import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "../src/db/schema.js";

/** In-memory DB with the same tables as src/db/schema.ts, for fast tests
 * that don't touch the filesystem or depend on a generated migration
 * file's random name.
 */
export function createTestDb() {
  const sqlite = new Database(":memory:");
  sqlite.exec(`
    CREATE TABLE artworks (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      source_image_uri TEXT NOT NULL,
      creator_id TEXT NOT NULL,
      owner_wallet_address TEXT NOT NULL,
      protection_profile TEXT NOT NULL,
      allow_ai_training INTEGER NOT NULL DEFAULT 0,
      watermark_payload_hex TEXT NOT NULL DEFAULT 'deadbeefcafef00d',
      encrypted_image_path TEXT NOT NULL DEFAULT './data/encrypted/test.enc',
      encrypted_dek_base64 TEXT NOT NULL DEFAULT 'ZmFrZQ==',
      encryption_iv TEXT NOT NULL DEFAULT 'ZmFrZQ==',
      encryption_auth_tag TEXT NOT NULL DEFAULT 'ZmFrZQ==',
      status TEXT NOT NULL DEFAULT 'UPLOADED',
      error_message TEXT,
      protect_job_id TEXT,
      protected_image_uri TEXT,
      perceptual_hash TEXT,
      metadata_hash TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
    CREATE TABLE asset_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      artwork_id TEXT NOT NULL,
      variant_name TEXT NOT NULL,
      storage_uri TEXT NOT NULL,
      width INTEGER NOT NULL,
      height INTEGER NOT NULL,
      scale_vs_source REAL NOT NULL,
      protection_status TEXT NOT NULL
    );
    CREATE TABLE ownership_records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      artwork_id TEXT NOT NULL,
      owner_wallet TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      chain TEXT NOT NULL,
      registry_address TEXT NOT NULL,
      tx_hash TEXT NOT NULL,
      block_number INTEGER NOT NULL,
      registered_at INTEGER NOT NULL
    );
  `);
  return drizzle(sqlite, { schema });
}
