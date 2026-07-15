import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "../src/db/schema.js";

export function createTestDb() {
  const sqlite = new Database(":memory:");
  sqlite.exec(`
    CREATE TABLE users (
      id TEXT PRIMARY KEY,
      email TEXT NOT NULL,
      password_hash TEXT,
      auth_provider TEXT NOT NULL DEFAULT 'LOCAL',
      provider_user_id TEXT,
      handle TEXT NOT NULL UNIQUE,
      display_name TEXT,
      avatar_uri TEXT,
      role TEXT NOT NULL DEFAULT 'CREATOR',
      wallet_address TEXT NOT NULL,
      encrypted_wallet_key TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'ACTIVE'
    );
    CREATE UNIQUE INDEX users_email_provider_unique ON users (email, auth_provider);
    CREATE UNIQUE INDEX users_provider_account_unique ON users (auth_provider, provider_user_id);
  `);
  return drizzle(sqlite, { schema });
}
