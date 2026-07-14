import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "../src/db/schema.js";

export function createTestDb() {
  const sqlite = new Database(":memory:");
  sqlite.exec(`
    CREATE TABLE users (
      id TEXT PRIMARY KEY,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      handle TEXT NOT NULL UNIQUE,
      display_name TEXT,
      avatar_uri TEXT,
      role TEXT NOT NULL DEFAULT 'CREATOR',
      wallet_address TEXT NOT NULL,
      encrypted_wallet_key TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'ACTIVE'
    );
  `);
  return drizzle(sqlite, { schema });
}
