import { sqliteTable, text, integer } from "drizzle-orm/sqlite-core";

/**
 * PROJECT_DESIGN.md §4's `users` table, first real implementation of it.
 * Social login (Google/Kakao) and self-custody wallet linking (MetaMask)
 * are out of scope for this pass -- passwordHash is always set, and
 * walletAddress is always the platform custodial wallet generated at
 * signup (src/auth/wallet.ts).
 */
export const users = sqliteTable("users", {
  id: text("id").primaryKey(),
  email: text("email").notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  handle: text("handle").notNull().unique(),
  displayName: text("display_name"),
  avatarUri: text("avatar_uri"),
  role: text("role").notNull().default("CREATOR"), // USER | CREATOR | MODERATOR | ADMIN
  walletAddress: text("wallet_address").notNull(),
  // RSA-PKCS1-wrapped custodial wallet private key (base64) -- only ever
  // unwrapped via infra/kms-adapter's unwrapKey() against the live KMS
  // server, never decrypted at rest. Nothing in this pass actually needs
  // to unwrap it yet (no server-side tx signing implemented); it's stored
  // now so that capability doesn't require a schema migration later.
  encryptedWalletKey: text("encrypted_wallet_key").notNull(),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  status: text("status").notNull().default("ACTIVE"),
});
