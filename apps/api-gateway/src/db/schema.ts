import { sqliteTable, text, integer, uniqueIndex } from "drizzle-orm/sqlite-core";

/**
 * PROJECT_DESIGN.md §4's `users` table. Self-custody wallet linking
 * (MetaMask) is still out of scope -- walletAddress is always the platform
 * custodial wallet generated at signup (src/auth/wallet.ts), regardless of
 * auth method.
 *
 * Two ways in: email/password (passwordHash set, authProvider "LOCAL") or
 * social login (Google/Kakao -- passwordHash null, authProvider set to the
 * provider name, providerUserId is that provider's stable account id).
 * (email, authProvider) together, not email alone, is what's actually
 * unique -- see the UNIQUE index below -- since a LOCAL and a GOOGLE
 * account could theoretically share an email without being the same
 * person (no auto-merge implemented; see src/routes/oauth.ts).
 */
export const users = sqliteTable(
  "users",
  {
    id: text("id").primaryKey(),
    email: text("email").notNull(),
    passwordHash: text("password_hash"), // null for social-login-only accounts
    authProvider: text("auth_provider").notNull().default("LOCAL"), // LOCAL | GOOGLE | KAKAO
    providerUserId: text("provider_user_id"), // null for LOCAL; the provider's account id otherwise
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
  },
  (table) => ({
    emailProviderUnique: uniqueIndex("users_email_provider_unique").on(table.email, table.authProvider),
    providerAccountUnique: uniqueIndex("users_provider_account_unique").on(table.authProvider, table.providerUserId),
  }),
);
