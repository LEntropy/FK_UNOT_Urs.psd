import { sqliteTable, text, integer, real, uniqueIndex } from "drizzle-orm/sqlite-core";

/**
 * Scoped-down slice of PROJECT_DESIGN.md §4's data model -- just the tables
 * the upload orchestration flow (§1-1) actually needs. No users/licenses/
 * community tables yet; those are separate future work, not implied to be
 * unnecessary.
 */

export const artworks = sqliteTable("artworks", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  // Local file path in this PoC -- same "not real object storage yet"
  // scope limit as protection-svc's imageUri (INTEGRATION.md).
  sourceImageUri: text("source_image_uri").notNull(),
  creatorId: text("creator_id").notNull(),
  ownerWalletAddress: text("owner_wallet_address").notNull(),
  protectionProfile: text("protection_profile").notNull(),
  allowAiTraining: integer("allow_ai_training", { mode: "boolean" }).notNull().default(false),
  // Generated at creation (routes/artworks.ts), passed through to
  // protection-svc's /protect request, and read back by detection-svc
  // (asset_client.get_artwork) for real per-artwork watermark detection --
  // previously dropped here entirely, forcing detection-svc to fall back to
  // a single project-wide constant (see detection-svc/README.md's "What
  // this does not do", now resolved).
  watermarkPayloadHex: text("watermark_payload_hex").notNull(),

  // Envelope encryption at rest (src/crypto/imageEncryption.ts,
  // PROJECT_DESIGN.md §6): the plaintext upload is encrypted with a
  // per-artwork AES-256-GCM DEK and deleted immediately; only the
  // ciphertext path and the KMS-wrapped DEK are kept. orchestration.ts
  // decrypts to a temp file right before protection-svc needs a real one,
  // and deletes that temp file once the protect job completes.
  encryptedImagePath: text("encrypted_image_path").notNull(),
  encryptedDekBase64: text("encrypted_dek_base64").notNull(),
  encryptionIv: text("encryption_iv").notNull(),
  encryptionAuthTag: text("encryption_auth_tag").notNull(),

  // PROJECT_DESIGN.md §3-2/§4: public feeds and the "following" feed both
  // filter on this; "followers"-only isn't enforced by any read path yet
  // (no per-viewer auth check in this service -- see community routes'
  // module doc), so it's accepted and stored but only "public"/"private"
  // currently change response behavior.
  visibility: text("visibility").notNull().default("public"),

  // Orchestration state machine: UPLOADED -> PROTECTING -> REGISTERING -> PUBLISHED
  //                                                     \-> FAILED (from any step)
  status: text("status").notNull().default("UPLOADED"),
  errorMessage: text("error_message"),
  // Set once, in setStatus() (orchestration.ts), the moment status first
  // becomes PUBLISHED -- feed ordering (routes/community.ts) sorts by this,
  // not updatedAt, so a later unrelated edit doesn't bump an old artwork
  // back to the top of "latest".
  publishedAt: integer("published_at", { mode: "timestamp" }),

  // Filled in as protection-svc's job progresses.
  protectJobId: text("protect_job_id"),
  protectedImageUri: text("protected_image_uri"),
  perceptualHash: text("perceptual_hash"),
  metadataHash: text("metadata_hash"),

  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

export const assetVersions = sqliteTable("asset_versions", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  artworkId: text("artwork_id").notNull(),
  variantName: text("variant_name").notNull(),
  storageUri: text("storage_uri").notNull(),
  width: integer("width").notNull(),
  height: integer("height").notNull(),
  scaleVsSource: real("scale_vs_source").notNull(),
  // rust-core's variants.rs ProtectionStatus: SAFE | UNKNOWN | UNSAFE --
  // see apps/protection-svc/rust-core/README.md for what these mean.
  protectionStatus: text("protection_status").notNull(),
});

export const ownershipRecords = sqliteTable("ownership_records", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  artworkId: text("artwork_id").notNull(),
  ownerWallet: text("owner_wallet").notNull(),
  contentHash: text("content_hash").notNull(),
  chain: text("chain").notNull(),
  registryAddress: text("registry_address").notNull(),
  txHash: text("tx_hash").notNull(),
  blockNumber: integer("block_number").notNull(),
  registeredAt: integer("registered_at", { mode: "timestamp" }).notNull(),
});

/**
 * PROJECT_DESIGN.md §3-2/§4 community tables. Same trust boundary as
 * `artworks` above: this service takes userId/creatorId/reporterId as
 * given, no auth of its own -- api-gateway is the only place identity gets
 * verified (src/routes/community.ts in api-gateway injects it from the JWT
 * before proxying here, same pattern as artworks.ts).
 */

export const follows = sqliteTable(
  "follows",
  {
    followerId: text("follower_id").notNull(),
    creatorId: text("creator_id").notNull(),
    createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  },
  (table) => ({
    pk: uniqueIndex("follows_pk").on(table.followerId, table.creatorId),
  }),
);

export const likes = sqliteTable(
  "likes",
  {
    userId: text("user_id").notNull(),
    artworkId: text("artwork_id").notNull(),
    createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  },
  (table) => ({
    pk: uniqueIndex("likes_pk").on(table.userId, table.artworkId),
  }),
);

export const collections = sqliteTable("collections", {
  id: text("id").primaryKey(),
  userId: text("user_id").notNull(),
  name: text("name").notNull(),
  isPublic: integer("is_public", { mode: "boolean" }).notNull().default(true),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});

export const bookmarks = sqliteTable(
  "bookmarks",
  {
    userId: text("user_id").notNull(),
    artworkId: text("artwork_id").notNull(),
    // Null = the user's default/uncategorized bookmarks, not an error --
    // collections are opt-in organization, not required to bookmark.
    collectionId: text("collection_id"),
    createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  },
  (table) => ({
    pk: uniqueIndex("bookmarks_pk").on(table.userId, table.artworkId),
  }),
);

export const comments = sqliteTable("comments", {
  id: text("id").primaryKey(),
  artworkId: text("artwork_id").notNull(),
  userId: text("user_id").notNull(),
  body: text("body").notNull(),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});

export const reports = sqliteTable("reports", {
  id: text("id").primaryKey(),
  reporterId: text("reporter_id").notNull(),
  artworkId: text("artwork_id").notNull(),
  reason: text("reason").notNull(),
  // PENDING -> RESOLVED | DISMISSED, set via the moderation queue endpoint.
  status: text("status").notNull().default("PENDING"),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});
