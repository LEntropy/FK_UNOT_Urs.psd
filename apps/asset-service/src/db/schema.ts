import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";

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

  // Orchestration state machine: UPLOADED -> PROTECTING -> REGISTERING -> PUBLISHED
  //                                                     \-> FAILED (from any step)
  status: text("status").notNull().default("UPLOADED"),
  errorMessage: text("error_message"),

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
