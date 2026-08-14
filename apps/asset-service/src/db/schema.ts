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
  // Advanced-options upload feature (2026-08-08) -- only meaningful when
  // protectionProfile is STRONG_PROTECTION; null means "use hybrid_
  // protect.py's own HYBRID_FULL default," not "no protection." Set at
  // upload time (routes/artworks.ts), read back by orchestration.ts when
  // building the protect-svc job request.
  strongProtectionLatentEpsilon: real("strong_protection_latent_epsilon"),
  strongProtectionPixelEpsilon: real("strong_protection_pixel_epsilon"),
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

  // JSON array of strings, user-confirmed at upload time (routes/artworks.ts).
  // Seeded from protection-svc's /suggest-tags (ml-engine/src/tag_suggest.py --
  // CLIP zero-shot ranking against a fixed tag vocabulary, PixAI-style
  // image-to-tag preview), but the frontend lets the user edit/remove/add
  // before submitting, so this column is the *confirmed* list, not raw
  // model output -- there's no separate "suggestedTags" column because
  // nothing downstream needs the unconfirmed suggestions once upload
  // completes. Not nullable: "never called suggest-tags" and "confirmed
  // zero tags" both just mean "no tags" to every reader, so default to an
  // empty JSON array string rather than distinguishing null from "[]".
  tags: text("tags").notNull().default("[]"),

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

  // Real, per-upload measurements (protection-svc's orchestrate.py,
  // evaluate.py's compute_protection_metrics -- VGG19 Gram-matrix style
  // drift vs. the cloak target, and PSNR vs. the original) -- what
  // GalleryPage/ArtworkDetailPage's plain-language protection indicator is
  // actually built from, not a display-only number invented in the UI
  // layer. Nullable: protection-svc skips this measurement under
  // USE_REMOTE_GPU (no local torch there) or if it errors -- a real
  // "we didn't measure this" is a different fact from a measured drift of
  // zero, so it's null, not 0, when unavailable.
  styleDriftScore: real("style_drift_score"),
  styleSimilarityToOriginal: real("style_similarity_to_original"),
  perceptualPsnrDb: real("perceptual_psnr_db"),

  // Distinct from protectionProfile === "STRONG_PROTECTION" (what the
  // creator *requested*) -- protection-svc's dual-arch Serverless attack
  // can fail and fall back to plain style_cloak rather than fail the
  // whole upload (orchestrate.py's own strong_protection branch), so this
  // is what actually ran, set from the protect job's own usedStrongProtection
  // field. Gates the Test Lab's real-LoRA-effect-test button: only a
  // strong_protection artwork has cleared this project's own
  // n=30-plus-replication real-effect validation
  // (PHASE4_SCOPING.md §6) -- showing that test for a plain style_cloak
  // upload would likely surface a near-zero/negative delta, per this
  // project's own multi-mechanism validation history.
  usedStrongProtection: integer("used_strong_protection", { mode: "boolean" }).notNull().default(false),

  // Creator-only opt-OUT from the coin-unlock original-preview feature
  // (2026-08-14 redesign). Previously this was inverted: a creator had to
  // opt IN (a separate "원본 미리보기 공개" toggle) before the coin button
  // even appeared for anyone, and generation only happened on that toggle.
  // Real product intent was the opposite -- the coin-unlock option should
  // be available by default on every published artwork, with a creator who
  // doesn't want their original ever unlockable (at any coin price) able
  // to turn it off. Default false: coin-unlock is available unless a
  // creator explicitly blocks it. See routes/artworks.ts's unlock route
  // for how this gates both the button's visibility (GET /:id surfaces it
  // as originalPreviewAvailable = !originalPreviewBlocked) and the actual
  // spend (still checked server-side, not just hidden client-side).
  originalPreviewBlocked: integer("original_preview_blocked", { mode: "boolean" }).notNull().default(false),

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

/**
 * Coin system (2026-08-10). No `users` table exists in this service (see
 * module doc above) -- userId is the same unvalidated JWT-sub string every
 * other table already trusts, so balances/ledger follow the same FK-less
 * pattern rather than inventing a users table just for this.
 *
 * Real-money top-up is explicitly out of scope for this project (test/
 * portfolio scope, not a real payment processor) -- the only way to gain
 * coins right now is coins.ts's lazy signup bonus. `chainTxHash` on
 * coinTransactions is a deliberately-unused skeleton column for a possible
 * future on-chain top-up path; it is always null today.
 */

export const coinBalances = sqliteTable("coin_balances", {
  userId: text("user_id").primaryKey(),
  balance: integer("balance").notNull().default(0),
  updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

export const coinTransactions = sqliteTable("coin_transactions", {
  id: text("id").primaryKey(),
  userId: text("user_id").notNull(),
  // Positive = credit (e.g. signup_bonus), negative = spend.
  amount: integer("amount").notNull(),
  reason: text("reason").notNull(),
  relatedArtworkId: text("related_artwork_id"),
  // Always null today -- see module doc above.
  chainTxHash: text("chain_tx_hash"),
  balanceAfter: integer("balance_after").notNull(),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});

// A row existing here IS the grant -- one coin spend unlocks the viewer's
// original-preview access to that artwork permanently, no expiry/re-spend.
export const originalPreviewUnlocks = sqliteTable(
  "original_preview_unlocks",
  {
    userId: text("user_id").notNull(),
    artworkId: text("artwork_id").notNull(),
    unlockedAt: integer("unlocked_at", { mode: "timestamp" }).notNull(),
  },
  (table) => ({
    pk: uniqueIndex("original_preview_unlocks_pk").on(table.userId, table.artworkId),
  }),
);

// Test Lab's real-LoRA-training score (routes/artworks.ts's POST
// /:id/score-protection) used to only ever exist in the caller's own React
// state -- closing or reloading the tab lost the last real result even
// though the several-minutes RunPod job that produced it already happened.
// One row per artwork (upserted each time a job completes), not a history
// table -- "the last real result, always retrievable" is what was asked
// for, not a timeline of every run.
export const scoreProtectionResults = sqliteTable("score_protection_results", {
  artworkId: text("artwork_id").primaryKey(),
  // The full ScoreProtectionJob response body (protectionSvc.ts's own
  // shape: sd15/sdxl arch results incl. base64 samples, threshold) --
  // stored as opaque JSON rather than normalized into columns since
  // nothing here needs to query into it, only round-trip it back to the
  // client exactly as protection-svc produced it.
  resultJson: text("result_json").notNull(),
  checkedAt: integer("checked_at", { mode: "timestamp" }).notNull(),
});

// Tracks a score-protection job from the moment it's submitted until its
// result lands in scoreProtectionResults -- exists solely so a restart
// mid-job (asset-service gets redeployed a lot during active development,
// see the real incident this fixed: a real ~11-minute RunPod job finished
// with a real result, but the in-memory fire-and-forget poll that would
// have persisted it had been killed by an unrelated restart hours
// earlier, so the result just sat in protection-svc/RunPod, invisible to
// this service and the UI, forever) doesn't silently lose the result.
// One row per in-flight job; deleted once routes/artworks.ts's own
// completion handler (live or recovered) persists the real result.
export const pendingScoreProtectionJobs = sqliteTable("pending_score_protection_jobs", {
  artworkId: text("artwork_id").primaryKey(),
  jobId: text("job_id").notNull(),
  submittedAt: integer("submitted_at", { mode: "timestamp" }).notNull(),
});

// Durable counterpart to delivery-gateway's in-memory BotPolicyStore
// (src/bot_policy.rs) -- that store is a hot-path read/write cache only
// (RwLock<HashMap>/Mutex<VecDeque>, lost on every restart), never intended
// to be the system of record. asset-service already owns every other piece
// of per-artwork settings/state, so the real per-artwork ALLOW/BLOCK/
// LOG_ONLY policy lives here instead -- delivery-gateway write-through's a
// PUT here before updating its own cache, and lazily hydrates its cache
// from here on first read of an artwork it hasn't seen since its last
// restart. One row per artwork that has ever had a policy explicitly set
// (no row = the default policy applies, same default delivery-gateway's
// own ArtworkBotPolicy::default() encodes).
export const botPolicies = sqliteTable("bot_policies", {
  artworkId: text("artwork_id").primaryKey(),
  defaultAction: text("default_action").notNull(),
  // JSON-encoded { [BotGroup]: BotAction } / { [botName]: BotAction } --
  // small, sparse, per-artwork maps with no query need into individual
  // keys, same reasoning as scoreProtectionResults.resultJson above.
  groupPoliciesJson: text("group_policies_json").notNull().default("{}"),
  botOverridesJson: text("bot_overrides_json").notNull().default("{}"),
  updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

// Durable counterpart to delivery-gateway's in-memory bot-access ring
// buffer -- that buffer is capped (max_logs) and process-lifetime only, so
// a creator checking "who's been crawling my art" after a restart or after
// the ring buffer rolled over would see nothing. delivery-gateway pushes
// one row here per classified-bot request (best-effort, fire-and-forget --
// see lib.rs's render_asset doc), in addition to still recording into its
// own fast in-memory tail.
export const botAccessLogs = sqliteTable("bot_access_logs", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  artworkId: text("artwork_id").notNull(),
  botName: text("bot_name").notNull(),
  botGroup: text("bot_group").notNull(),
  action: text("action").notNull(),
  // Already /24-masked by delivery-gateway before this ever leaves that
  // process (bot_policy.rs's mask_ip) -- never the real full client IP.
  clientIp: text("client_ip").notNull(),
  userAgent: text("user_agent").notNull(),
  responseStatus: integer("response_status").notNull(),
  timestamp: integer("timestamp", { mode: "timestamp" }).notNull(),
});

// Compliance audit trail (2026-08-14, adapted from the compliance handoff's
// complience/Audit Log DB Schema.md -- ported from that draft's Postgres
// syntax (BIGSERIAL, CREATE RULE) to this project's actual SQLite/drizzle
// stack; the append-only enforcement lives as SQLite triggers in this
// table's migration instead of Postgres RULEs, see drizzle/0013's own SQL).
// Records consent/rights-relevant state changes a creator or viewer makes
// (bot access policy, original-preview coin-unlock availability, upload-
// time AI-training consent) -- the kind of "who changed what, when" trail
// that matters if a Do-Not-Train claim or takedown ever gets legally
// contested, distinct from this project's operational logs (which rotate/
// aren't append-only). payloadHash (not the raw payload) is what actually
// gets tamper-evidence from being append-only -- the full payload can
// still change shape over time without invalidating old hashes.
export const complianceAuditLogs = sqliteTable("compliance_audit_logs", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  // Nullable -- not every logged action has a wallet address behind it
  // (e.g. an anonymous bot-policy read has no actor identity today, see
  // routes/artworks.ts's own trust-boundary notes elsewhere in this file).
  userWallet: text("user_wallet"),
  actionType: text("action_type").notNull(),
  targetArtworkId: text("target_artwork_id"),
  ipAddress: text("ip_address"),
  userAgent: text("user_agent"),
  payloadHash: text("payload_hash").notNull(),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
});

export const loraGenerationJobs = sqliteTable("lora_generation_jobs", {
  id: text("id").primaryKey(),
  userId: text("user_id").notNull(),
  sourceArtworkId: text("source_artwork_id").notNull(),
  // QUEUED -> RUNNING -> COMPLETED | FAILED
  status: text("status").notNull().default("QUEUED"),
  resultPath: text("result_path"),
  coinCost: integer("coin_cost").notNull(),
  errorMessage: text("error_message"),
  createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
  updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});
