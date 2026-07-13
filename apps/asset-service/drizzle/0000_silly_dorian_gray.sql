CREATE TABLE `artworks` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`source_image_uri` text NOT NULL,
	`creator_id` text NOT NULL,
	`owner_wallet_address` text NOT NULL,
	`protection_profile` text NOT NULL,
	`allow_ai_training` integer DEFAULT false NOT NULL,
	`status` text DEFAULT 'UPLOADED' NOT NULL,
	`error_message` text,
	`protect_job_id` text,
	`protected_image_uri` text,
	`perceptual_hash` text,
	`metadata_hash` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `asset_versions` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`artwork_id` text NOT NULL,
	`variant_name` text NOT NULL,
	`storage_uri` text NOT NULL,
	`width` integer NOT NULL,
	`height` integer NOT NULL,
	`scale_vs_source` real NOT NULL,
	`protection_status` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `ownership_records` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`artwork_id` text NOT NULL,
	`owner_wallet` text NOT NULL,
	`content_hash` text NOT NULL,
	`chain` text NOT NULL,
	`registry_address` text NOT NULL,
	`tx_hash` text NOT NULL,
	`block_number` integer NOT NULL,
	`registered_at` integer NOT NULL
);
