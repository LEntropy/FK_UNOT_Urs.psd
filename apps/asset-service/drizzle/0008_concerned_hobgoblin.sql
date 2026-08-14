CREATE TABLE `coin_balances` (
	`user_id` text PRIMARY KEY NOT NULL,
	`balance` integer DEFAULT 0 NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `coin_transactions` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`amount` integer NOT NULL,
	`reason` text NOT NULL,
	`related_artwork_id` text,
	`chain_tx_hash` text,
	`balance_after` integer NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `lora_generation_jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`source_artwork_id` text NOT NULL,
	`status` text DEFAULT 'QUEUED' NOT NULL,
	`result_path` text,
	`coin_cost` integer NOT NULL,
	`error_message` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `original_preview_unlocks` (
	`user_id` text NOT NULL,
	`artwork_id` text NOT NULL,
	`unlocked_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `original_preview_unlocks_pk` ON `original_preview_unlocks` (`user_id`,`artwork_id`);