-- drizzle-kit can't auto-generate "drop NOT NULL from a column" for SQLite
-- (no ALTER COLUMN support) -- hand-written table rebuild instead, the
-- standard SQLite pattern: new table -> copy -> drop old -> rename.
-- Verified against a populated table (with existing signup rows) before
-- shipping this, not just against an empty dev db.
PRAGMA foreign_keys=OFF;--> statement-breakpoint

CREATE TABLE `__new_users` (
	`id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`password_hash` text,
	`auth_provider` text DEFAULT 'LOCAL' NOT NULL,
	`provider_user_id` text,
	`handle` text NOT NULL,
	`display_name` text,
	`avatar_uri` text,
	`role` text DEFAULT 'CREATOR' NOT NULL,
	`wallet_address` text NOT NULL,
	`encrypted_wallet_key` text NOT NULL,
	`created_at` integer NOT NULL,
	`status` text DEFAULT 'ACTIVE' NOT NULL
);--> statement-breakpoint
INSERT INTO `__new_users` (`id`,`email`,`password_hash`,`auth_provider`,`provider_user_id`,`handle`,`display_name`,`avatar_uri`,`role`,`wallet_address`,`encrypted_wallet_key`,`created_at`,`status`)
  SELECT `id`,`email`,`password_hash`,'LOCAL',NULL,`handle`,`display_name`,`avatar_uri`,`role`,`wallet_address`,`encrypted_wallet_key`,`created_at`,`status` FROM `users`;--> statement-breakpoint
DROP TABLE `users`;--> statement-breakpoint
ALTER TABLE `__new_users` RENAME TO `users`;--> statement-breakpoint

CREATE UNIQUE INDEX `users_handle_unique` ON `users` (`handle`);--> statement-breakpoint
CREATE UNIQUE INDEX `users_email_provider_unique` ON `users` (`email`,`auth_provider`);--> statement-breakpoint
CREATE UNIQUE INDEX `users_provider_account_unique` ON `users` (`auth_provider`,`provider_user_id`);--> statement-breakpoint

PRAGMA foreign_keys=ON;
