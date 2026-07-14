CREATE TABLE `users` (
	`id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`password_hash` text NOT NULL,
	`handle` text NOT NULL,
	`display_name` text,
	`avatar_uri` text,
	`role` text DEFAULT 'CREATOR' NOT NULL,
	`wallet_address` text NOT NULL,
	`encrypted_wallet_key` text NOT NULL,
	`created_at` integer NOT NULL,
	`status` text DEFAULT 'ACTIVE' NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `users_email_unique` ON `users` (`email`);--> statement-breakpoint
CREATE UNIQUE INDEX `users_handle_unique` ON `users` (`handle`);