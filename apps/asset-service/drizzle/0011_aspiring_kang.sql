CREATE TABLE `bot_access_logs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`artwork_id` text NOT NULL,
	`bot_name` text NOT NULL,
	`bot_group` text NOT NULL,
	`action` text NOT NULL,
	`client_ip` text NOT NULL,
	`user_agent` text NOT NULL,
	`response_status` integer NOT NULL,
	`timestamp` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `bot_policies` (
	`artwork_id` text PRIMARY KEY NOT NULL,
	`default_action` text NOT NULL,
	`group_policies_json` text DEFAULT '{}' NOT NULL,
	`bot_overrides_json` text DEFAULT '{}' NOT NULL,
	`updated_at` integer NOT NULL
);
