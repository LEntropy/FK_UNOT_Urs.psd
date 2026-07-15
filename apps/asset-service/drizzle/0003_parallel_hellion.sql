CREATE TABLE `bookmarks` (
	`user_id` text NOT NULL,
	`artwork_id` text NOT NULL,
	`collection_id` text,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `collections` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`name` text NOT NULL,
	`is_public` integer DEFAULT true NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `comments` (
	`id` text PRIMARY KEY NOT NULL,
	`artwork_id` text NOT NULL,
	`user_id` text NOT NULL,
	`body` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `follows` (
	`follower_id` text NOT NULL,
	`creator_id` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `likes` (
	`user_id` text NOT NULL,
	`artwork_id` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `reports` (
	`id` text PRIMARY KEY NOT NULL,
	`reporter_id` text NOT NULL,
	`artwork_id` text NOT NULL,
	`reason` text NOT NULL,
	`status` text DEFAULT 'PENDING' NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
ALTER TABLE `artworks` ADD `visibility` text DEFAULT 'public' NOT NULL;--> statement-breakpoint
ALTER TABLE `artworks` ADD `published_at` integer;--> statement-breakpoint
CREATE UNIQUE INDEX `bookmarks_pk` ON `bookmarks` (`user_id`,`artwork_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `follows_pk` ON `follows` (`follower_id`,`creator_id`);--> statement-breakpoint
CREATE UNIQUE INDEX `likes_pk` ON `likes` (`user_id`,`artwork_id`);