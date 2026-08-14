CREATE TABLE `pending_score_protection_jobs` (
	`artwork_id` text PRIMARY KEY NOT NULL,
	`job_id` text NOT NULL,
	`submitted_at` integer NOT NULL
);
