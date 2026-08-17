-- Replaces strong_protection_latent_epsilon/strong_protection_pixel_epsilon
-- (hybrid_protect.py's now-retired two-epsilon advanced option) with a
-- single strong_protection_epsilon column matching clean_protect.py's
-- single-epsilon "강도" control (2026-08-15). Deliberately NOT backfilled
-- from either old column: those were latent-space epsilon values for a
-- different mechanism (hybrid_protect.py) on a different numeric scale
-- than clean_protect.py's pixel-space epsilon -- carrying an old number
-- forward under the new column would silently misrepresent what a past
-- upload actually asked for, not preserve it.
ALTER TABLE `artworks` ADD `strong_protection_epsilon` real;--> statement-breakpoint
ALTER TABLE `artworks` DROP COLUMN `strong_protection_latent_epsilon`;--> statement-breakpoint
ALTER TABLE `artworks` DROP COLUMN `strong_protection_pixel_epsilon`;