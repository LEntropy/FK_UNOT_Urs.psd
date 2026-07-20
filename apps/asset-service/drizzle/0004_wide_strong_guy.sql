ALTER TABLE `artworks` ADD `style_drift_score` real;--> statement-breakpoint
ALTER TABLE `artworks` ADD `style_similarity_to_original` real;--> statement-breakpoint
ALTER TABLE `artworks` ADD `perceptual_psnr_db` real;