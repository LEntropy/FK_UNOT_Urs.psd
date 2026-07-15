-- SQLite can't add a NOT NULL column without a default to a non-empty
-- table (same issue as 0001's watermark_payload_hex migration). Unlike
-- that one, there's no sensible real default here -- rows created before
-- this migration were never encrypted, so there's no ciphertext to
-- backfill. '' is a sentinel: any attempt to decrypt one of these rows
-- (decryptToTempFile) will fail loudly (empty base64 -> unwrapKey error),
-- which is correct -- they were never actually protected by this
-- mechanism and shouldn't silently appear to be.
ALTER TABLE `artworks` ADD `encrypted_image_path` text NOT NULL DEFAULT '';--> statement-breakpoint
ALTER TABLE `artworks` ADD `encrypted_dek_base64` text NOT NULL DEFAULT '';--> statement-breakpoint
ALTER TABLE `artworks` ADD `encryption_iv` text NOT NULL DEFAULT '';--> statement-breakpoint
ALTER TABLE `artworks` ADD `encryption_auth_tag` text NOT NULL DEFAULT '';