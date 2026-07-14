-- SQLite can't add a NOT NULL column without a default to a non-empty
-- table -- drizzle-kit's generated version omitted the default and would
-- fail against any already-populated artworks table (e.g. the live Pi
-- deployment). Existing rows get the same fallback constant
-- detection-svc already uses when this column is unset.
ALTER TABLE `artworks` ADD `watermark_payload_hex` text NOT NULL DEFAULT 'deadbeefcafef00d';