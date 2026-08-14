"""Seed local protected images for detection-svc integration testing.

Inserts a handful of `artworks` (+ matching `ownership_records`) rows
directly into asset-service's own SQLite file (DATABASE_URL, see
.env.example) using the real perceptual hash (apps/detection-svc/src/
phash_match.py -> ml-engine's compute_perceptual_hash_from_path) so
detection-svc's own phash-match lookups have real, non-fabricated rows to
match against -- not a fixture with a fake/random hash.

Edit IMAGES below to point at whatever demo images you actually have
locally before running; the three placeholders aren't checked into the
repo. Run from apps/asset-service:

    python tools/seed_detection_artworks.py
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = SERVICE_DIR.parent
sys.path.insert(0, str(WORKSPACE / "detection-svc" / "src"))

from phash_match import compute_perceptual_hash_from_path  # noqa: E402

# (artworkId, title, path to a real local image file) -- adjust to whatever
# demo images you have on disk; these three filenames are just examples,
# not files this repo ships.
IMAGES = [
    ("demo_watermarked", "Watermarked demo", WORKSPACE / "tl-watermarked.png"),
    ("demo_c2pa", "C2PA signed demo", WORKSPACE / "tl-c2pa-signed.png"),
    ("demo_protected_art10", "Integrated protected art 10", WORKSPACE / "art-protected-v2" / "art10.png"),
]


def main() -> None:
    database = SERVICE_DIR / "data" / "asset-service.db"
    if not database.is_file():
        raise FileNotFoundError(
            f"{database} does not exist yet -- run asset-service's own "
            "migrations first (pnpm db:migrate from apps/asset-service)."
        )
    conn = sqlite3.connect(database)
    now = int(time.time() * 1000)
    for artwork_id, title, image_path in IMAGES:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        phash = compute_perceptual_hash_from_path(str(image_path))
        # Columns match apps/asset-service/src/db/schema.ts's `artworks`
        # table (camelCase field -> snake_case column, drizzle-orm/
        # sqlite-core default). Everything not listed here (tags,
        # usedStrongProtection, the strong-protection epsilon overrides,
        # etc.) is nullable or has a DB-level default in that schema, so
        # omitting them is fine for a seed row.
        conn.execute(
            """INSERT OR REPLACE INTO artworks (
                id, title, source_image_uri, creator_id, owner_wallet_address, protection_profile,
                allow_ai_training, watermark_payload_hex, encrypted_image_path, encrypted_dek_base64,
                encryption_iv, encryption_auth_tag, visibility, status, published_at, protected_image_uri,
                perceptual_hash, metadata_hash, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                artwork_id, title, str(image_path), "local-demo", "0x0000000000000000000000000000000000000001",
                "L3_ANTI_TRAIN", 0, "deadbeefcafef00d", str(image_path), "local-demo", "local-demo",
                "local-demo", "public", "PUBLISHED", now, str(image_path), phash, "0x" + file_hash, now, now,
            ),
        )
        conn.execute("DELETE FROM ownership_records WHERE artwork_id = ?", (artwork_id,))
        conn.execute(
            """INSERT INTO ownership_records
               (artwork_id, owner_wallet, content_hash, chain, registry_address, tx_hash, block_number, registered_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                artwork_id, "0x0000000000000000000000000000000000000001", "0x" + file_hash,
                "local-demo", "0x0000000000000000000000000000000000000000", "0x" + "0" * 64, 0, now,
            ),
        )
        print(f"seeded {artwork_id}: {image_path.name} {phash}")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
