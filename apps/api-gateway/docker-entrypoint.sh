#!/bin/sh
set -e

mkdir -p "$(dirname "${DATABASE_URL:-./data/api-gateway.db}")"
node dist/db/migrate.js

exec "$@"
