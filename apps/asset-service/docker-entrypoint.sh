#!/bin/sh
set -e

# In docker-compose, REGISTRY_ADDRESS isn't known until contracts-deploy
# finishes deploying to the local anvil chain -- pick it up from the same
# shared file blockchain-svc's entrypoint waits on, rather than hardcoding
# a real-testnet address that wouldn't match the local chain.
REGISTRY_FILE="${REGISTRY_ADDRESS_FILE:-}"
if [ -n "$REGISTRY_FILE" ]; then
  echo "waiting for $REGISTRY_FILE (written by contracts-deploy) ..."
  until [ -s "$REGISTRY_FILE" ]; do
    sleep 1
  done
  export REGISTRY_ADDRESS="$(cat "$REGISTRY_FILE")"
  echo "using REGISTRY_ADDRESS=$REGISTRY_ADDRESS"
fi

mkdir -p "$(dirname "${DATABASE_URL:-./data/asset-service.db}")"
node dist/db/migrate.js

exec "$@"
