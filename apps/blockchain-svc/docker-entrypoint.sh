#!/bin/sh
set -e

REGISTRY_FILE="${REGISTRY_ADDRESS_FILE:-/shared/registry-address.txt}"

echo "waiting for $REGISTRY_FILE (written by contracts-deploy) ..."
until [ -s "$REGISTRY_FILE" ]; do
  sleep 1
done

export REGISTRY_ADDRESS="$(cat "$REGISTRY_FILE")"
echo "using REGISTRY_ADDRESS=$REGISTRY_ADDRESS"

exec "$@"
