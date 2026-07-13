#!/bin/sh
# One-shot deploy of OwnershipRegistry against the compose-local anvil chain
# and authorization of the deployer as a relayer -- the same two steps
# apps/blockchain-svc/test/anvil.ts does programmatically for tests, done
# here as compose-native shell so blockchain-svc can start against a real
# (if ephemeral) contract instead of needing a mock.
set -e

RPC_URL="${RPC_URL:-http://anvil:8545}"
# Foundry/anvil's well-known default account #0 -- fine for a throwaway
# local dev chain, never used against real Amoy/mainnet.
PRIVATE_KEY="${PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}"
DEPLOYER_ADDRESS="${DEPLOYER_ADDRESS:-0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266}"
OUT_FILE="${OUT_FILE:-/shared/registry-address.txt}"

echo "waiting for anvil at $RPC_URL ..."
until cast block-number --rpc-url "$RPC_URL" >/dev/null 2>&1; do
  sleep 1
done

cd /contracts
forge build

DEPLOY_OUTPUT=$(forge create src/OwnershipRegistry.sol:OwnershipRegistry \
  --rpc-url "$RPC_URL" \
  --private-key "$PRIVATE_KEY" \
  --constructor-args "$DEPLOYER_ADDRESS" \
  --broadcast)

REGISTRY_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE "Deployed to: 0x[0-9a-fA-F]{40}" | cut -d' ' -f3)
if [ -z "$REGISTRY_ADDRESS" ]; then
  echo "failed to parse deployed address from forge create output:"
  echo "$DEPLOY_OUTPUT"
  exit 1
fi

echo "OwnershipRegistry deployed at $REGISTRY_ADDRESS"

cast send "$REGISTRY_ADDRESS" "setRelayer(address,bool)" "$DEPLOYER_ADDRESS" true \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY"

mkdir -p "$(dirname "$OUT_FILE")"
echo "$REGISTRY_ADDRESS" > "$OUT_FILE"
echo "wrote registry address to $OUT_FILE"
