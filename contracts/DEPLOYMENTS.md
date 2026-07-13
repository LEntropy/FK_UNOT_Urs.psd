# Deployments

## Polygon Amoy (testnet, chainId 80002)

| Contract | Address | Deployed | Deployer |
|---|---|---|---|
| `OwnershipRegistry` | `0x12fe026abacd896956ccf71044640af04c7e8a97` | 2026-07-06 | `0xCD836EEED3Cac282B053c1261f198f9eb848Aab2` |

Explorer (verified source): https://amoy.polygonscan.com/address/0x12fe026abacd896956ccf71044640af04c7e8a97#code

Status: ✅ deployed, ✅ verified on Polygonscan, ✅ smoke-tested (`register`/`verify` round trip confirmed on-chain, tx `0xde44e7533f76faf0f0ffea12657b6221ea80d149c9ba34e499d0d33aa4440a1b`)

Redeploy with:
```shell
forge script script/Deploy.s.sol --rpc-url amoy --broadcast
```

Re-verify with:
```shell
CTOR_ARGS=$(cast abi-encode "constructor(address)" <deployer_address>)
forge verify-contract <new_address> src/OwnershipRegistry.sol:OwnershipRegistry --chain amoy --constructor-args "$CTOR_ARGS"
```
