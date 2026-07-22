# Deployments

## Polygon Amoy (testnet, chainId 80002)

| Contract | Address | Deployed | Deployer |
|---|---|---|---|
| `OwnershipRegistry` | `0x8B3c2f06c772C301AFAF4609ab8f10470B1a8E16` | 2026-07-22 | `0xCD836EEED3Cac282B053c1261f198f9eb848Aab2` |

Explorer (verified source): https://amoy.polygonscan.com/address/0x8b3c2f06c772c301afaf4609ab8f10470b1a8e16#code

Status: ✅ deployed, ✅ verified on Polygonscan, ✅ relayer authorized (`setRelayer`, tx `0xefc152500db149897222a93cb1a03514d07689b610734ae944f99e882257d546`), ✅ smoke-tested on the real commit-reveal flow (`commit` tx `0xfbd6a25ef571967863981e001b3512eec0ba8d85c041a0749e2c1757d63a8d3b` → `register` tx `0x7be06318832fe27e71b7d36c1e2a746bf55e64d9cbef0dc25bc8c9d8d262cd0c` → `verify` confirmed on-chain)

**Redeployed 2026-07-22** (superseding the original `0x12fe0...c7e8a97`, still verified on Polygonscan but no longer the live address blockchain-svc points at) to fix a real security-review finding: `register`/`registerFor` originally broadcast `contentHash` in plaintext in the public mempool before confirmation, letting anyone watching resubmit the same hash at a higher gas price and become the recorded "first registrant" instead of the real caller — directly undermining the registry's entire purpose. Fixed with a commit-reveal scheme (`commit(commitHash)` → wait `MIN_COMMIT_AGE` (1) blocks → `register`/`registerFor(..., nonce)`, which recomputes and checks the commitment). Also added `Pausable` (emergency stop for `commit`/`register`/`registerFor`/`transfer`, `verify` stays callable) and switched `Ownable` → `Ownable2Step` (a mistyped `transferOwnership` call is no longer an unrecoverable admin lockout). See `src/OwnershipRegistry.sol`'s own doc comments on `commitBlock`/`MIN_COMMIT_AGE` for the full attack/defense analysis, and `test/OwnershipRegistry.t.sol`'s `test_commitReveal_defeatsACopyCatFrontRunner` for a working demonstration.

Old address `0x12fe026abacd896956ccf71044640af04c7e8a97` is now abandoned — it has no commit-reveal protection and should not be used for new registrations. Existing on-chain records there aren't migrated automatically (see `OwnershipRegistryERC721.sol`'s own doc comment on why a hash-registry migration always mints fresh IDs, same reasoning applies here).

Redeploy with:
```shell
forge script script/Deploy.s.sol --rpc-url amoy --broadcast
```

Re-verify with:
```shell
CTOR_ARGS=$(cast abi-encode "constructor(address)" <deployer_address>)
forge verify-contract <new_address> src/OwnershipRegistry.sol:OwnershipRegistry --chain amoy --constructor-args "$CTOR_ARGS"
```
