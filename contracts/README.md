# DONTAI Contracts

Foundry project for DONTAI's on-chain ownership registry. See `PROJECT_DESIGN.md` §5 at the repo root for the full design (why Polygon, what gets anchored on-chain vs. off-chain, gasless registration via relayer).

## Structure

```
contracts/
├── src/
│   ├── OwnershipRegistry.sol         # the deployed-on-Amoy MVP registry (see DEPLOYMENTS.md)
│   └── OwnershipRegistryERC721.sol   # ERC-721 upgrade -- written, tested, NOT deployed anywhere. See "ERC-721 migration" below.
├── script/
│   ├── Deploy.s.sol                  # deployment script for OwnershipRegistry.sol
│   └── DeployERC721.s.sol            # deployment script for OwnershipRegistryERC721.sol (not yet run)
├── test/
│   ├── OwnershipRegistry.t.sol       # 10 tests
│   └── OwnershipRegistryERC721.t.sol # 12 tests
├── lib/
│   ├── forge-std/              # Foundry's testing/scripting std lib
│   └── openzeppelin-contracts/ # for Ownable, access control, etc.
├── foundry.toml                # solc 0.8.24, remappings, rpc endpoints (amoy/polygon)
├── .env.example                # copy to .env and fill in RPC URL + key (never commit .env)
└── .gitignore
```

## Setup

```shell
cp .env.example .env      # fill in AMOY_RPC_URL and PRIVATE_KEY (testnet key only)
forge build
forge test
```

## Local dev node

```shell
anvil
```

## Deploy to Polygon Amoy testnet

```shell
forge script script/Deploy.s.sol --rpc-url amoy --broadcast --verify
```

## ERC-721 migration (prepared, not scheduled)

`OwnershipRegistryERC721.sol` is PROJECT_DESIGN.md §8 Phase 4's "온체인 소유권
이전/ERC-721 승격" item: the same content-hash-anchoring model as
`OwnershipRegistry.sol` (only a hash + `doNotTrain` flag on-chain), but
ownership is now real ERC-721 state (`ownerOf`, `transferFrom`,
`approve`/`setApprovalForAll`) instead of the original's bespoke
single-owner-only `transfer()`. That's the actual motivation: marketplace
and wallet compatibility (OpenSea listing, MetaMask NFT display,
approval-based transfers) that the original contract structurally can't do.

**Written and tested, deliberately not deployed anywhere yet** (no
testnet address, no mainnet address, not referenced by any running
service). Deploying it is a real transaction with real gas cost, and
cutting `apps/blockchain-svc` over to it is a live-service change (new
`REGISTRY_ADDRESS`) -- both are decisions for whoever is about to spend
that money and flip that switch, not something to do as a side effect of
writing the contract. Before either step:

1. **Deploy to Amoy testnet first**, same as `OwnershipRegistry.sol` was:
   `forge script script/DeployERC721.s.sol --rpc-url amoy --broadcast --verify`,
   then record the address in `DEPLOYMENTS.md`.
2. **Decide the cutover semantics.** This is a new contract with its own
   token numbering -- registering `HASH_A` here mints a new tokenId
   independent of whatever id `OwnershipRegistry.sol` assigned it. Existing
   registrations don't carry over automatically; either re-register each
   still-active artwork against the new contract (an on-chain transaction
   per artwork, at the current owner's cost or the relayer's) or run both
   contracts side by side (`verify()` checked against whichever one a given
   artwork was actually registered on) and let new uploads use the ERC-721
   one going forward. PROJECT_DESIGN.md doesn't currently pick one -- worth
   deciding before real artworks depend on it.
3. **Only after Amoy has run clean for a while** does mainnet become a real
   option, and that's its own separate, explicit decision (real POL/MATIC,
   irreversible once broadcast) -- not implied by anything here.

## Useful commands

```shell
forge build       # compile
forge test        # run tests
forge fmt         # format solidity
forge snapshot    # gas snapshots
cast <subcommand> # interact with deployed contracts / chain data
```
