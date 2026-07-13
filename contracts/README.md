# DONTAI Contracts

Foundry project for DONTAI's on-chain ownership registry. See `PROJECT_DESIGN.md` §5 at the repo root for the full design (why Polygon, what gets anchored on-chain vs. off-chain, gasless registration via relayer).

## Structure

```
contracts/
├── src/
│   └── OwnershipRegistry.sol   # core contract (currently a skeleton — fill in the TODOs)
├── script/
│   └── Deploy.s.sol            # deployment script (reads PRIVATE_KEY from env)
├── test/
│   └── OwnershipRegistry.t.sol # test skeleton — fill in the TODOs
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

## Useful commands

```shell
forge build       # compile
forge test        # run tests
forge fmt         # format solidity
forge snapshot    # gas snapshots
cast <subcommand> # interact with deployed contracts / chain data
```
