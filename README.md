# DONTAI

AI-training-protection + blockchain copyright-registration platform for
illustrators. Full design/architecture: [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md).

## Repo layout

Four independent sub-projects, no shared build tooling (no monorepo
manager) — each has its own lockfile/toolchain and is built/tested
separately, both locally and in CI.

```
apps/
  asset-service/     Node/TS + Express + SQLite (Drizzle) -- upload orchestration state machine
  blockchain-svc/     Node/TS + ethers.js -- on-chain ownership registration/verification
  protection-svc/     Python (FastAPI) + Rust -- AI-training-protection ("cloak") + watermark pipeline
    ml-engine/          PyTorch style-confusion cloak
    rust-core/           DCT watermark, C2PA manifest, resolution variants
  detection-svc/      Python (FastAPI) -- Phase 3 infringement detection & evidence packaging
contracts/            Solidity (Foundry) -- OwnershipRegistry.sol
```

Each service has its own README (and some have an `INTEGRATION.md`
documenting its HTTP API contract with its neighbors) — start there for
service-specific detail. This file is the map, not the manual.

## Who owns what (as of this writing)

- **backend** (`asset-service`, `blockchain-svc`) and **AI protection
  engine** (`protection-svc/ml-engine`) and **KMS** are active,
  in-progress work by other teammates.
- **CI/docker-compose** and **`detection-svc`** (Phase 3) were added in
  this pass — see git log for the commit that introduced them.
- If you're touching `ml-engine`'s cloak algorithm specifically: there are
  currently two parallel implementations being compared for merge (see
  team chat) — check before assuming this repo's version is canonical.

## Running everything locally

```bash
cp .env.example .env   # optional: only needed to set GOOGLE_APPLICATION_CREDENTIALS
docker compose up --build
```

Boots: a local `anvil` chain (free, no real testnet needed), deploys
`OwnershipRegistry` to it automatically, then `blockchain-svc` (3001),
`protection-svc` (8000), `asset-service` (3002), `detection-svc` (8003).
See each Dockerfile for what's baked in; `docker-compose.yml`'s header
comment explains the anvil-vs-real-Amoy tradeoff.

To point at the real Polygon Amoy testnet instead (e.g. for demo-parity
with the Pi deployment), override `AMOY_RPC_URL`/`RELAYER_PRIVATE_KEY`/
`REGISTRY_ADDRESS` on the `blockchain-svc` service in `docker-compose.yml`
rather than relying on the anvil auto-deploy.

## Running a single service without Docker

Each service's own README has the exact steps (venv/npm install, env vars,
dev server command). Quick links:
[asset-service](apps/asset-service/README.md) ·
[protection-svc](apps/protection-svc/INTEGRATION.md) ·
[detection-svc](apps/detection-svc/README.md) ·
[contracts](contracts/README.md)

## Tests

| Sub-project | Command |
|---|---|
| `contracts` | `cd contracts && forge test` |
| `apps/blockchain-svc` | `cd apps/blockchain-svc && npm test` (spawns a local anvil chain, no real testnet/secrets needed) |
| `apps/asset-service` | `cd apps/asset-service && npm test` (fully mocked, no `.env` needed) |
| `apps/protection-svc/rust-core` | `cd apps/protection-svc/rust-core && cargo test` |
| `apps/detection-svc` | `cd apps/detection-svc && pytest` (HTTP-mocked, no API keys needed) |

`apps/protection-svc`'s Python cloak code (`ml-engine`) has no automated
test suite yet — CI only lints it. See `.github/workflows/ci.yml`'s
`protection-svc-lint` job comment for why that's not silently papered
over.

## CI

`.github/workflows/ci.yml` runs all of the above per-subproject, plus a
`docker compose build` validation job, on every push/PR to `main`.

## Contributing / branch workflow

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
