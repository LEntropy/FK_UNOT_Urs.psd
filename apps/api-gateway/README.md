# api-gateway

PROJECT_DESIGN.md §2/§3-1: auth + routing layer in front of the other
services, which currently have **zero authentication of their own**
(confirmed across asset-service/blockchain-svc/protection-svc/detection-svc
-- this is the first auth introduced anywhere in the stack).

## Scope

- `POST /auth/signup`, `POST /auth/login`, `POST /auth/refresh` -- email +
  password, bcrypt-hashed, JWT access (short-lived) + refresh (long-lived).
- `GET /me` -- profile for the bearer token's owner.
- `POST /artworks`, `GET /artworks`, `GET /artworks/:id` -- thin authenticated
  proxy to `asset-service`. `creatorId`/`ownerWalletAddress` are taken from
  the verified JWT, never from the request body, so the frontend only ever
  talks to this service and can't upload as someone else.
- **Custodial wallet provisioning** at signup (`src/auth/wallet.ts`): a real
  wallet is generated (`ethers.Wallet.createRandom()`), its private key is
  RSA-wrapped via `@dontai/kms-adapter`'s `wrapKey()` against the live KMS
  server's org public key, and only the ciphertext (`encryptedWalletKey`) is
  stored. Nothing in this pass unwraps it yet (no server-side tx signing
  implemented) -- stored now so that capability doesn't need a later schema
  migration.

## What this does not do

- **JWT is HS256 + a shared secret, not KMS-signed.** The real KMS C server
  (`src/protocol.c`) only implements envelope-key decrypt -- there is no
  `Sign()` RPC to move to. `src/auth/jwt.ts` is the single swap-out point if
  that's added later.
- **No social login** (Google/Kakao) -- email/password only, per explicit
  scope decision for this pass.
- **No Docker image yet.** This service depends on `../../infra/kms-adapter`
  via a `file:` reference, which a Docker build rooted at `apps/api-gateway`
  can't resolve (outside the build context). Needs either a repo-root build
  context or a published/vendored copy of kms-adapter before this can be
  containerized -- deliberately left out rather than shipping a broken
  Dockerfile. Current deployment target (the Pi) runs it as a native
  process instead (`deploy/pi/start_api_gateway.sh`), matching how the other
  four services are already deployed there.

## Quick start

```bash
npm install
cp .env.example .env
npm run db:migrate
npm run dev
```

## Tests

```bash
npm test
```

`test/artworksProxy.test.ts` mocks `src/clients/assetService.ts` (no real
asset-service needed). `test/auth.test.ts` exercises the real
`wrapKey()` (client-side RSA encrypt against the checked-in
`kms-keys/teamA1_key_v1_pub.pem` fixture) but never calls the live KMS
server -- only `infra/kms-adapter`'s own roundtrip test does that.
