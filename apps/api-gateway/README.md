# api-gateway

PROJECT_DESIGN.md §2/§3-1: auth + routing layer in front of the other
services, which currently have **zero authentication of their own**
(confirmed across asset-service/blockchain-svc/protection-svc/detection-svc
-- this is the first auth introduced anywhere in the stack).

Full HTTP contract: [`openapi.yaml`](openapi.yaml) (OpenAPI 3.0, validated
with `openapi-spec-validator`) -- hand-written from the actual routes, not
generated, so if it ever drifts the code wins.

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
- **Social login** (`GET /auth/google`, `/auth/kakao` + their `/callback`
  routes, `src/auth/oauth.ts` + `src/routes/oauth.ts`) -- standard
  authorization-code flow via plain `fetch` against each provider's REST
  endpoints (no OAuth client library). CSRF-protected via an in-memory
  `state` token (10-minute TTL). On success, redirects the browser to
  `${WEB_URL}/oauth-callback#accessToken=...&refreshToken=...` (tokens in
  the URL *fragment*, not the query string, so they never land in server
  access logs or a `Referer` header). A provider with no
  `<PROVIDER>_CLIENT_ID`/`SECRET` configured responds `501` instead of
  erroring, so partial setup (e.g. Google only) is fine. **Requires real
  app registration you do yourself** -- see `.env.example` for the Google
  Cloud Console / Kakao Developers links; this repo has no way to obtain
  those credentials on your behalf.
- `users.authProvider` + `users.providerUserId` (unique together) identify
  the social account; `(email, authProvider)` is the real uniqueness
  constraint on top of email, so a Google account and a LOCAL account can
  share the same email as separate rows -- signing up locally never
  collides with a prior Google/Kakao login using the same address.
- **Community proxy** (`src/routes/community.ts`) -- likes, follows,
  bookmarks/collections, comments, reports, and the `/feed`
  (latest/popular/following) endpoint, all thin authenticated proxies to
  `asset-service`'s own community routes with identity injected from the
  JWT (never trusting a `userId` in the request body). Moderation
  endpoints (`GET /moderation/reports`, `PATCH /moderation/reports/:id`)
  are gated to `MODERATOR`/`ADMIN` roles -- asset-service itself has no
  concept of roles, so this gate exists only here.
- **Delivery Gateway signing proxy** (`GET /artworks/:id/render-url`) --
  the only trusted caller of `apps/delivery-gateway`'s `/internal/sign`
  (see that service's README's trust-boundary note). Always signs as
  `logged_in`/`thumbnail`, never `anonymous`, since every caller of this
  route is already authenticated (`requireAuth`) -- there's no public/
  unauthenticated browsing path in this stack yet. Returns an absolute
  URL (delivery-gateway's own origin + the signed path) since the browser
  hits delivery-gateway directly for the image bytes, not through this
  gateway.

## What this does not do

- **JWT is HS256 + a shared secret, not KMS-signed.** The real KMS C server
  (`src/protocol.c`) only implements envelope-key decrypt -- there is no
  `Sign()` RPC to move to. `src/auth/jwt.ts` is the single swap-out point if
  that's added later.

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
`test/oauth.test.ts` mocks only `exchangeCodeForProfile` (the actual network
call to Google/Kakao) and exercises everything else for real -- state
generation/validation, user lookup-or-create, the LOCAL/social email
collision case, and the redirect-with-tokens-in-hash response.
