# asset-service — upload orchestrator

Implements the upload pipeline (`PROJECT_DESIGN.md` §1-1): ties
`protection-svc` and `blockchain-svc` together with persistence. Not the
full asset-service scope from `PROJECT_DESIGN.md` §3-2 (no community
features, feed, follows, moderation yet) — scoped to the orchestration
spine that connects the two services this project has already built and
tested independently.

## The state machine

```
UPLOADED -> PROTECTING -> REGISTERING -> PUBLISHED
                 \-> FAILED (from either step, with errorMessage set)
```

`POST /artworks` inserts the row and returns `202` immediately; the actual
work (`src/orchestration.ts`'s `runUploadPipeline`) runs fire-and-forget in
the background — same reasoning as `protection-svc`'s own job design: a
protect job alone can take from ~1 minute to hours
(`ml-engine/README.md`), so nothing here can block inside a request/response
cycle. `GET /artworks/:id` is how a caller observes progress.
`GET /artworks?creatorId=` lists artworks (optionally scoped to one
creator) — added for `apps/api-gateway`'s gallery proxy, which is the only
caller that needs it today.

## Quick start

```bash
cp .env.example .env
npm install
npx tsx src/db/migrate.ts   # creates ./data/asset-service.db
npm run dev                  # or: npx tsx src/index.ts
```

Requires `protection-svc` (`apps/protection-svc/server.py`) and
`blockchain-svc` (`apps/blockchain-svc`) running and reachable at the URLs
in `.env` (defaults: `:8000` and `:3001`).

```bash
curl -X POST http://localhost:3002/artworks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Artwork",
    "sourceImageUri": "C:/path/to/image.jpg",
    "creatorId": "artist_123",
    "ownerWalletAddress": "0xCD836EEED3Cac282B053c1261f198f9eb848Aab2",
    "protectionProfile": "L1_PREVIEW",
    "allowAiTraining": false
  }'
# -> 202 { "id": "ast_...", "status": "UPLOADED" }

curl http://localhost:3002/artworks/ast_...
# poll until "status": "PUBLISHED"
```

**Verified with a real end-to-end run**: a real painting through
`POST /artworks` -> polled to `PUBLISHED` -> confirmed the resulting
`ownershipRecords` entry against `blockchain-svc`'s own
`GET /assets/verify/:contentHash`, independently, and got `exists: true`
back. Three separately-running HTTP services (this one, protection-svc,
blockchain-svc), talking to each other over real network calls, ending in
an actual Polygon Amoy transaction.

## The 409 (duplicate content hash) handling that was designed but never coded until now

`apps/blockchain-svc/INTEGRATION.md` specified this handling back when
blockchain-svc was built, but nothing called it until asset-service existed
to test it. `src/orchestration.ts`:

- `blockchain-svc` returns `409` when a `contentHash` is already registered.
- asset-service re-verifies via `GET /assets/verify/:contentHash`.
- If the on-chain owner matches this artwork's `ownerWalletAddress`: treated
  as idempotent success (e.g. a retried request) — `PUBLISHED`, with an
  `ownershipRecords` row noting no new transaction was needed.
- If the owner differs: a genuine hash collision, marked `FAILED` with an
  explicit error rather than silently overwritten. (Verified with a mocked
  test in `test/orchestration.test.ts`, not just read from the design doc —
  see "Tests" below.)

## Tests

```bash
npm test
```

`test/orchestration.test.ts` mocks `protectionSvc`/`blockchainSvc`'s client
functions (not the whole HTTP stack) and runs `runUploadPipeline` against an
in-memory SQLite DB (`test/testDb.ts`), covering: the happy path, a failed
protect job, idempotent 409 handling, and genuine-collision 409 handling.
Fast and deterministic — the real end-to-end run above (a real protect job
taking ~90s+ and a real on-chain transaction) is a manual verification step,
not part of this suite, for the same reason ml-engine's GPU-dependent tests
aren't either.

## Envelope encryption at rest

`src/crypto/imageEncryption.ts`: every upload is AES-256-GCM encrypted with
a fresh per-artwork DEK immediately on `POST /artworks`, and the plaintext
is deleted right after -- not just an extra encrypted copy sitting next to
an unencrypted one. The DEK itself is RSA-wrapped (client-side, no network
call -- `@dontai/kms-adapter`'s `wrapKey()`) against the org's KMS public
key and stored alongside the ciphertext path. Decrypting back (needed once,
briefly, right before `protection-svc` processes the image) calls the live
KMS server's `unwrapKey()` and writes to a temp file that's deleted again
once the protect job finishes -- see `orchestration.ts`. This is the first
real use of `infra/kms-adapter` for actual image data, not just the
custodial-wallet/relayer-key uses elsewhere in this project.

## Object storage (§5-1)

`src/storage/objectStorage.ts`: the encrypted original (the ciphertext
`imageEncryption.ts` produces, the single most sensitive asset in this
system) now goes through a real storage abstraction instead of always
being a hardcoded local path. `STORAGE_BACKEND=local` (default) preserves
every existing local-dev workflow unchanged. `STORAGE_BACKEND=s3` talks to
any S3-compatible endpoint (MinIO locally via `docker compose --profile
storage up minio`, real S3/R2/etc. in a real deployment) via the `minio`
client library, which despite its name speaks plain S3 API against any
compatible service. `encryptedImagePath` in the DB becomes either a local
path or an `s3://bucket/key` URI depending on backend -- `decryptToTempFile`
reads through the same abstraction either way, no branching in
`orchestration.ts`.

**Scoped deliberately narrow**: only the encrypted original is migrated.
`protection-svc`'s generated public variants (watermarked/thumbnail
outputs, served through `delivery-gateway`) stay on local disk -- they're
regenerable derivatives, not the thing storage security is actually
protecting. Migrating those is real future work, not attempted here.

## Community features (§3-2)

`src/routes/community.ts`: feed, follows, likes, bookmarks/collections,
comments, and reports -> moderation queue. Same trust boundary as
`routes/artworks.ts` -- this service takes `userId`/`creatorId`/`reporterId`
as given in the request body, no auth of its own. `apps/api-gateway`'s own
`src/routes/community.ts` is the only place identity actually gets verified
(it injects the real id from the JWT before proxying here) and is also
where the moderation endpoints are role-gated to `MODERATOR`/`ADMIN` --
this service's `/moderation/*` routes have no role check themselves, so
calling them directly (unproxied) bypasses that.

- `POST/DELETE /artworks/:id/likes`, `GET .../likes/count` -- liking twice
  is idempotent (unique index + `onConflictDoNothing`), not an error.
- `POST/DELETE /artworks/:id/bookmarks` (optional `collectionId`),
  `GET /users/:userId/bookmarks`; `POST /collections`, `GET /collections`.
- `POST/DELETE /users/:creatorId/follow`, `GET .../followers/count`
  -- rejects following yourself.
- `POST /artworks/:id/comments`, `GET /artworks/:id/comments` (newest
  first, with a `rowid` tiebreak since same-millisecond timestamps are real
  under load, not just a test artifact).
- `POST /artworks/:id/reports` -> `PENDING`; `GET /moderation/reports`;
  `PATCH /moderation/reports/:id` (`RESOLVED`/`DISMISSED`) -- one-way, a
  second attempt on an already-resolved report is a `409`, not silently
  overwritten.
- `GET /feed?type=latest|popular|following` -- `latest`/`popular` only
  return `visibility=public` + `status=PUBLISHED` artworks, ordered by
  `publishedAt` (a dedicated column, set once in `orchestration.ts`'s
  `setStatus()` the moment status first becomes `PUBLISHED` -- not
  `updatedAt`, so an unrelated later edit can't bump an old artwork back to
  the top of "latest"). `popular` orders by live `COUNT(likes)`, computed
  per request rather than a denormalized counter -- fine at this scale, a
  real counter would need to also handle unlikes. `following` requires
  `userId` and 400s without it.

## What this does not do

- `sourceImageUri` (the *plaintext upload path*, before encryption) is
  still always a local file path, same PoC-scope limit as `protection-svc`'s
  `imageUri` (`apps/protection-svc/INTEGRATION.md`) -- the object-storage
  migration above covers the encrypted-at-rest ciphertext, not this.
- The S3 backend's code has real unit test coverage for the local backend
  and the URI-parsing logic, but hasn't been exercised end-to-end against
  a live MinIO in this session (blocked by an unrelated, pre-existing
  disk-space issue on this project's Pi deployment target) -- treat it as
  reviewed-but-not-live-verified until that happens.
- No auth, no per-tenant isolation (this service itself -- `apps/api-gateway`
  sits in front of it now, but asset-service's own endpoints still trust
  whatever creatorId a caller sends).
- `visibility=followers` is accepted and stored but not enforced by any read
  path -- there's no per-viewer auth check in this service to know who's
  asking, so "followers-only" behaves like "public" today. A real
  enforcement would need api-gateway to pass the viewer's own id through to
  reads, not just writes, which it doesn't do yet.
- Job state (both here and in protection-svc) is not resumable across a
  process restart mid-job — a killed server loses in-flight orchestration
  state, `artworks.status` just stays wherever it was.
