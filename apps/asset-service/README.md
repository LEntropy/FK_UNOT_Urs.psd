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

## What this does not do

- No object storage — `sourceImageUri` is a local file path, same PoC-scope
  limit as `protection-svc`'s `imageUri` (`apps/protection-svc/INTEGRATION.md`).
- No auth, no per-tenant isolation.
- No community features (§3-2's feed/follows/likes/moderation) — this is
  just the upload orchestration spine.
- Job state (both here and in protection-svc) is not resumable across a
  process restart mid-job — a killed server loses in-flight orchestration
  state, `artworks.status` just stays wherever it was.
