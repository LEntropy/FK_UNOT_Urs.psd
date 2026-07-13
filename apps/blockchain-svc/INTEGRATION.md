# Integration: protection-svc → blockchain-svc

This is the contract between the (not-yet-built) protection pipeline and this
service, so protection-svc can be implemented later without re-deriving how
on-chain registration is supposed to work.

## Where this fits in the upload pipeline

From `PROJECT_DESIGN.md` §1-1, the full upload flow is:

```
upload → hash originals → KMS-encrypt original → protection pipeline
       → watermark/C2PA → store protected version → blockchain register
       → publish
```

`blockchain-svc` only owns the second-to-last step. It does not know about
images, KMS, or storage — it only accepts already-computed hashes and returns
an on-chain record. The **orchestrator** (asset-service, or a pipeline worker
in Phase 1) is what calls protection-svc first, then calls this service with
protection-svc's output. See `PROJECT_DESIGN.md` §5-5 for the asset-service
side of this handoff.

## Who computes what

| Hash | Computed by | How |
|---|---|---|
| `perceptualHash` | protection-svc | pHash of the **protected/public** image variant (not the original) — so registration matches what's actually published |
| `metadataHash` | protection-svc or asset-service | `keccak256(canonicalJSON({title, creatorId, license, ...}))` — must be deterministic (stable key order) |
| `contentHash` | blockchain-svc (`computeContentHash`, `src/hash.ts`) | `keccak256(perceptualHash ‖ metadataHash)` — **do not recompute this elsewhere**; call this API and let it derive contentHash, or import the same formula if another service ever needs to verify locally |

`doNotTrain` is not a hash — it's the license flag from the `licenses` table
(`allow_ai_training` negated). Pass it straight through.

## Request contract

```
POST /assets/register
{
  "ownerAddress": "0x...",          // creator's wallet (custodial or self-managed)
  "perceptualHash": "0x<32 bytes>", // from protection-svc
  "metadataHash": "0x<32 bytes>",   // from protection-svc or asset-service
  "doNotTrain": true
}
```

Response (`201`):
```json
{
  "contentHash": "0x...",
  "ownerAddress": "0x...",
  "doNotTrain": true,
  "txHash": "0x...",
  "blockNumber": 41498503
}
```

`409` means this exact `(perceptualHash, metadataHash)` pair was already
registered — treat as idempotent success if the returned `contentHash`'s
on-chain owner (`GET /assets/verify/:contentHash`) matches `ownerAddress`;
otherwise it's a genuine collision worth flagging (two different creators'
pipelines produced the same hash — should not happen with real images, but
surfacing it beats silently overwriting).

## What NOT to do

- Don't call `/assets/register` synchronously in the request path that serves
  the upload response to the browser — `tx.wait()` takes several seconds on
  Amoy. Phase 1 can get away with a synchronous call from a background job;
  Phase 2+ should move this behind a queue (Redis/BullMQ, per
  `PROJECT_DESIGN.md` stack) so a slow block doesn't block the user-facing
  upload flow.
- Don't have protection-svc call this API directly. Keep protection-svc a
  pure image-transform service (Rust/Python) with no blockchain awareness —
  the orchestrator composes the two.
