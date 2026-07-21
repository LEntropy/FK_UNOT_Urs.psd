# Integration: asset-service <-> protection-svc

Mirrors `apps/blockchain-svc/INTEGRATION.md`'s format. protection-svc is now
a real HTTP service (`apps/protection-svc/server.py`, FastAPI) implementing
exactly the job contract below — not just a design document asset-service
has to guess at. Validated end to end over real HTTP: `POST /protect` a
real painting -> poll `GET /protect/{jobId}` to completion -> feed its
`perceptualHash`/`metadataHash` into `blockchain-svc`'s own
`POST /assets/register` -> confirmed on-chain via `GET /assets/verify`. Two
independent HTTP services, talking to each other with real network calls,
not two scripts imported into the same process.

## What protection-svc does not know

Same principle as blockchain-svc: no images-as-DB-rows knowledge, no users,
no licenses, no blockchain. It takes an image + a protection profile and
returns a protected image + the hashes `blockchain-svc` needs. asset-service
is still the orchestrator (`PROJECT_DESIGN.md` §1-1, §5-5).

## Two sub-components, one contract

Per `PROJECT_DESIGN.md` §2/§3-3, protection-svc is Rust (`rust-core`) +
Python (`ml-engine`), not one monolith. asset-service should only ever call
the combined pipeline below as a single job — it should not need to know
these are two processes internally.

```
original image
     |
     v
[ml-engine]  style_cloak.py --preset <profile> --eot   (slow: seconds-to-minutes, see below)
     |
     v
[rust-core]  watermark (built) + C2PA manifest (built, but claim signature
             doesn't validate — see below) + resolution variants (built)
     |
     v
final public image variant(s) -> perceptualHash computed HERE, not earlier
```

`rust-core`'s resolution-variant generator (`src/variants.rs`) produces the
Delivery Gateway's named tiers (`PROJECT_DESIGN.md` §3-5) and tags each one
`Safe`/`Unknown`/`Unsafe` based on the measured 0.5x/0.25x protection
breakpoints below — asset-service should surface that tag rather than
treating every generated size as equally protected. It resizes the
already-protected source as-is; it does not re-run the cloak or watermark
per variant, so an `Unsafe`-tagged thumbnail is genuinely unprotected, not
just unverified.

### 256px processing size blocks the 1280/2048 variant tiers entirely

**Update: fixed.** `server.py`'s `ProtectRequest.size` now defaults to
`orchestrate.py`'s `choose_processing_size()` (the real upload's own
resolution, capped at 1024) instead of a fixed 256 -- see that function's
doc comment for the real GPU measurement (a live user report of visibly
excessive noise plus mushy upscaling) that forced this: processing at a
fixed 256 and upscaling back up measurably lost on *both* perceptual
quality (PSNR) and protection strength (styleDriftScore) compared to
processing closer to the real resolution, on the same real test image.
`public_preview_1280`/`2048` are reachable now for any upload whose own
resolution supports them. The rest of this section is kept for the
historical reasoning (why 256 was chosen in the first place, and the
resize-robustness data below, which is unaffected by this change).

This wasn't visible from testing `ml-engine` and `rust-core` in isolation —
it only showed up once `orchestrate.py` actually ran the full chain.
`ml-engine`'s `cloak()` defaults to processing at 256x256 (`size` param,
`style_cloak.py`), which becomes the *published* image's actual resolution
(the watermark and variants both operate on whatever `cloak()` produced).
At 256x256, `public_preview_1280` and `public_preview_2048` are impossible
to reach — they'd require upscaling, which `variants.rs` correctly refuses
to do. Only `grid_thumbnail_512` and `grid_thumbnail_150` are ever produced,
regardless of the real uploaded artwork's resolution.

`size` is a real parameter, not a hard limit — `orchestrate.py --size 1024`
runs mechanically fine, and has now been re-validated at 1024x1024
(`ml-engine/README.md` and `rust-core/README.md`'s "Re-validated at
1024x1024" sections). Result: **the two 0.25x-resize breakpoints diverge at
higher resolution, for the exact reasons their original explanations
predicted.**

- ML cloak (information-floor failure): dramatically better at 1024px
  (-144% retained at 256px -> -14% at 1024px). Makes sense — 1024 * 0.25 =
  256px, which turns out to be "enough room," same as this project's own
  validated baseline resolution.
- Watermark (geometric/block-misalignment failure): barely moves (37.5% BER
  at 256px -> 29.7% at 1024px, still a clear failure). Makes sense too — the
  block-grid misalignment a resize causes is a ratio problem, not an
  absolute-pixel problem, so it doesn't improve just because the starting
  image was bigger.

**Practical upshot**: processing at a higher `size` would measurably help
close the ML cloak's small-thumbnail gap "for free," but does essentially
nothing for the watermark's — that one still needs an actual
resampling-synchronization fix (not attempted). `orchestrate.py`'s
`sizeValidated` flag (`true` only at the default 256) is still accurate for
"has this exact run's numbers been independently reproduced" — 1024px is
now *validated* in the sense that we know what changes and what doesn't,
but production capacity planning needs to account for the VRAM cost this
surfaced too: `eot_samples` and `size` multiply, not add, in GPU memory
pressure — a naive `size=1024, eot_samples=3` run pushed an 8GB GPU to
~94.5% VRAM and became far slower than linear pixel-count scaling would
predict (over 2 hours vs. an expected 16-32 minutes), while
`eot_samples=1` at the same size completed comfortably at ~69% VRAM.

`rust-core`'s watermark is implemented and independently robustness-tested
(`rust-core/README.md`) — importantly, it has **different failure
characteristics than ml-engine's cloak**: strong against JPEG at every
tested quality and against moderate (0.5x) resizing, but breaks down at
aggressive (0.25x) resizing for a *geometric* reason (block-grid
misalignment), not the *information floor* reason the ML cloak fails for.
Both mechanisms happen to fail around the same resize threshold, for
unrelated reasons — see `rust-core/README.md`'s side-by-side table before
assuming either one protects an image at every generated resolution.

`rust-core`'s C2PA manifest embedding also works — title, custom
`com.dontai.ownership` assertion (where blockchain-svc's contentHash/txHash
would go), content-hash data integrity, and now the claim signature itself
all embed and validate correctly on read-back. (Previously the signature
was misreported as `claimSignature.mismatch` — root-caused to a known
upstream `c2pa` crate bug around self-signed certs missing an Organization
subject attribute, fixed by adding one; full writeup in
`rust-core/README.md`'s "C2PA manifest" section.) The remaining caveat is
that the signing identity is self-signed, not from a real C2PA-trusted CA,
so `signingCredential.untrusted` is still an expected status — an
operational PKI question, not a cryptographic one. blockchain-svc's
on-chain registration remains the mechanism this project stands behind for
provenance today, but the C2PA manifest's signature can now be trusted as
real proof the manifest wasn't tampered with after signing.

**perceptualHash must be computed on the final published variant** (after
rust-core's watermark/C2PA step), not on ml-engine's raw cloak output —
otherwise the on-chain hash won't match what a downstream verifier
re-hashes from the actually-published image.

## API contract (job-based, not synchronous) — implemented

Cloaking is not fast. `ml-engine`'s own measurements (`ml-engine/README.md`):
~500 steps takes seconds on GPU, ~90s on CPU for `L3_ANTI_TRAIN` without EOT,
and roughly `eot_samples + 1`x that with `--eot` on (our validated default —
see the EOT section of `ml-engine/README.md` for why it's worth the cost).
This is a **background job**, not a request/response call in the upload path
— same rule as `blockchain-svc/INTEGRATION.md`'s "don't call this
synchronously." `apps/protection-svc/server.py` (FastAPI) implements exactly
this:

```
POST /protect
{
  "imageUri": "ml-engine/out/real/starry_night.jpg",  // local file path in this PoC, not real object storage yet
  "protectionProfile": "L3_ANTI_TRAIN",   // PROJECT_DESIGN.md §3-4 preset name
  "eot": true,                             // optional; omit to use the per-preset default (see below)
  "title": "My Artwork",
  "creatorId": "artist_123",
  "allowAiTraining": false,
  "watermarkPayloadHex": "deadbeefcafef00d",
  "size": 256
}
-> 202 { "jobId": "job_abc123", "status": "queued" }

GET /protect/{jobId}
-> 200 {
  "status": "completed",                 // queued | processing | completed | failed
  "protectedImageUri": "out/job_abc123/watermarked.png",
  "perceptualHash": "0x<32 bytes>",       // computed on protectedImageUri, post-watermark
  "metadataHash": "0x<32 bytes>",         // canonical JSON hash of title/creator/license
  "appliedPreset": "L3_ANTI_TRAIN",
  "eotUsed": true,
  "size": 256,
  "sizeValidated": true,
  "variants": [ { "name": "grid_thumbnail_150", "width": 150, "height": 150, "scaleVsSource": 0.59, "protectionStatus": "SAFE" }, ... ],
  "processingTimeMs": 87000
}
```

**Verified with a real end-to-end run, over actual HTTP** (not two scripts
sharing a process): `POST /protect` a real painting at `L1_PREVIEW` ->
polled `GET /protect/{jobId}` to `completed` (~97s) -> passed its
`perceptualHash`/`metadataHash` to `blockchain-svc`'s
`POST /assets/register` -> got back a real `txHash` -> confirmed with
`GET /assets/verify/{contentHash}` returning `exists: true`. Two
independently running HTTP services actually talking to each other.

**Concurrency**: the executor backing `/protect` is deliberately
`max_workers=1` — running two jobs at once on the same GPU risks an
out-of-memory crash, not just slowness, per the VRAM findings in
`ml-engine/README.md`'s 1024px re-validation section (`eot_samples x size`
can already approach an 8GB card's limit for a *single* job). A second
`POST /protect` while one is running just queues behind it; there's no
worker pool yet.

**What's still a PoC shortcut, not production**: `imageUri` is a local file
path (no object storage integration -- contrast with asset-service, which
now has a real S3-compatible option for the encrypted original,
`apps/asset-service/README.md`'s "Object storage" section); there's no
auth. Job state used to be an in-memory dict (lost on restart) -- now
persists to SQLite (`jobs_db.py`), so a *finished* job's status/result
survives a restart, and anything genuinely mid-flight when the process
dies gets marked failed with an honest "interrupted by restart" message
on the next startup rather than silently vanishing. This is not the same
as resumable -- there's no checkpoint mechanism to actually continue a
partial GPU optimization from. All noted in `server.py`'s own docstring
and `jobs_db.py`'s module doc, not hidden.

On `completed`, asset-service takes `perceptualHash` + `metadataHash`
straight into `blockchain-svc`'s `POST /assets/register`
(`blockchain-svc/INTEGRATION.md` — same field names on purpose, so the
handoff is a direct pass-through, no re-derivation).

## Preset -> ml-engine parameters

Concrete mapping from `PROJECT_DESIGN.md` §3-4 to what `ml-engine/src/style_cloak.py`
actually takes (`PRESETS` dict + `--eot` flags):

| Profile | epsilon | steps | eot | eot_scales |
|---|---|---|---|---|
| `L1_PREVIEW` | 0.02 | 150 | false (cheap tier, skip the 4x cost) | - |
| `L2_PORTFOLIO` | 0.04 | 300 | true | `[0.3, 1.0]` continuous |
| `L3_ANTI_TRAIN` | 0.08 | 500 | true | `[0.3, 1.0]` continuous |
| `L4_LICENSED` | n/a | n/a | n/a | protection pipeline is skipped entirely — see §3-4 |

## Known limitation asset-service/Delivery Gateway must respect

`ml-engine`'s EOT experiments (README.md) found the cloak's effect
**collapses below roughly 0.3x of the protected image's resolution** — not a
training bug, an information floor (a 256x256 image resized to 64x64 doesn't
have room left to carry the perturbation, no matter how it was trained).

**Concretely**: if Delivery Gateway (`PROJECT_DESIGN.md` §3-5) generates a
small grid thumbnail (e.g. 150px) from a 2000px protected image, that's a
~0.075x reduction — deep inside the collapse zone. The protection is
effectively void at that thumbnail tier, even though the 1280px/2048px
"public preview" variants (well above the 0.3x floor relative to a typical
upload) are fine.

**Update after building `rust-core`'s watermark**: it was assumed here that
small thumbnails could at least fall back on watermark/C2PA traceability
even without style-confusion intact. Measured now (`rust-core/README.md`):
**the watermark also fails at 0.25x resize**, just for a different
(geometric, block-grid misalignment) reason than the ML cloak's
(information-floor) failure. The two failure zones aren't identical but they
overlap enough that "fall back on the watermark for small thumbnails" is not
a safe assumption without checking the actual thumbnail size against both
mechanisms' measured breaking points.

**Until this is solved for real** (would need per-thumbnail-size cloaking
and/or a resampling-synchronization scheme for the watermark — not
attempted yet), asset-service should not represent small thumbnails as
"protected" the same way it represents the main preview, for *either*
mechanism.

## Style-target selection: opt-in auto-selection now wired in

Every upload's cloak target used to be a single fixed image
(`ml-engine/out/style_target.png`) regardless of the creator's own style.
The `ai-engine` branch's LoRA validation experiment found this is sometimes
close to worst-case: pre-cloak Gram-matrix similarity between the original
and its target correlates with the cloak's real (CLIP-measured, actually
trained LoRA) degradation effect — a more dissimilar target gives a bigger
real effect (controlled follow-up: Pearson r=-0.516, n=5). See that
branch's `ml-engine/README.md` for the full experiment; `ml-engine/src/
select_style_target.py` (now on `main` too) is the deliverable.

`orchestrate.py`'s `protect()` now calls `_maybe_auto_select_style_target()`
before cloaking, but it's **off by default** — set
`STYLE_TARGET_CANDIDATES_DIR` to a directory of candidate images to opt in;
unset (the default, including on the current Pi deployment) leaves
behavior exactly as before. Also a no-op under `USE_REMOTE_GPU=1`:
selection needs a local torch/VGG19 pass per candidate, which is exactly
what remote-GPU mode exists to avoid — extending `remote_gpu.py` to run
selection remotely too is unstarted. No curated candidate pool ships with
this repo (`ai-engine`'s pool is 10 famous paintings assembled for that
experiment, not a production asset) — turning this on for real needs one.

## perceptualHash: implemented

`ml-engine/src/perceptual_hash.py` — standard DCT pHash (`imagehash` library,
`hash_size=16` for an exact 256-bit / 32-byte output, matching
`blockchain-svc`'s `bytes32` with no padding/truncation). This is a
different mechanism from `evaluate.py`'s Gram-matrix cosine similarity
(style comparison, not a content fingerprint) — see `ml-engine/README.md`'s
"perceptualHash" section for the validation numbers (near-0 Hamming distance
across JPEG/resize of the same image, single-digit distance from our own
cloak's output, ~140/256 for an unrelated image).

**Resolved**: `orchestrate.py`'s `protect()` calls this on `watermarked_path`
(rust-core's output), not on ml-engine's raw cloak output — the ordering the
pipeline diagram above always called for. Confirmed by reading `protect()`
directly (step 4/4 runs after rust-core's embed/variants steps) and by
`test/test_server.py`'s coverage of the job that wraps it.

## GPU dependency in production

`ml-engine/remote/` (SSH to a second PC) is a **dev-only** workflow for this
PoC — it is not a production job runner. Before this integration ships,
protection-svc needs either (a) a real GPU worker pool/queue consumer, (b) a
managed GPU inference endpoint, or (c) CPU-only processing for lower tiers
(`L1_PREVIEW`/`L2_PORTFOLIO`) if GPU capacity isn't available yet — CPU
results are functionally identical to GPU (verified in `ml-engine/README.md`),
just slower, which is tolerable for a background job but not for something
serving a live user-facing wait.

## Tests

`test/test_server.py` (uses `ml-engine`'s venv — `httpx`/`pytest` added
there rather than giving `server.py` its own venv, since it already imports
`orchestrate.py` which needs `ml-engine`'s torch-heavy deps to even import):

```bash
apps/protection-svc/ml-engine/.venv/Scripts/python.exe -m pytest apps/protection-svc/test/ -q
```

Covers `server.py`'s HTTP job contract (`202`/`404`/`400` cases, the
queued→completed and queued→failed transitions, the `styleTargetUri`
default/override) against a mocked `orchestrate.protect()` — no torch/GPU
actually runs. `orchestrate.protect()` itself has no automated test (its
real dependencies — a GPU-capable `cloak()`, the compiled `rust-core`
binary — are exactly the parts `ml-engine/README.md` and
`rust-core/README.md`'s own manual, GPU-dependent experiments already
cover; duplicating that here would just mock away everything worth
testing).
