# Phase 4 scoping — "고도화" (PROJECT_DESIGN.md §8)

A design pass over PROJECT_DESIGN.md §8's four Phase 4 items, before any of
them get implemented. Each section below covers: what the item actually
requires technically, how it relates to what's already built, an honest
feasibility read, and a recommendation for what to build first vs. defer.
None of the four items are implemented yet — this document exists so the
next implementation pass has a plan to work from instead of starting cold.

## 1. Concept Misalignment Layer (Nightshade-style)

### What it actually is, and how it differs from what's already built

PROJECT_DESIGN.md §3-3 lists four protection layers. Layer [2] (Style
Confusion, **built** — `apps/protection-svc/ml-engine/src/style_cloak.py`)
and layer [3] (Concept Misalignment, **not built**) sound similar but
target different failure modes in a downstream model:

- **Style Confusion** (built) perturbs the image in VGG19 feature space so
  a style-LoRA trained on it learns the wrong *style* -- validated for
  real against actual SD1.5/SDXL LoRA training runs
  (`ml-engine/README.md`'s LoRA-validation experiment, +0.0130 mean
  CLIP-similarity-to-true-image degradation, n=30).
- **Concept Misalignment** (Nightshade's actual mechanism) perturbs the
  image so that the *(image, caption)* pairing a model learns during
  fine-tuning is wrong -- e.g. an image captioned "a photo of a dog" is
  perturbed toward a different concept's visual features in a
  text-image-aligned embedding space (CLIP), so a model trained on enough
  poisoned pairs starts associating "dog" captions with the wrong visual
  features. This targets the *text encoder / cross-attention alignment*,
  not the *style embedding* -- a genuinely different optimization target,
  not a variant of the existing cloak.

### Feasibility read

The existing `style_cloak.py` already has the right shape to extend: an
iterative optimization loop against a frozen feature extractor
(`VGG19` today) with a perceptual-distance constraint. A concept-misalign
variant would swap in a CLIP (or open_clip) joint image-text embedding as
the target space, optimizing the image to sit near a *different* concept's
embedding than its real caption implies, subject to the same perceptual
budget.

**Where this needs an honest scope-down, matching this project's existing
practice** (the C2PA and LoRA-drift sections both report real, modest,
measured effects rather than oversold claims): published Nightshade
research demonstrates model-level concept corruption using *coordinated,
large-scale* poisoning (many images, many contributors, one shared target
concept). A single creator protecting their own individual artworks cannot
replicate that at the model level -- the honest, defensible claim for a
per-artwork tool is narrower: *"if this specific image is used in
fine-tuning, its caption-to-visual-feature association is measurably
wrong"* — a per-image poisoning effect, not "this defeats the model." That
distinction should be stated as plainly here as the C2PA section states
"don't rely on this as real proof of authenticity yet" was stated before
that bug was fixed.

### Recommended validation methodology (before writing production code)

Mirror the LoRA-validation experiment's actual methodology
(`experiments/lora_validation/`) rather than trusting a proxy metric:
1. Train a real SD1.5 LoRA on a small set of (concept-misaligned image,
   real caption) pairs.
2. Measure whether generation from that caption produces the *original*
   concept's features (protection failed) or a measurably different
   concept's features (protection working) -- CLIP-similarity between
   generated samples and both the true concept and the decoy concept, same
   shape as the existing baseline-vs-cloaked delta measurement.
3. Expect the same kind of image-dependent, noisy-per-image, real-in-
   aggregate result the existing experiment found -- design the sample
   size assuming that going in (start at n=4-6 images, expand only if the
   aggregate signal looks real, exactly like the L2_PORTFOLIO preset-
   scaling experiment's own history in this project).

### Where it plugs in

New module in `apps/protection-svc/ml-engine/src/` (e.g.
`concept_misalign.py`), gated as a new protection-profile capability (not
necessarily a new profile tier -- could be an opt-in flag on top of
`L3_ANTI_TRAIN`/`L4_LICENSED`, matching `allow_ai_training`'s existing
opt-in shape) rather than folded into the existing style-cloak function,
since the two have genuinely different loss targets and shouldn't share
one optimization loop.

## 2. Honeypot assets / honeypot URLs

### Direct extension of what `delivery-gateway` already built

`apps/delivery-gateway`'s crawler classification
(`src/crawlers.rs`/`is_known_ai_crawler`) currently does one of two things
per PROJECT_DESIGN.md §3-5's "차단 또는 decoy": it blocks (`403`). The
decoy half was explicitly deferred to this phase
(`apps/delivery-gateway/README.md`'s "What this does not do").

### Design

- **Honeypot assets**: instead of a flat `403` for a *known* crawler hit
  on a real artwork's signed URL, serve a decoy image variant with a
  unique, per-hit watermark payload (reusing `rust-core`'s existing
  watermark mechanism, `apps/protection-svc/rust-core/src/watermark.rs` --
  no new embedding tech needed, just a new payload-generation policy: one
  unique payload per honeypot serve, not the artwork's real stable
  payload). If that exact payload later surfaces in a scraped dataset or a
  third-party product, it's direct, individual proof *that specific
  crawler hit* is the source -- something a flat block can never provide.
- **Honeypot URLs**: a small number of fake artwork IDs that are never
  linked from any real page, seeded only into `robots.txt` as a
  `Disallow`'d path or into a hidden, unlinked sitemap. A real user can
  never navigate to one by clicking anything; a crawler that ignores
  `Disallow` (or one that scrapes `robots.txt` itself looking for
  "interesting" disallowed paths, a real behavior some scrapers exhibit)
  and requests it is unambiguously not a human following links. Any hit is
  a high-confidence bad-actor signal *by construction*, with zero
  false-positive risk from real traffic -- much stronger signal than
  anything UA-string-based.
- **Detection loop**: every honeypot hit logs `(ip, user_agent, timestamp,
  which_honeypot)`. This is the highest-confidence input to item 3 below
  (adaptive anti-scrape) -- a honeypot hit should immediately and
  permanently flag that fingerprint, not just nudge a score.

### Where it plugs in

New module inside `apps/delivery-gateway` (e.g. `src/honeypot.rs`) rather
than a separate service -- it needs the same signed-URL/variant-serving
machinery already built there, and the crawler-classification list it
extends already lives in that crate.

## 3. Adaptive anti-scrape (bulk-collection pattern learning)

### Current state and its real limitation

`delivery-gateway` today has exactly two static defenses: a per-IP
sliding-window rate limiter (`src/rate_limit.rs`) and a fixed UA denylist
(`src/crawlers.rs`). Both are honestly documented as limited in that
service's own README ("in-memory, single-process"; "does not attempt to
detect generic scraping bots... a crawler that lies about its UA string is
indistinguishable from a real browser here") -- and both are trivially
defeated by IP rotation or UA spoofing, which is exactly what a
determined, adaptive scraper does.

### Honest feasibility read

Real behavioral/reputation-based bot detection is a hard, ongoing problem
that production anti-bot vendors spend significant engineering effort on,
and it is **not** something a PoC-scale project can build with confidence
using synthetic reasoning alone -- it needs real traffic data to tune
false-positive rates against, which this project does not have (no real
public deployment with meaningful scraper traffic yet). Scoping a "full"
adaptive system now would produce untested, unvalidatable code -- the
opposite of this project's stated practice of measuring real effects
before claiming them.

### What's actually worth building now vs. deferring

**Worth building now, and now built** (`apps/delivery-gateway/src/
enumeration.rs`): distinct-artwork enumeration detection -- **adapted from
the plan below once implementation started**, this project's artwork IDs
turned out to already be random 16-hex-char strings
(`asset-service`'s `ast_${randomUUID()...}`), not a guessable sequence, so
literal "sequential ID" detection had nothing to detect. The applicable
signal is the same underlying behavior this section originally reasoned
about: a real user's session touches a handful of artworks (whatever the
UI's links present); a scraper touches many *distinct* artworks quickly
regardless of whether the IDs happen to be sequential or random, because
it's enumerating a feed/sitemap/guessed list rather than browsing.
Tracking distinct-artwork-count per IP in a sliding window (configurable
via `ENUMERATION_MAX_DISTINCT_ARTWORKS`/`ENUMERATION_WINDOW_SECONDS`)
captures that without depending on an ID scheme this project doesn't
have. Repeatedly re-requesting the same artwork never trips it.

*(Original plan, kept for context on the reasoning): "A scraper
enumerating `ast_1, ast_2, ast_3, ...` across many requests from one
fingerprint in a short window is a strong, cheap, low-noise signal" --
correct in spirit, wrong about this project's actual ID format.)*

**Worth deferring** (needs real production data first): a full
scoring/reputation system with escalating friction (slow-down → CAPTCHA-
style challenge → block → honeypot redirect) based on request-timing
distributions and cross-session correlation. Building this without real
traffic to validate against risks either being useless (thresholds too
loose) or actively harmful (false-positives blocking real users) -- worth
scoping in detail only once there's a real deployment generating the
traffic patterns to tune it against. Also worth noting as a real
limitation of what's built now: it's IP-based only, so a scraper rotating
IPs defeats it the same way it defeats the existing rate limiter.

### Where it plugs in

Implemented as `delivery-gateway`'s own `src/enumeration.rs` module (same
per-IP `DashMap` shape as `rate_limit.rs`, checked right after it in
`render_asset`) rather than a new service.

## 4. On-chain ownership transfer / ERC-721 upgrade, mainnet transition

### Current state

`contracts/src/OwnershipRegistry.sol`: a custom, non-ERC721,
mapping-based `Record` struct (`owner`, `contentHash`, `timestamp`,
`doNotTrain`) deployed to Polygon Amoy testnet
(`contracts/DEPLOYMENTS.md`). This deliberately satisfies PROJECT_DESIGN.md
§5-1's stated principle -- "이미지 자체를 온체인에 올리지 않는다...
**존재/소유 증명 앵커만** 올린다" (anchor only, not the image) -- and
nothing about that principle requires ERC-721 specifically.

### ERC-721 upgrade: treat as a separate decision from mainnet transition

**Case for ERC-721**: interoperability with existing wallet UIs,
marketplaces, and tooling that already understand the standard;
standardized `transferFrom`/`approve` semantics instead of this project's
own hand-rolled `transfer()`.

**Case against, right now**: real added gas cost and attack surface
(approval-based transfer flows are a well-known source of real-world
exploits -- phishing an `approve` call is the single most common NFT theft
vector) for a feature (marketplace/wallet interop) this project doesn't
currently need for its actual stated use case (existence/ownership proof +
do-not-train flag, not a tradeable collectible market). **Recommendation**:
do not upgrade to ERC-721 as part of the mainnet transition. Revisit only
if/when a real product requirement for marketplace interoperability
appears -- adding standard compliance later, once real registered records
already exist, is itself a migration this scoping should already flag
honestly (existing Amoy `tokenId`s would need an explicit mapping strategy
to ERC-721 `tokenId`s, not an in-place reinterpretation).

### Mainnet transition checklist

- **Security audit is a hard blocker, not optional.** The custom contract
  has never been audited. Real mainnet funds (both the registration gas
  cost callers pay, and the relayer wallet's own balance for
  custodial-wallet users) are a fundamentally different risk than a free
  testnet -- this is the single most important gap to close before
  considering mainnet, ahead of any of the other three items in this
  document.
- **Gas cost model.** Polygon mainnet gas is cheap relative to Ethereum L1
  but not free -- needs an actual per-registration cost estimate (current
  contract's `register()` call, real gas units × current Polygon gas
  price × MATIC/POL price) documented before launch, not assumed away
  because testnet was free.
- **Relayer key custody upgrade.** The KMS-backed relayer key
  (`apps/blockchain-svc`'s `RELAYER_ENCRYPTED_KEY` path, already built
  this session) is the right foundation, but a single relayer wallet
  holding real mainnet funds needs stronger custody than one KMS-wrapped
  key on one Pi -- a real deployment should consider a multi-sig or HSM-
  backed signer before mainnet, not the same setup that's fine for a
  testnet demo.
- **Relayer balance monitoring.** This project's own operational history
  already surfaced the relayer running low on testnet funds as a real,
  recurring problem (noted in this session's asset-encryption E2E
  verification, where blockchain registration failed only due to low
  relayer funds) -- a mainnet deployment needs actual balance alerting,
  not discovering this manually after a registration silently fails.

### Recommended order

Audit first. Everything else in this section (gas model, key custody,
monitoring) is worth doing regardless of audit outcome, but none of it
matters if the contract itself has an exploitable bug once real value is
on the line.
