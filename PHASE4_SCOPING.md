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
  CLIP-similarity-to-true-image degradation, n=30). **That number is from
  an older preset config** (epsilon roughly double today's L2/L3 values,
  no color_weight/perceptual_mask/clip_transfer_weight/AMP) that only ever
  existed on a since-abandoned branch -- re-validated against the actual
  current `main` preset config on 2026-07-23 (n=10, reduced sample):
  +0.0123 mean delta, 95% CI [-0.0066, +0.0312] (includes zero at this
  sample size, same as small subsamples of the original n=30 would show).
  The effect size held up close to the original measurement despite the
  preset changes -- no evidence the changes degraded LoRA-training
  protection -- but it remains a weak, image-dependent effect either way
  (one image/seed combination even flipped sign, -0.0475), not a strong
  guarantee then or now. See `ml-engine/experiments/lora_validation/`'s
  own README/results for the full re-validation writeup.
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

### What's actually built, and what's honestly still missing

`apps/protection-svc/ml-engine/src/concept_misalign.py` implements the
optimization loop above: `model.py`'s new `ConceptFeatureExtractor` wraps
a CLIP (`open_clip`, `ViT-B-32/openai`) image encoder the same way
`StyleFeatureExtractor` wraps VGG19, and `misalign()` runs the same
epsilon-bounded gradient-descent shape as `style_cloak.py`'s `cloak()`,
minimizing `1 - cosine_similarity` between the image's CLIP embedding and
a decoy concept image's CLIP embedding instead of VGG19 Gram-matrix MSE.
Wired into `orchestrate.py`'s `protect()` as a fully opt-in
`concept_misalign_target_path` parameter (`None` by default, `--concept-
misalign-target` on the CLI) -- when set, runs after style-cloaking and
before watermarking; when unset (the default for every existing caller),
`protect()`'s behavior is byte-for-byte unchanged from before this file
existed.

**Two real gaps, stated plainly rather than glossed over:**

1. **No CLIP-text-side signal.** This pipeline optimizes toward a decoy
   *image's* CLIP embedding, not toward a mismatched *caption's* CLIP
   text embedding, because `orchestrate.py`'s inputs (`title`,
   `creator_id`) aren't the kind of descriptive training caption a real
   fine-tuning pipeline would pair with the image -- there's no real
   caption in this project's data model to target against. Using a decoy
   *image* embedding as the target is a reasonable proxy (CLIP's
   image-text space is joint, so pulling toward another image's region of
   that space still pulls away from whatever real caption the image would
   otherwise pair with correctly) but is not the literal mechanism
   described above, and that gap should stay visible rather than get
   quietly assumed away.
2. **The recommended validation methodology (three paragraphs up: train a
   real SD1.5 LoRA on misaligned-image/real-caption pairs, measure
   generation drift) has not been run.** It needs a real GPU LoRA-training
   run, the same kind `ml-engine/README.md`'s LoRA-validation experiment
   used on a separate GPU machine -- not available in the session this
   file was written in. Separately, even a CPU-only smoke test of
   `concept_misalign.py` itself (confirming the loop runs and the
   embedding actually drifts, short of any training-based claim) could not
   be completed in that same session: loading `open_clip`'s pretrained
   CLIP checkpoint was blocked by that environment's own external-code
   safety gate before a single forward pass ran. **Until someone runs
   either check for real, `concept_misalign.py` is unexercised code that
   compiles and follows the designed mechanism -- not a verified
   mechanism, and absolutely not a verified protection effect.** This is
   why it's wired as strictly opt-in with no default-on path anywhere

   **Update -- run for real, result is negative.** The validation
   experiment (`apps/protection-svc/ml-engine/experiments/
   concept_misalignment_validation/`, mirroring `experiments/
   lora_validation/`'s structure) was executed on the GPU PC via
   `remote/run_concept_misalignment_validation.ps1`: 5 images x 3 seeds x
   2 conditions = 30 real SD1.5 LoRA trainings, then generation + CLIP
   scoring against both the true and decoy concepts (n=15 image x seed
   combinations). Result: **mean delta_true = -0.0058** (95% CI
   [-0.0180, +0.0063], includes zero) and **mean delta_decoy = -0.0044**
   (95% CI [-0.0102, +0.0014], includes zero) -- verdict **WEAK/FAIL**.
   Training on the misaligned image did not measurably push generation
   away from the true concept or toward the decoy concept; both deltas
   are noise-level and, if anything, trend slightly the wrong direction.

   The CLIP-embedding optimization itself does work as designed (concept
   loss converges from ~0.3 to <0.001 within the epsilon budget during
   `misalign()` -- confirmed live during this run), so the gap is
   specifically that a single-image LoRA's training dynamics don't
   propagate that pixel-level perturbation into the text-encoder/
   cross-attention association the way the mechanism assumes -- plausibly
   because a 1-image LoRA at these settings essentially memorizes the
   image/trigger pair rather than learning a generalizable caption-to-
   visual-feature association fine-grained enough for a small pixel
   perturbation to redirect.

   **Follow-up (ruling out the single-image explanation): also run, also
   negative.** To check whether that single-image explanation was actually
   right, a second experiment (`experiments/concept_misalignment_
   validation/prepare_multiimage.py` + `remote/run_concept_misalignment_
   multiimage_validation.ps1`) trained one shared LoRA per condition per
   seed across all 5 images/triggers *jointly* instead -- closer to a real
   scraper's actual training set than an isolated single-image LoRA.
   Result (n=15, 3 shared LoRAs per condition): mean delta_true = -0.0020
   (95% CI includes zero), mean delta_decoy = +0.0027 (95% CI includes
   zero) -- **still WEAK/FAIL**, both means far below the 0.03 threshold.
   `delta_decoy` flipped sign (single-image: -0.0044) but stayed
   noise-level either way.

   **Conclusion, combining both experiments: the null result is not an
   artifact of the single-image setup.** A more realistic joint
   multi-image training regime shows the same lack of effect, which is a
   better-supported negative finding than either experiment alone --
   `concept_misalign.py`'s CLIP-embedding pixel perturbation does not
   measurably redirect what a LoRA learns to associate with an image's
   caption, under either training configuration tested. Stays strictly
   opt-in with no default-on path, now for a stronger reason than
   "unvalidated" -- it's validated twice and the effect wasn't there. Full
   per-run numbers in `experiments/concept_misalignment_validation/
   out/report.txt` and `out_multiimage/report_multiimage.txt` on the GPU
   PC (not committed -- generated output, see that experiment's
   `.gitignore`).
   (`orchestrate.py`, `server.py`'s HTTP API does not expose it at all,
   matching `select_style_target.py`'s existing env-var-only, no-HTTP-
   exposure pattern for the same "no curated pool/no validation yet"
   reason).

## 2. Honeypot assets / honeypot URLs

### Direct extension of what `delivery-gateway` already built

`apps/delivery-gateway`'s crawler classification
(`src/crawlers.rs`/`is_known_ai_crawler`) currently does one of two things
per PROJECT_DESIGN.md §3-5's "차단 또는 decoy": it blocks (`403`). The
decoy half was explicitly deferred to this phase
(`apps/delivery-gateway/README.md`'s "What this does not do").

### Design (as originally planned)

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

### What's actually built (`apps/delivery-gateway/src/honeypot.rs`), and how it differs

**Honeypot URLs are built as planned** -- `GET /decoy/:token`, never linked
from any real page, seeded only into `robots.txt`'s `Disallow` list
(`HONEYPOT_TOKENS` env, or one random token auto-generated at startup).
Same "hit is unambiguous by construction" reasoning as above, unchanged.

**Honeypot *assets* were simplified, not built as originally planned.**
The per-hit-unique-watermark-payload design above requires the decoy to be
served through the *same* real-artwork signed-URL path a known crawler
already reached (`render_asset`'s step 2, the existing `403` branch) --
i.e. swapping the `403` for a uniquely-watermarked real image, so a later
leak of that exact payload proves which crawler hit did it. What's built
instead is a dedicated route (`/decoy/:token`) serving one fixed, static
1x1 PNG (`honeypot::DECOY_PNG_1X1`) to every hit, with no per-hit
watermark and no connection to any real artwork's signed-URL flow. This
still delivers the honeypot-*URL* signal (§ above) at full strength, but
gives up the "prove which specific hit leaked" capability the
watermarked-real-image design would have provided. Reason for the
simplification: wiring a *known-crawler-only* branch of `render_asset`
into a per-hit watermark-and-serve call is meaningfully more surface area
(real artwork lookup, real watermark payload generation, real image
encode) for a benefit (leak attribution) that has no way to be exercised
or verified without an actual leaked-payload incident to test against --
same "don't build unvalidatable machinery" reasoning §3 below applies to
adaptive anti-scrape. The static-decoy version is fully testable and
already is (see `apps/delivery-gateway/tests/integration.rs`). Watermarked
per-hit decoys on the real signed-URL path remain real, well-scoped future
work if this project reaches a stage with actual scraper incidents to
attribute.

**Detection loop**: implemented as `HoneypotTracker::record_hit` /
`GET /internal/honeypot-hits` (ops-only, no auth of its own, same trust
boundary as every other `/internal/*` route in this project) -- logs
`(token, ip, user_agent, unix_time)` per hit, in-memory.

**Update**: the "immediately and permanently flag that fingerprint" half
of this section's own recommendation is now implemented, not just logged
-- `HoneypotTracker` tracks a `flagged_ips` set alongside its hit log, and
`render_asset` blocks any request from a flagged IP with `403` before the
rate limiter or enumeration detector even run (both of those stay
soft/resettable, since they reason about ambiguous signals a real heavy
user could trip; a honeypot hit has none of that ambiguity -- see
§3 below for why that made it "worth building now" rather than the fuller
scoring loop). Tested in
`apps/delivery-gateway/tests/integration.rs`'s
`a_honeypot_hit_blocks_that_ip_from_a_later_real_render_request`.

### Where it plugs in

`apps/delivery-gateway/src/honeypot.rs`, alongside `src/enumeration.rs` and
`src/rate_limit.rs` -- same in-memory `DashMap`-backed-state shape, wired
into `AppState`/`build_router` in `src/lib.rs`.

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

**Update**: one more piece of "worth building now" surfaced without
needing real production data -- §2's honeypot hits were already
unambiguous by construction (no real user can ever trigger one), which is
exactly the property that made turning a hit into an immediate, permanent
IP block safe to ship without traffic to tune against (unlike the
scoring/reputation system above, which genuinely does need that data).
See §2's "Update" note for what's now implemented.

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

**Update**: `contracts/src/OwnershipRegistryERC721.sol` now exists --
written and tested (`contracts/test/OwnershipRegistryERC721.t.sol`),
consistent with the recommendation above in that it's explicitly *not
deployed anywhere* and *not* wired into any running service. Having the
contract ready removes the "would need to write and test this from
scratch" cost from the "revisit later" path this section already
recommends, without pre-committing to the migration itself -- see
`contracts/README.md`'s "ERC-721 migration" section for the deploy/cutover
plan this still leaves as a separate, explicit decision.

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

## 5. Protection-strength visual-quality tuning (teammate exploratory findings, 2026-07-24)

A teammate independently explored two things on a separate local clone
(`work/FK_UNOT_Urs.psd-main`, not merged into this repo -- only their
write-ups, `PROTECTION_STRENGTH_TEST_REPORT.md` and
`LAPTOP_PROTECTION_STRENGTH_HANDOFF.md`, ended up here as untracked files;
the actual `protection_pipeline/` code, sweep scripts, and comparison
images were never shared into this repository): (1) whether tuning
`style_cloak.py`'s existing knobs (VGG19 layer weights, perturbation
frequency placement, `mask_low`/`mask_high` region masking) can push
protection strength higher without visible quality loss, and (2) a
separate, non-`style_cloak` pipeline (P2T/SCL/CML/TL, from an earlier
`laptop-protection-strength-test.zip` handoff) layered on top of the
existing L2 preset.

**(1) Knob-tuning result, at reduced scale (256px, 60 steps, `art2` from a
different 5-image set than this repo's own `starry_night`/etc. benchmark)**:
VGG-layer-weight variants (`shallow`/`middle`/`deep`/`middle_deep`),
frequency-band placement (`low`/`mid`/`high`) and mixing, and region-mask
range sweeps **all failed human visual QA** ("이미지가 깨져 보임") even when
automated style-drift/PSNR numbers looked fine or even improved -- echoing
this repo's own finding in `style_cloak.py`'s `L3_ANTI_TRAIN` comment
block (TV-regularization and LPIPS terms both collapsed the real
adversarial effect while "fixing" the metric). The teammate's own
methodology caveat, and a real one: this was all done at 256px/60 steps
with `clip_transfer_weight` off, not this repo's actual production path
(1024px, official step counts, EOT, perceptual mask, CLIP transfer) --
so **read this as "the fast-search methodology couldn't validate these
knobs," not "VGG-layer/frequency/mask tuning is proven not to work."** A
real 1024px/300-step run of the official L2 preset on one image
(`experiments/run_official_l2_art2.py`, prepared but never executed --
estimated 2-4h/image on their CPU-only environment) is the natural next
step if anyone picks this back up, before spending more time on knob
variants at reduced scale.

**(2) The separate P2T/SCL/CML/TL pipeline**: not this repo's
`style_cloak.py`/`concept_misalign.py` -- different loss functions,
parameters that don't map 1:1 to `epsilon`/`steps`/`color_weight`. Reached
"L2+2" (a small step past the existing `L2_PORTFOLIO` baseline) with no
visible degradation by teammate's own eye, `L3_ANTI_TRAIN`-equivalent and
`L3_BALANCED_CANDIDATE` variants both rejected for visible noise/artifacts
in flat regions -- same failure mode this repo's own `style_cloak.py`
`L3_ANTI_TRAIN` comment already documents for its color-balance/mask
tuning. **No LoRA-attack validation of this pipeline has been run** (planned
for "final candidate only, on Colab GPU," never reached). Given this
repo's own `PHASE4_SCOPING.md`/experiment history (see [[lora-protection-research]]
memory) found that *five different* mechanisms -- style_cloak,
concept_misalign, diffusion_attack, aspl_attack, hybrid_attack -- all
failed to produce a statistically significant real-LoRA protection effect
despite each one's own proxy metric moving in the intended direction, the
prior for P2T/SCL/CML also passing a real LoRA-attack test should be set
low until it's actually run against the same 5-image/2-seed/real-training
benchmark this repo's `experiments/*_validation/` directories already use
-- visible-quality tuning alone was never the bottleneck in any of those
five; the proxy-to-real-training transfer gap was.

## 6. Real architecture-diverse ASPL attack (2026-08-06) -- first validated protection effect, and why it isn't wired into `orchestrate.py` yet

### The result, stated plainly

`ml-engine/src/ensemble_attack_multiarch.py` (`multiarch_ensemble_attack`,
`MULTIARCH_FULL` preset) is a sixth attempt after the five documented
failures above (§5's closing paragraph), and the first to pass. Where the
five failures each attacked a single surrogate model (or, for
`ensemble_attack.py`, several differently-configured LoRA adapters on *one*
shared SD1.5 backbone -- the most VRAM this project's original 8GB GPU
could hold), this one alternately fine-tunes and PGD-attacks two genuinely
separate frozen backbones at once, SD1.5 and SDXL, fp32, at
`aspl_attack.py`'s own `L3_ANTI_TRAIN` iteration counts (outer_iters=30,
surrogate_steps=3, pgd_steps=6) -- a combination that needs more VRAM than
this project had access to until moving the validation runs to a rented
RunPod A40 (48GB). Validated against 10 synthetic anime-style original-
character illustrations (generated via Illustrious-XL specifically to
avoid scraping a real artist's copyrighted work for a protection-research
benchmark -- see [[lora-protection-research]] memory) x 3 seeds = n=30, real
`train_network.py`/`sdxl_train_network.py` LoRA trainings, CLIP-similarity
delta scored the same way every prior experiment in this project was:

- **SD1.5: mean delta +0.0414, 95% CI [+0.0225, +0.0603]** on the full n=30
  -- CI excludes zero. Checked twice more before trusting it (this
  project's own stated practice, per §5's closing paragraph, of not
  overclaiming a proxy or single-run result): with the single strongest
  image (`silver_garden`, +0.16-0.17, 3-8x every other image) removed,
  n=27 still gives +0.0272, 95% CI [+0.0162, +0.0383] -- not driven by one
  outlier. Then re-run from scratch on four *new* character illustrations
  never used in the first 30 (`desert_wanderer`/`library_scholar`/
  `ice_skater`/`punk_guitarist`), n=12: +0.0332, 95% CI [+0.0184, +0.0481].
  **All three independent checks exclude zero.** This is the first of nine
  attack mechanisms tried across this project's history to clear that bar.
- **SDXL: does not work, and an earlier claim here was wrong.** The
  original n=30 showed a statistically significant *negative* effect
  (mean -0.0134, 95% CI [-0.0261, -0.0006] -- the attacked LoRA ending up
  *more* similar to the true image than baseline). That did not replicate
  on the four new images (mean +0.0099, 95% CI [-0.0017, +0.0215] --
  includes zero, sign even flipped). The original negative result was very
  likely one image (`winter_scarf`, -0.07 to -0.09, more than double every
  other image's magnitude) doing to the SDXL numbers what `silver_garden`
  did to the SD1.5 numbers -- except SD1.5's effect survived removing its
  outlier and SDXL's didn't survive changing the image set at all.
  **Correct read: SDXL protection has no measured effect, positive or
  negative, not "measured to backfire."**

### Why this can't reuse `remote_gpu.py`'s existing dispatch as-is

Every other mechanism in this document that reached a "wire it in" stage
(`concept_misalign.py`, §1) plugged into the *same* deployment `style_cloak`
already uses -- an opt-in function call inside `protect()`, running either
in-process or delegated to the GPU PC over `remote_gpu.py`'s existing SSH
path (`orchestrate.py:79-84`), on hardware this project already owns and
pays nothing marginal to use. `multiarch_ensemble_attack` cannot reuse that
path:

- **VRAM**: `remote_gpu.py`'s target is the GPU PC's RTX 5060 Ti (8GB) --
  the same card five of the nine attack mechanisms in this project's
  history were constrained by. Loading SD1.5 (~2GB) + SDXL (~7GB) fp32
  plus two LoRA surrogates' optimizer state does not fit; this mechanism
  was specifically *un*-runnable until the RunPod A40 (48GB) migration.
- **Latency**: `style_cloak.cloak()` at production settings (1024px, EOT)
  measures ~129s/image (`ml-engine/README.md`) -- already slow enough that
  `server.py` made `/protect` an async job-queue (`202` + poll) rather than
  a synchronous response, single-worker due to the 8GB VRAM ceiling. A
  single `multiarch_ensemble_attack` call (just the attack step -- a real
  deployment needs none of the validation experiments' LoRA-training/
  scoring steps, only `multiarch_ensemble_attack()` itself) scales from
  this project's own measured `CALIBRATION`-preset timing (175s for
  outer_iters=10) to `MULTIARCH_FULL`'s outer_iters=30 at roughly the same
  per-iteration cost -- **on the order of 15 minutes per image**, not
  2 minutes. An order of magnitude slower than the mechanism already
  identified as this pipeline's latency bottleneck.
- **Cost**: the GPU PC is hardware this project already owns; `style_cloak`
  and `concept_misalign` cost nothing marginal per request. An A40 has no
  local equivalent here -- running this mechanism means renting cloud GPU
  time, at a real, recurring, per-request dollar cost (RunPod A40 secure-
  cloud measured at $0.44/hr during this validation; a ~15min request is
  roughly **$0.11 of GPU time per protected image**, before accounting for
  cold-start/idle overhead on a rented-by-the-hour pod). That is a product/
  pricing decision (who pays it, and whether it's worth it), not just an
  engineering one -- exactly the kind of thing this document's other
  sections (§3's "don't build unvalidatable machinery," §4's mainnet gas
  cost model) already treat as a real gate, not a footnote.

### Recommendation (2026-08-06, updated -- A40-class capacity is available)

Originally scoped this section around RunPod Serverless specifically
*because* renting a by-the-hour A40 pod only for occasional per-request use
looked wasteful next to the owned, marginal-cost-free GPU PC. That
tradeoff changes if a 48GB-class GPU is available on an ongoing basis
rather than spun up per request (confirmed feasible for this project,
2026-08-06) -- at that point the simpler, more consistent design is to
treat it the same way this codebase already treats its one existing remote
GPU target, not build a second, differently-shaped dispatch mechanism next
to it. Still **do not** fold this into `protect()`'s default path, and do
not present it as "AI-training protection" without qualification -- it is
**SD1.5-training protection**, measured; SDXL is an open question, not a
covered case.

1. **New opt-in tier, not a `protectionProfile` value.** `server.py`'s
   `protectionProfile` is validated against `style_cloak.PRESETS`
   (`server.py:192`) -- this needs its own parameter, same shape as
   `concept_misalign_target_path`'s opt-in flag (`orchestrate.py`), e.g.
   `strong_protection: bool`, defaulting to off so every existing caller's
   behavior stays byte-for-byte unchanged, matching this project's own
   established pattern for adding an unproven-at-scale mechanism next to a
   proven one.
2. **Extend `remote_gpu.py`'s existing pattern with a second remote
   target, not a new dispatch mechanism.** With an A40-class pod kept
   available rather than rented per-request, `remote_gpu.py`'s SSH-to-a-
   known-host shape (`_connection()`, `GPU_HOST`/`GPU_USER`/`GPU_SSH_KEY`
   env vars) fits this directly -- add a parallel set of env vars
   (`MULTIARCH_GPU_HOST` etc., or a `GPU_TARGET=multiarch` selector reusing
   the same three) and a `remote_multiarch_cloak()` function alongside
   `remote_cloak()`, SSHing to the A40 pod's `kohya_ss`/`ml-engine` venv
   the same way `remote_cloak()` already SSHes to the GPU PC's, instead of
   standing up a separate Serverless worker image/API integration for a
   single function call. Simpler, reuses code this project has already
   tested in production, and avoids adding a second remote-execution
   pattern to maintain. (RunPod Serverless remains worth revisiting later
   specifically if request volume grows enough that scale-to-zero starts
   mattering for cost -- not needed at this project's current scale.)
3. **Async job-queue already fits the latency, no new architecture
   needed** -- `/protect` is already `202` + poll (`server.py`), built
   because `style_cloak` was already too slow to be synchronous. A ~15min
   `strong_protection` job is a longer wait on the same mechanism, not a
   new one.
4. **Surface the scope honestly wherever this is offered** -- an
   "SD1.5 only" caveat is a product-facing fact, not an implementation
   detail to bury, independent of who's paying for the GPU time.

Full experimental record, robustness checks, and replication numbers: see
the `lora-protection-research` memory (this session's own working notes,
not part of this repo) and `apps/protection-svc/ml-engine/experiments/
ensemble_validation/run_multiarch_n30.py`/`run_multiarch_n30_score.py`.

**Update (2026-08-06) -- built and verified live, with one real design
change from the recommendation above.** Given this project's actual
current traffic (development stage, sporadic real requests), keeping an
A40-class pod running continuously turned out not to be the cost-effective
choice item 2 above assumed -- comparing real RunPod pricing at build
time, A40 secure-cloud ($0.44/hr) was *cheaper* than every 24GB-class
alternative checked (RTX 3090 $0.50/hr, L4 $0.49/hr, RTX 4090 $0.74/hr),
so downsizing the GPU wouldn't have helped either. The design that
actually shipped: build `docker/strongprotect/Dockerfile` -- a
self-contained image with the checkpoints and exact dependency set
`multiarch_ensemble_attack()` needs already baked in (SD1.5+SDXL,
diffusers/transformers/peft/torch pinned to this project's validated
versions -- deliberately *not* a kohya_ss clone, since production only
ever calls the attack directly, never LoRA training) -- and create a fresh
A40 pod from that image **per request**, instead of keeping one pod (or a
persistent network volume attached to one) running between requests.
RunPod's own Network Volume feature (the natural way to keep checkpoints
attached across a pod's full lifecycle without baking them into an image)
turned out to have **no data-center overlap with A40 availability** at
build time -- volumes were only offered in a different DC list than the
one with A40 stock -- which is what forced the Docker-image approach over
a network volume in the first place. A first build attempt reused the
`runpod/pytorch` base image (2026-08-05's experiment pods' usual choice)
and came out to 49.4GB, almost entirely jupyter/filebrowser/multi-Python-
version tooling this headless worker never touches; rebuilt on a plain
`nvidia/cuda` runtime base with a hand-written minimal SSH setup instead,
down to 36GB. Verified for real: pushed to Docker Hub, created a fresh A40
pod from the image with no setup step, confirmed the checkpoints/deps were
already present, and ran `remote_multiarch_cloak()` against it successfully
(165s attack, valid output) -- pod creation to a completed attack took
under 8 minutes total, image pull+extract included, with no manual setup
step in between. `remote_gpu.py`'s `remote_multiarch_cloak()` (added
alongside `remote_cloak()`, per item 2's original design) is unchanged by
this -- it dispatches over SSH to whatever `MULTIARCH_GPU_HOST` currently
points at, whether that's a long-lived pod or a freshly-created one; only
which of those a caller creates first changed.

**Update (2026-08-06, continued) -- the "SDXL has no effect" conclusion above was
only true for the *joint* attack; isolated, SDXL passes too.** Built
`ml-engine/src/aspl_attack_sdxl_only.py` (reuses `ensemble_attack_multiarch.py`'s
`SDXLBranch`, otherwise structured exactly like `aspl_attack.py`'s single-
architecture ASPL loop -- same clamp/projection logic, `SDXL_FULL` preset
matching `MULTIARCH_FULL`'s iteration counts) specifically to test whether
SDXL is unresponsive to this attack class, or was merely losing out to SD1.5
for a shared epsilon/optimization budget in the joint design (the same
failure mode §5's `hybrid_attack` already demonstrated once: combining
objectives made things worse, not better). It was the latter. Validated on
RunPod A40 x3 pods in parallel, same methodology/CLIP-scoring as every prior
experiment (see [[lora-protection-research]] memory for full numbers):
original 10-image n=30 (+0.0479, 95% CI [+0.0216,+0.0742]), robust to
removing 1 or 2 outlier images (n=27: +0.0334 [+0.0105,+0.0563]; n=24:
+0.0411 [+0.0174,+0.0647]), and an independent n=12 replication on 4 brand
new images (+0.0539, [+0.0076,+0.1002]) -- **all four checks exclude zero**,
clearing this project's own "don't conclude before replication" bar the same
way the SD1.5 result did. Net effect size (+0.033 to +0.054) matches or
slightly exceeds SD1.5's own validated range.

**Production design implication**: SD1.5 and SDXL must be attacked
**separately** (two independent calls -- `aspl_attack.py` for SD1.5,
`aspl_attack_sdxl_only.py` for SDXL -- each with its own full epsilon
budget), not jointly via `multiarch_ensemble_attack()`. Joint attack is now
a known-worse design, not just an unvalidated one: it measurably suppresses
SDXL's own effect without improving SD1.5's. This roughly doubles per-image
attack time/cost (~10min each vs ~15min combined) but is the only design
that has actually cleared validation for both architectures. The "SD1.5
only, SDXL is an open question" caveat two paragraphs up is now stale --
replace with "both SD1.5 and SDXL are covered, attacked independently" once
this is wired into `remote_multiarch_cloak()`/`orchestrate.py` (not done
yet as of this update -- current `strong_protection` path still calls the
joint `multiarch_ensemble_attack()`).

**Note on RunPod's pod-creation API (2026-08-06)**: this update's pods were
all created manually via the RunPod console, not via this project's usual
`mcp__plugin_runpod_runpod__create-pod` tool -- that tool's REST v2 call
failed with "no instances available" on every GPU type/cloud/image/template
combination tried, while the console (GraphQL-backed) worked normally the
whole time and the account balance/API key were both confirmed fine. Root
cause not fully pinned down (looks like a v2 REST pods-endpoint issue,
possibly specific to this MCP tool's parameter mapping -- its `gpuTypeIds`
array param doesn't match the real v2 schema's single `gpu: {id, count}`
object, discovered by fetching `api.runpod.io/v2/openapi.json` directly).
**This directly undercuts the "spin up a pod per request" architecture
item 2 above assumed** -- if pod creation is this brittle from server-side
automated calls, a real per-request `strong_protection` job could fail the
same way. Worth checking RunPod **Serverless/Endpoints** (a different
product, purpose-built for "spin up a worker per request, scale to zero")
as the actual production mechanism instead of driving the Pods API directly,
before building this out further.

**Update (2026-08-07) -- wired into production and verified live; the
chained composition itself is now validated too, not just each stage
alone.** `remote_gpu.py` gained `remote_dual_arch_cloak()` (runs
`aspl_attack.py` on the original, then `aspl_attack_sdxl_only.py` on that
output) and `orchestrate.py`'s `strong_protection` branch now calls it
instead of the joint `remote_multiarch_cloak()` -- the "SD1.5 only" caveat
above is stale, both architectures are covered. `docker/strongprotect/
Dockerfile` rebuilt with both attack scripts baked in, pushed, and live-
tested end to end against a fresh pod (718.8s, valid output). Separately,
ran `ml-engine/experiments/dual_arch_validation/run_dual_arch_n30.py`
(n=30, same 10-image set) to check whether the *chained* image -- not just
each single-stage attack on a pristine original -- still protects both
architectures: **SD1.5 mean delta +0.1655, 95% CI [+0.1341, +0.1968]**
(4-5x the single-stage effect -- the compounding perturbation behaves like
a much larger effective epsilon budget, not two objectives fighting each
other), **SDXL mean delta +0.0446, 95% CI [+0.0211, +0.0681]**, robust to
removing 1-2 outlier images. Neither effect is suppressed by chaining;
SD1.5's is amplified. Full numbers in [[lora-protection-research]] memory.

**RunPod pod-creation workaround found**: the REST v2 `create-pod` bug
above is real and still unresolved, but GraphQL's `podFindAndDeployOnDemand`
mutation (`https://api.runpod.io/graphql`) works reliably where REST v2
doesn't -- confirmed by manually curling it with `templateId` pointed at
this project's `dontai-strongprotect` template. Filed as a support request
to RunPod with the reproduction details. For any future on-demand-pod
automation, prefer the GraphQL mutation over the REST v2 Pods endpoint
until RunPod confirms a fix. (Practical note: this requires a raw RunPod
API key in the request URL -- store it as a local machine env var, e.g.
`RUNPOD_API_KEY`, referenced inline in commands, rather than pasting it
into chat -- a full-account-scope credential shouldn't be typed into a
conversation transcript.)

**Update (2026-08-07) -- tested RunPod Serverless empirically against the
Pods approach; Serverless wins for this workload, not just Pods'
create-pod bug.** Built `docker/strongprotect-serverless/` (thin layer on
`lentropy/dontai-strongprotect:latest` -- adds the `runpod` SDK and a
`handler.py` wrapping the same dual-arch attack `remote_dual_arch_cloak()`
runs over SSH), pushed it, and ran a real job through `create-endpoint`/
`run-endpoint` (both worked fine via the REST v2 API -- the create-pod bug
above is specific to the Pods endpoint, not RunPod's v2 API generally).

Real measured numbers, one A40 job, same dual-arch attack:
- Pure execution time: Serverless 685.1s vs Pods 718.8s -- essentially the
  same (Serverless if anything slightly faster).
- Total time for a request when the image is already cached on the host
  but the worker had scaled down: Serverless ~11.6min (11s queue delay +
  685s execution -- just a container restart, no image re-pull) vs Pods
  ~15-20min (full pod creation, networking, SSH setup every time -- Pods
  has no "warm but scaled down" state, only "exists" or "doesn't").
- **Idle-cost risk**: Serverless scales to zero automatically
  (`idleTimeout`, tested at 60s) -- structurally cannot leak a forgotten
  idle resource. Pods requires us to remember to delete every one --
  this session found and deleted a pod that had been sitting idle for
  4.5 hours before anyone noticed, for exactly this reason.
- **API reliability**: `create-endpoint`/`run-endpoint`/`get-job-status`
  all worked on the first try, no GraphQL workaround needed (unlike
  `create-pod`).

**Recommendation**: build any future on-demand `strong_protection`
automation on Serverless (`create-endpoint` + `run-endpoint`/`runsync-endpoint`),
not the Pods API -- better matches this project's own "sporadic, low-volume,
cost-sensitive" traffic pattern (item 2/3's own framing above), avoids the
Pods create-pod bug entirely, and removes the idle-pod-cleanup failure mode
by construction. `docker/strongprotect-serverless/Dockerfile` and
`handler.py` are a working starting point, not yet wired into
`orchestrate.py`/`remote_gpu.py` (that integration -- calling the endpoint's
`run`/`status` URLs instead of SSH -- is real future work, not done as
part of this comparison).

**Update (2026-08-07) -- wired into production; this is now
`strong_protection`'s real mechanism, not just a comparison artifact.**
Created a persistent Serverless endpoint (`dontai-strongprotect`, id
`wtutl0miby687b`, `AMPERE_48`/A40, `workersMin=0`/`workersMax=2`,
`idleTimeout=60s`, from the already-pushed
`lentropy/dontai-strongprotect-serverless:latest` image) via
`create-endpoint`, and live-smoke-tested it end to end with a real job
(fast `L1_PREVIEW`/`CALIBRATION` presets, not the full production ones --
this was a wiring/connectivity check, not another effect-size
measurement): submitted, worker cold-started in ~17s, completed in 148s,
returned a real, valid 1024x1024 PNG. `remote_gpu.py` gained
`serverless_dual_arch_cloak()` (submits to `.../run`, polls `.../status/{id}`
until `COMPLETED`/`FAILED`/`CANCELLED`/`TIMED_OUT` -- same two-phase job
shape as every other polling loop in this project, not `runsync`, since
a real production job runs several minutes) and `orchestrate.py`'s
`strong_protection` branch now calls it instead of the SSH-based
`remote_dual_arch_cloak()` (still in `remote_gpu.py`, kept for manual/
debugging use against a hand-started pod, just no longer production's
default path). Needs `RUNPOD_API_KEY` and `RUNPOD_STRONGPROTECT_ENDPOINT_ID`
set wherever protection-svc actually runs -- falls back to `style_cloak`
on any failure (missing env vars, endpoint unreachable, job failed), same
as the SSH path always did.
