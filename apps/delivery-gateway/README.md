# delivery-gateway

PROJECT_DESIGN.md §3-5: the Delivery Gateway. Rust (axum), the only service
in this stack fronting *served images* rather than JSON APIs -- signed,
short-TTL URLs instead of permanent ones, plus the access-control layer
(referer/UA/rate-limit) that a real deployment needs in front of anything
publicly reachable.

## Why this exists

Every other service in this project (`asset-service`, `blockchain-svc`,
`protection-svc`) answers "what is this artwork's metadata/status/on-chain
record" -- none of them are meant to be the thing a browser's `<img src>`
points at directly. This is that thing, and it enforces the policy
`asset-service` itself doesn't (and per its own README, explicitly
shouldn't -- it "has no auth of its own").

## Scope

- **Signed URLs, not permanent ones** (§3-5's "영구 URL 금지 → 정책 기반
  signed URL"): `POST /internal/sign {artworkId, viewer}` issues
  `/asset/{id}/render?variant=...&exp=...&sig=...`, HMAC-SHA256 over
  `{id}|{variant}|{exp}` (`src/signing.rs`). Binds the signature to one
  specific `(artworkId, variant)` pair and a short TTL (`SIGN_TTL_SECONDS`,
  default 300s) -- a leaked URL for one variant can't be replayed to fetch
  a different one, and stops working shortly after issuance either way.
- **Viewer-based variant selection** (§3-5: "비로그인 → 1280px, 로그인
  유저 → 2048px"): `viewer: "anonymous" | "logged_in" | "thumbnail"` maps
  to `public_preview_1280` / `public_preview_2048` / `grid_thumbnail_512`
  (rust-core's already-built variant names, `apps/protection-svc/rust-core/
  src/variants.rs`). `thumbnail` isn't in the original design text but
  reuses an existing variant for gallery views rather than leaving it
  unreachable through this gateway.
- **Real access control on every render request**, in this order:
  1. Signature + expiry check.
  2. Known-AI-crawler User-Agent block (`src/crawlers.rs`: GPTBot,
     ChatGPT-User, OAI-SearchBot, Google-Extended, ClaudeBot, anthropic-ai,
     CCBot, Bytespider, PerplexityBot, Diffbot) -- `403`, even with an
     otherwise-valid signed token.
  3. Referer allowlist (`ALLOWED_REFERER_HOSTS`) -- hotlink protection.
     A *present* Referer that doesn't match is blocked; a *missing*
     Referer is allowed (direct navigation and privacy-respecting browsers
     routinely strip it -- treating "absent" the same as "disallowed"
     would break normal use, not just hotlinking).
  4. Per-IP sliding-window rate limit (`src/rate_limit.rs`, in-memory --
     see "What this does not do").
  5. Distinct-artwork enumeration detection (`src/enumeration.rs`,
     `PHASE4_SCOPING.md`'s adaptive-anti-scrape item) -- flags an IP that
     touches more than `ENUMERATION_MAX_DISTINCT_ARTWORKS` distinct
     artworks within `ENUMERATION_WINDOW_SECONDS`. **Adapted from the
     original scoping text, not literal "sequential ID" detection**: this
     project's artwork IDs are random 16-hex-char strings
     (`asset-service`'s `ast_${randomUUID()...}`), not a guessable
     sequence -- the applicable signal is the same underlying behavior
     (a scraper touches many distinct artworks fast; a browsing session
     touches a handful) adapted to an ID scheme that was never
     enumerable to begin with. Repeatedly re-requesting the *same*
     artwork (a real user reloading a page) never trips this.
  6. Fetches the artwork's real `assetVersions` from `asset-service`
     (`GET /artworks/:id`) and serves the matching variant's file bytes
     from disk, with `X-Robots-Tag: noindex, noimageindex` and a short
     `Cache-Control`.
- **`robots.txt`** (`GET /robots.txt`, `src/lib.rs`'s `robots_txt`):
  disallows `/asset/` for everyone, plus an explicit per-crawler
  `Disallow: /` for each name in `crawlers::AI_CRAWLER_USER_AGENTS` --
  same list the real enforcement (step 2 above) uses, so the two can't
  drift apart. **Cooperative only, no enforcement power** (§3-5's own
  caveat) -- a crawler that ignores this file entirely still can't get
  past step 2's real check.

## Trust boundary

Same pattern as every other internal call in this project: `/internal/sign`
trusts whatever `viewer` the caller claims, no auth of its own. In the real
stack, `api-gateway` is the only caller (it has already verified the JWT
and knows whether the request is actually authenticated) -- calling this
endpoint directly, unproxied, means the caller is trusted to tell the
truth, same as asset-service trusting a given `creatorId`.

## Quick start

```bash
cargo build
cp .env.example .env
DELIVERY_SIGNING_SECRET=$(node -e "console.log(require('crypto').randomBytes(48).toString('hex'))") cargo run
```

```bash
curl -X POST http://localhost:4500/internal/sign \
  -H 'Content-Type: application/json' \
  -d '{"artworkId":"ast_123","viewer":"anonymous"}'
# {"url":"/asset/ast_123/render?variant=public_preview_1280&exp=...&sig=..."}

curl http://localhost:4500/asset/ast_123/render?variant=public_preview_1280&exp=...&sig=...
curl http://localhost:4500/robots.txt
```

## Tests

```bash
cargo test
```

`src/signing.rs`/`src/crawlers.rs`/`src/rate_limit.rs` each have real unit
tests (signature tamper/expiry/cross-variant/cross-artwork rejection,
crawler substring matching, sliding-window pruning). `tests/integration.rs`
runs the actual axum router end-to-end via `tower::ServiceExt::oneshot`
against a **real** mocked `asset-service` HTTP server (`wiremock`, a real
listener on a random local port, not a mocked function) -- proves the full
request path including the real outbound HTTP call, not just each piece in
isolation.

## What this does not do

- **Rate limiting and enumeration detection are both in-memory,
  single-process.** Fine for this PoC's one instance; a real multi-instance
  deployment needs both in a shared store (Redis) instead, same "not built
  yet" scope note as everywhere else in this project that currently uses
  in-memory state.
- **Enumeration detection is IP-based only.** A determined scraper rotating
  IPs defeats it the same way it defeats the rate limiter -- this is the
  cheap, high-signal check `PHASE4_SCOPING.md` recommended building first,
  not the full behavioral/reputation system it also describes and
  explicitly defers (needs real production traffic to tune against, which
  this project doesn't have).
- **No decoy/honeypot responses.** §3-5 offers "차단 또는 decoy" for
  detected crawlers; this only implements the block half. Honeypot assets
  are explicitly Phase 4 scope (PROJECT_DESIGN.md's Nightshade-style
  concept-misalignment layer + honeypot section) -- connecting to that
  later, not duplicating it here.
- **No object storage** -- variant files are read from a local filesystem
  path (`assetVersions.storageUri`), same PoC-scope limit as every other
  service (`asset-service`/`protection-svc` READMEs).
- **UA classification is an allowlist-of-one-thing** (a denylist of known
  AI crawlers) -- it does not attempt to detect *generic* scraping bots,
  headless browsers, or UA spoofing. A crawler that lies about its UA
  string is indistinguishable from a real browser here.
