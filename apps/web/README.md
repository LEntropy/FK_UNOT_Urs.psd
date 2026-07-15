# web

PROJECT_DESIGN.md §2's `apps/web` (React + TS + Vite, TanStack Query,
Zustand). Talks only to `apps/api-gateway` (`VITE_API_GATEWAY_URL`) --
never calls asset-service/blockchain-svc/protection-svc/detection-svc
directly.

## Scope

- **Login / Signup** -- email + password, plus Google/Kakao social login
  (`OAuthButtons`, `OAuthCallbackPage`) -- stores the JWT pair in
  `localStorage` via `src/store/auth.ts`.
- **Upload** -- title + server-local image path (no object storage yet,
  see `apps/asset-service/README.md`) + protection preset.
- **Feed** (`/`, `FeedPage.tsx`) -- 최신/인기/팔로잉 tabs against
  api-gateway's `/feed` proxy.
- **My artworks** (`/my-artworks`, `GalleryPage.tsx`) -- `GET /artworks`
  scoped to the logged-in user (this was the app's home page before the
  feed existed).
- **Artwork detail** (`/artworks/:id`) -- polls every 2s until the status
  machine reaches `PUBLISHED`/`FAILED`, links out to the Polygon Amoy
  explorer once an on-chain tx exists. Once `PUBLISHED`: the actual
  protected image (via `ArtworkImage`, see below), like/follow/report
  buttons, and a comment section.
- **Moderation** (`/moderation`, `MODERATOR`/`ADMIN` only) -- pending
  reports queue, resolve/dismiss. The `NavBar` link and page-level redirect
  are UX convenience only; the real access control is api-gateway's own
  role gate (`src/routes/community.ts`'s `requireModerator`) -- calling
  the API directly with a non-moderator token still 403s regardless of
  what this page does.

### Images are never a permanent URL

`ArtworkImage` (`src/components/ArtworkImage.tsx`) fetches a signed,
short-TTL render URL from api-gateway
(`GET /artworks/:id/render-url`, which is the only trusted caller of
`apps/delivery-gateway`'s `/internal/sign` -- see that service's README)
and points an `<img>` straight at delivery-gateway with it. Refetched
every 4 minutes (delivery-gateway's default TTL is 5) so a long-open tab
doesn't start 403ing on an expired token. There is no code path anywhere
in this app that constructs a permanent image URL.

### Like/follow buttons don't know your prior state

`LikeButton`/`FollowButton` have no "did I already like/follow this"
endpoint to query -- community.ts's schema doesn't track that per-viewer.
They start every page load assuming "not yet", which is honest about the
gap rather than guessing; the underlying like/follow calls are idempotent
either way, so this never desyncs the actual count, it just can't show
"you already did this" across a reload. Documented in the components'
own comments too.

## What this does not do

A real file uploader (still a local path in a text field), a detection-svc
UI (case tracking/evidence review has no frontend yet), and honeypot/decoy
responses (Phase 4 scope, `PHASE4_SCOPING.md`) are the remaining explicit
gaps.

## Quick start

```bash
npm install
cp .env.example .env   # point VITE_API_GATEWAY_URL at a running api-gateway
npm run dev
```

## Production

```bash
npm run build   # -> dist/
npm run serve   # tiny Express static server (server.mjs), matches the
                 # Pi's native-process deployment pattern (no nginx there)
```
