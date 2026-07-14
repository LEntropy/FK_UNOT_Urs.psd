# web

PROJECT_DESIGN.md §2's `apps/web` (React + TS + Vite, TanStack Query,
Zustand). Talks only to `apps/api-gateway` (`VITE_API_GATEWAY_URL`) --
never calls asset-service/blockchain-svc/protection-svc/detection-svc
directly.

## Scope (4 screens)

- **Login / Signup** -- email + password, stores the JWT pair in
  `localStorage` via `src/store/auth.ts`.
- **Upload** -- title + server-local image path (no object storage yet,
  see `apps/asset-service/README.md`) + protection preset.
- **Gallery** -- `GET /artworks` (scoped to the logged-in user).
- **Artwork detail** -- `GET /artworks/:id`, polls every 2s until the
  status machine reaches `PUBLISHED`/`FAILED`, links out to the Polygon
  Amoy explorer once an on-chain tx exists.

## What this does not do

Social login, community features (feed/follow/like/comments), a real file
uploader, detection-svc UI -- all explicitly out of scope for this pass.

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
