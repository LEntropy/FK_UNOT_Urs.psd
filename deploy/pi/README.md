# deploy/pi

Native-process launch scripts for the Pi (`Philosophyz.iptime.org`), matching
the pattern the first four services already used there (`nohup ... &
disown`, one `.log` file per service in `/media/philosophyz/SSD/dontai/`).
No Docker on this host -- earlier attempts at cross-compiling for ARM were
reverted in favor of this (see root git log).

## Secrets are never committed here

`start_api_gateway.sh` sets non-secret config inline (`ASSET_SERVICE_URL`,
`PORT`, `KMS_HOST`/`KMS_PORT`), but `apps/api-gateway`'s actual `.env`
(with `JWT_ACCESS_SECRET`/`JWT_REFRESH_SECRET`) lives **only** on the Pi at
`/media/philosophyz/SSD/dontai/apps/api-gateway/.env` -- generate it there
from `apps/api-gateway/.env.example` with real random secrets
(`node -e "console.log(require('crypto').randomBytes(48).toString('hex'))"`),
never check a filled-in copy into this repo. (`deploy/pi/*.env` is
gitignored specifically because an earlier version of this deploy briefly
committed a copy with the example's placeholder secret still in it, which
had become the *live* signing key for a since-exposed public port --
rotated once caught, but don't recreate the pattern.)

## Scripts

| Script | Starts | Port |
|---|---|---|
| `start_kms.sh` | the C KMS server (`/media/philosophyz/SSD/opt/kms`) | 8443 |
| `start_api_gateway.sh` | `apps/api-gateway` | 4000 |
| `start_web.sh` | `apps/web`'s `server.mjs` (static `dist/` server) | 5173 |
