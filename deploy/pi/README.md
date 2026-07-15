# deploy/pi

Native-process launch scripts for the Pi (`Philosophyz.iptime.org`)
(`nohup ... & disown`, one `.log` file per service in
`/media/philosophyz/SSD/dontai/`).

**Correction to an earlier version of this doc**: this host is *not*
Docker-free -- it runs a large number of unrelated containers for other
projects, and (discovered while deploying the runbook/PATCH work)
`detection-svc` itself was actually running as a Docker container
(`ghcr.io/lentropy/dontai-detection-svc:latest`, `restart: unless-stopped`,
host networking) published by `.github/workflows/docker-publish.yml`, not
as a native process like the other five services. That container is now
**stopped** (not removed) in favor of a native-process deployment matching
everything else here, because `docker-publish.yml` appears to not
actually rebuild the image on source changes -- a `docker pull` after
pushing the watermarkPayloadHex/runbook commits returned the same 46-hour-old
image digest with none of that code in it, despite the workflow itself
reporting `success` in about a minute (implausibly fast for what its own
header comment describes as a slow multi-arch build -- almost certainly a
buildx GHA-cache hit that isn't invalidating on source changes). **Not
investigated further** -- flagged here rather than fixed, since the
practical unblock (run it natively like everything else) was faster and
this is worth its own dedicated look. `apps/protection-svc`'s
`docker-publish.yml` job may have the identical problem; unconfirmed.

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
| `start_asset_service.sh` | `asset-service` | 3002 |
| `start_detection_svc.sh` | `detection-svc` (`.venv/bin/python`, not global `python3` -- see script comment) | 8003 |

`start_detection_svc.sh` needs a one-time venv setup on a fresh deploy:
`cd detection-svc && python3 -m venv .venv && ./.venv/bin/python -m pip
install -r requirements.txt`. Also expects `src/` to actually contain all
of `asset_client.py`/`db.py`/`evidence_bundle.py`/`evidence_capture.py`/
`phash_match.py`/`rust_watermark.py`/`vision.py` (mirroring
`apps/detection-svc/src/` in this repo) -- an earlier ad-hoc deploy only
had `server.py`+`db.py` sitting flat, which happened to not matter while
the real traffic was silently going to the Docker container above instead.
