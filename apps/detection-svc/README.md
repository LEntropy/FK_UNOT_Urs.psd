# detection-svc

Phase 3 of `PROJECT_DESIGN.md` (§3-7 "Monitoring & Detection", §7 침해 대응
런북): given a registered artwork, find suspected unauthorized copies on the
web and assemble an evidence package for each match.

## Scope

Implements runbook steps 1-3 of §7 only:
1. **탐지/신고 접수** — `POST /scan/{artworkId}` (proactive) or `POST /reports`
   (a caller-submitted suspect URL).
2. **자동 증거 수집** — for each candidate URL: pHash Hamming-distance
   comparison against the artwork's registered hash, rust-core watermark
   detection, a downloaded copy of the image, HTTP headers, and a
   best-effort screenshot (Playwright — skipped gracefully if Chromium
   isn't installed).
3. **증거 패키지 생성** — JSON bundle (always) + best-effort PDF, per
   PROJECT_DESIGN.md §3-7's exact field list.

Steps 4-6 of the runbook (권리자 알림, 대응 옵션 안내, 케이스 추적) are
product/human workflow and are **not** automated here — see
[`RUNBOOK.md`](RUNBOOK.md) for the actual checklist a person follows,
including a DMCA/infringement notice template. `PATCH /cases/{caseId}`
lets that checklist record progress (`NOTIFIED`/`RESOLVED`/`ESCALATED`)
instead of tracking it outside the system entirely.

## What this does not do

- **No signing.** The evidence bundle's `signature` field is always `null`.
  KMS (the signing authority per §6-1) is a separate, in-progress
  workstream — this service flags the gap rather than inventing a
  placeholder crypto scheme that would need reconciling later.
- **Watermark attribution: now per-artwork.** `asset-service` generates a
  random `watermarkPayloadHex` per artwork at creation
  (`routes/artworks.ts`), passes it through to protection-svc's `/protect`
  request, and returns it from `GET /artworks/:id`. `server.py` reads
  `artwork.get("watermarkPayloadHex")` and only falls back to the
  project-wide `DEFAULT_WATERMARK_HEX` constant for artworks created before
  this fix (or rows with it unset for any other reason) —
  `test/test_watermark_fallback.py` covers both paths.
- **Reverse-image search is optional.** Without `GOOGLE_VISION_API_KEY`
  configured, `/scan` still runs pHash + watermark checks against any URL
  supplied via `/reports`, but skips the proactive web-wide search (no
  candidate URLs to check). Uses the plain Vision REST API with an API
  key rather than the `google-cloud-vision` SDK's service-account-key
  flow — many GCP orgs now block service-account key creation by default
  (`iam.disableServiceAccountKeyCreation`), and a plain API key sidesteps
  that entirely.
- **No DB access to asset-service.** Everything needed comes from
  `GET {ASSET_SERVICE_URL}/artworks/:id` — zero coupling to asset-service's
  schema or storage.

## API

- `POST /scan/{artworkId}` → `202 {caseId, status: "queued"}`
- `POST /reports {artworkId, suspectUrl}` → `202 {caseId, status: "queued"}`
- `GET /cases/{caseId}` → case status (`OPEN` → `EVIDENCE_READY` /
  `NO_MATCH_FOUND` / `FAILED`) + evidence record list
- `PATCH /cases/{caseId} {status, note?}` → records a manual runbook step
  (`NOTIFIED`/`RESOLVED`/`ESCALATED` only, and only from `EVIDENCE_READY`
  or another manual state — see `RUNBOOK.md`)
- `GET /evidence/{caseId}` → full JSON evidence bundle(s)
- `GET /health`

## Quick start

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m playwright install chromium   # optional, for screenshots
cp .env.example .env
python server.py
```

## Tests

```bash
pytest
```

All HTTP (asset-service, Vision API) is mocked with `respx`/`unittest.mock`
— no real network calls, no API key needed to run the suite.
