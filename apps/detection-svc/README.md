# detection-svc

Phase 3 of `PROJECT_DESIGN.md` (§3-7 "Monitoring & Detection", §7 침해 대응
런북): given a registered artwork, find suspected unauthorized copies on the
web and assemble an evidence package for each match.

## Scope

Implements runbook steps 1-3 of §7, for two different threats:

**Re-posted/copied images** (the original scope):
1. **탐지/신고 접수** — `POST /scan/{artworkId}` (proactive) or `POST /reports`
   (a caller-submitted suspect URL).
2. **자동 증거 수집** — for each candidate URL: pHash Hamming-distance
   comparison against the artwork's registered hash, rust-core watermark
   detection, a downloaded copy of the image, HTTP headers, and a
   best-effort screenshot (Playwright — skipped gracefully if Chromium
   isn't installed).
3. **증거 패키지 생성** — JSON bundle (always) + best-effort PDF, per
   PROJECT_DESIGN.md §3-7's exact field list.

**Unauthorized model training** — `POST /model-leak-reports {artworkId,
suspectModelUrl}`: given a suspect LoRA `.safetensors` file (e.g. found on
CivitAI/HuggingFace), does generating images from it come out anomalously
close to this artwork? Reuses this project's own LoRA-training-protection
validation methodology (`apps/protection-svc/ml-engine/src/
model_leak_detect.py`) as a detection signal instead — see that module's
doc for the mechanism, and `src/protection_client.py`'s doc for why this
is the one endpoint that calls out to protection-svc (needs real GPU
inference, which this service can't do itself). Compares against the
artwork's *published* protected image, never the private original — the
only version a real infringer could have scraped. Evidence type
`model_leak` (verdict `SUSPECTED_LEAK`) or `model_leak_no_match`.

**Two follow-ups that reduce how much a human has to remember to trigger**:
- **Periodic auto-rescan** (`AUTO_SCAN_ENABLED=1`) -- a background thread
  re-checks every published artwork on a rolling interval
  (`AUTO_RESCAN_INTERVAL_SECONDS`, default weekly), not just at upload
  time or whenever a creator happens to click "웹에서 자동 검색". Creates
  cases with `trigger=auto_scan` (distinguishable from a manual `scan`),
  reuses the exact same evidence pipeline. Off by default -- see
  `server.py`'s own note on why a background thread doing real network
  calls at import time would be a bad default for anything that just
  imports this module (every test file does).
- **Evidence-ready email** -- once a case (from any of scan/report/
  model-leak-report/auto-scan) reaches `EVIDENCE_READY`, the artwork's
  creator gets an email via api-gateway's `POST /internal/notify-evidence-ready`
  (`src/notify_client.py`) -- api-gateway owns the users table/SMTP config,
  same "this service can't do it itself" reasoning as evidence signing.
  Best-effort: SMTP not configured on api-gateway's end, or unreachable,
  both degrade to "no email sent" rather than failing the case.

Steps 4-6 of the runbook (권리자 알림, 대응 옵션 안내, 케이스 추적) are
product/human workflow and are **not** automated here — see
[`RUNBOOK.md`](RUNBOOK.md) for the actual checklist a person follows,
including a DMCA/infringement notice template. `PATCH /cases/{caseId}`
lets that checklist record progress (`NOTIFIED`/`RESOLVED`/`ESCALATED`)
instead of tracking it outside the system entirely.

## What this does not do

- **Evidence signing is best-effort, not guaranteed.** `src/evidence_signing.py`
  calls api-gateway's `POST /internal/sign-evidence` (Ed25519, KMS
  envelope-encrypted key — see `apps/api-gateway/src/evidenceSigning.ts`)
  to fill in the bundle's `signature` field; verified end-to-end against
  the real production KMS server. If api-gateway or KMS is unreachable at
  bundle-build time, `signature` falls back to `null` rather than failing
  the whole case (same best-effort treatment as the screenshot/PDF steps).
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
- `POST /model-leak-reports {artworkId, suspectModelUrl}` → `202 {caseId, status: "queued"}`
- `GET /cases/{caseId}/dmca-notice` → RUNBOOK.md's DMCA template auto-filled from each real evidence bundle (`src/dmca_notice.py`)
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
