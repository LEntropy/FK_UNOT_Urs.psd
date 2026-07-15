# Infringement response runbook (steps 4-6)

`PROJECT_DESIGN.md` §7 defines a 6-step runbook. Steps 1-3 (탐지/신고 접수,
자동 증거 수집, 증거 패키지 생성) are automated by `server.py` — a case
reaching `EVIDENCE_READY` means that part is done. Steps 4-6 are explicitly
**product/human workflow, not automated here** (`README.md`'s scope
section) — this is the checklist a person follows for those, and where to
record progress so "케이스 상태 추적" means more than a note kept outside
the system.

> **We provide evidence and response materials, not legal judgment.**
> Whether to pursue a claim, and how, is the rights holder's decision (with
> a lawyer if it goes beyond a platform takedown or DMCA notice) — nothing
> here is legal advice.

## Step 4 — 권리자 알림 (notify the rights holder)

1. Poll or watch for cases reaching `EVIDENCE_READY`:
   `GET /cases/{caseId}` (or `GET /evidence/{caseId}` for the full bundle).
2. Notify the artwork's creator — today this is manual (email / a message
   through whatever the platform's normal notification channel is). A
   "크리에이터 대시보드" surfacing new cases automatically is asset-service's
   future work, not this service's.
3. Record it: `PATCH /cases/{caseId} {"status": "NOTIFIED", "note": "..."}`.
   The note should say *how* they were notified (email address, dashboard,
   etc.) and when — this is itself part of the evidence trail for step 5.

## Step 5 — 대응 옵션 안내 (response options)

Read the bundle (`GET /evidence/{caseId}`) and pick a path based on
`discoveredUrl`:

- **On-platform content** → this is a platform moderation action (remove
  the infringing upload), not something detection-svc does — hand off to
  whatever the platform's takedown/moderation flow is.
- **External site** → send the DMCA/infringement notice template below,
  filled in from the bundle's fields.
- **Suspected AI training dataset inclusion** → attach the bundle's
  `onchainTransaction` record (proves prior registration + the
  `doNotTrain` flag) as your Do-Not-Train evidence. There's no automated
  dataset-scraper-notification flow — this is a manual escalation.

### DMCA / infringement notice template

Fill in the bracketed fields from `GET /evidence/{caseId}`'s bundle.

```text
To: [host/platform's designated DMCA agent or abuse contact]
From: [rights holder name / contact]
Date: [today]

Re: Notice of Copyright Infringement

I am the copyright owner (or authorized to act on the owner's behalf) of
the artwork registered on-chain at:

  Content hash:     [bundle.onchainTransaction.txHash's associated contentHash]
  Chain / registry:  [bundle.onchainTransaction.chain] / [bundle.onchainTransaction.registryAddress]
  Transaction:        [bundle.onchainTransaction.txHash]
  Registered at:       [bundle.registeredAt]

The following URL hosts a copy of this work without authorization:

  [bundle.discoveredUrl]
  Discovered at: [bundle.discoveredAt]

Supporting evidence (attached):
  - Perceptual-hash similarity to the registered work
    (distance: [bundle.phashDistance] / 256, lower = more similar)
  - [If present] Embedded watermark match: [bundle.watermarkDetection]
  - Screenshot and HTTP headers captured at time of discovery

I have a good-faith belief that this use is not authorized by the
copyright owner, its agent, or the law. I swear, under penalty of perjury,
that the information in this notice is accurate and that I am the
copyright owner or authorized to act on the owner's behalf.

Signature: [name]
```

## Step 6 — 케이스 상태 추적 (case tracking)

State machine: `OPEN` → `EVIDENCE_READY` (automated) → `NOTIFIED` →
`RESOLVED` / `ESCALATED` (manual, via `PATCH /cases/{caseId}`).

- `RESOLVED` — the infringing copy was taken down, or the matter was
  otherwise closed (e.g. licensed after the fact). Include what happened
  in `note`.
- `ESCALATED` — going beyond a platform takedown/DMCA notice (e.g. legal
  counsel engaged). This service's involvement ends here; anything past
  this point is off-platform.

`PATCH /cases/{caseId}` only accepts `NOTIFIED`/`RESOLVED`/`ESCALATED`,
and only from a case currently `EVIDENCE_READY` or already in one of those
three states — it's a progress log for the human steps above, not a
general-purpose status override for the automated states (`OPEN` /
`NO_MATCH_FOUND` / `FAILED`).
