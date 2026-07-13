# Contributing

## Branch workflow

- `main` is the shared, always-working branch — it currently has the full
  working stack (contracts, blockchain-svc, asset-service, protection-svc,
  detection-svc, CI, docker-compose). Don't push directly to it once your
  own work starts changing things — land it via PR.
- **Long-lived area branches** exist for each in-progress workstream,
  branched off `main`: `backend`, `ai-engine`, `kms`. These are where that
  area's owner does day-to-day work without touching `main` — commit
  directly to your own area branch as much as you want, open a PR into
  `main` when it's ready to share. No area branch exists yet for
  `detection`/`systems`/`contracts` since that work already landed in
  `main`; create one the same way (`git branch <area> main`) if you start
  a stretch of work there that shouldn't hit `main` commit-by-commit.
- **Short-lived feature branches**: for a single focused change, branch
  off either `main` or your area branch, named `<area>/<short-description>`
  (e.g. `backend/fix-409-retry`, `ai-engine/style-loss-v2`,
  `kms/grpc-adapter`). `<area>` matches the sub-project you're touching
  (see root [`README.md`](README.md#repo-layout)).
- Rebase or merge `main` into your branch before opening a PR if it's gone
  stale — don't let conflicts pile up for review time.

## Before opening a PR

Run the test command for whatever you touched (see root README's test
table). CI re-runs all of them regardless, but catching a break locally is
faster than waiting on a CI round-trip.

## PR review

- At least one other teammate reviews before merge — two sets of eyes
  catch more than CI does, especially for the parts CI can't check (ML
  algorithm choices, contract logic, API contract changes).
- All CI jobs must be green before merging (see
  `.github/workflows/ci.yml` — one job per sub-project, plus a
  docker-compose build check).
- **If your change touches a shared API contract** (e.g. the JSON shape
  `protection-svc` returns to `asset-service`, or what `asset-service`
  returns from `GET /artworks/:id` that `detection-svc` reads), update the
  relevant `INTEGRATION.md`/README in the same PR — don't let docs drift
  from what the code actually does. This project has repeatedly caught
  real bugs specifically because those docs describe measured behavior,
  not aspirational behavior; a stale doc defeats that.

## Cross-team touch points (read before changing these)

A few files are read by more than one sub-project's owner. Changing their
shape is a cross-team change, not a local one:

- `asset-service`'s `GET /artworks/:id` response shape — consumed by
  `detection-svc` (`apps/detection-svc/src/asset_client.py`). Adding
  fields is safe; renaming/removing existing ones isn't, without checking
  in with whoever owns detection-svc.
- `protection-svc`'s job-result field names (`perceptualHash`,
  `metadataHash`, `watermarkPayloadHex`, ...) — consumed by both
  `asset-service` and (potentially, once wired up per
  `apps/detection-svc/README.md`'s noted gap) `detection-svc`.
- `rust-core`'s CLI surface (`embed`/`detect`/`variants`/... subcommands)
  — both `protection-svc/orchestrate.py` and `detection-svc/src/rust_watermark.py`
  shell out to the same binary and parse its stdout.

## Branch protection (repo admin setup, one-time)

Not yet enabled server-side as of this writing — no automated tool ran
this repo's GitHub settings, so if you're a repo admin, turn these on
under **Settings → Branches → Add branch protection rule** for `main`:

- Require a pull request before merging (at least 1 approval).
- Require status checks to pass before merging — select all jobs from the
  `CI` workflow.
- Require branches to be up to date before merging.
- (Optional but recommended) Restrict force-pushes and branch deletion.
