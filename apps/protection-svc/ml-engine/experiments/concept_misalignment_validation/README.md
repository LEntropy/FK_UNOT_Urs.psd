# Concept Misalignment validation experiment

Answers the question `concept_misalign.py`'s own module doc says plainly
has NOT been answered: does training a real LoRA on a (concept-misaligned
image, real caption) pair make generation from that caption drift toward
the *decoy* concept instead of the true one? See PHASE4_SCOPING.md §1's
"Recommended validation methodology" for the full reasoning behind this
design, and that same section's "What's actually built, and what's
honestly still missing" for exactly what gap this closes.

**Status: prepared, not run.** These scripts exist and are believed
correct (they compile; mirrored closely from
`experiments/lora_validation/`'s already-proven, already-run structure on
the `ai-engine` branch), but nothing here has executed on real GPU
hardware, and no results exist yet. Do not cite a pass/fail verdict for
Concept Misalignment until `run_concept_misalignment_validation.ps1` has
actually been run and `out/report.txt` reviewed.

## Requires

Same GPU PC setup as `experiments/lora_validation/` (see that experiment's
own README for the full environment writeup) -- this project's existing
kohya_ss + SD1.5 setup, nothing new to install except `peft` if not
already present (same one-time step). Two Python environments matter:

- **ml-engine's own `.venv`** (torch/Pillow/open_clip, has
  `concept_misalign.misalign()`) -- runs `prepare_dataset.py`. CPU-only
  works but is slow; the GPU PC's copy of this venv has CUDA torch too.
- **kohya_ss's `venv`** (diffusers/transformers/peft/accelerate) -- runs
  training and `generate_and_score.py` (diffusers pipeline + CLIP
  scoring).

**Known blocker to check first**: `concept_misalign.py`'s own module doc
states that in the environment it was *written* in, downloading
`open_clip`'s pretrained CLIP checkpoint was blocked by that sandboxed
environment's own external-code safety gate, before even a CPU smoke test
could run. `run_concept_misalignment_validation.ps1` checks for this
explicitly before doing anything else (loads `ConceptFeatureExtractor`
once) and throws immediately with this file's context if it fails --
resolve that (approve the download on this machine, or pre-seed the
checkpoint cache) before the real run.

## Scripts

| Script | Venv | What it does |
|---|---|---|
| `prepare_dataset.py` | ml-engine | Builds baseline + concept-misaligned kohya-style datasets for all 5 images in `IMAGE_CONFIGS`, each misaligned toward its own decoy concept image. Both conditions use the *same* trigger/caption (unlike the style-cloak experiment's cross-cloaking) -- see the script's module doc for why that's the point. Writes `out/manifest.json`. |
| `generate_and_score.py` | kohya | Given trained baseline+misaligned LoRAs per (image, seed), generates samples via diffusers and scores CLIP similarity to BOTH the true concept image and the decoy concept image. Reports two deltas (`delta_true`, `delta_decoy`) with 95% CIs, not one. |

`remote/run_concept_misalignment_validation.ps1` (in
`apps/protection-svc/ml-engine/remote/`) orchestrates both scripts plus
the actual LoRA training end-to-end on the GPU PC -- sync this repo
there, then run that file directly.

## Reading the result

Per PHASE4_SCOPING.md §1: "if this specific image is used in fine-tuning,
its caption-to-visual-feature association is measurably wrong" is the
honest claim scope here -- a per-image poisoning effect, not "this
defeats the model." A real pass needs BOTH:

- `delta_true` (baseline's similarity to the true image minus misaligned's
  similarity to the true image) positive, above the same 0.03 threshold
  `lora_validation` uses, with a 95% CI excluding zero -- generation drifts
  *away* from the true concept.
- `delta_decoy` (misaligned's similarity to the decoy concept minus
  baseline's similarity to the decoy concept) positive, same threshold and
  CI requirement -- generation drifts *toward* the decoy concept, not just
  toward noise.

Both conditions matter: a `delta_true` win with a near-zero or negative
`delta_decoy` would mean training just got worse in some generic way (not
evidence of the specific *concept redirection* claim this layer makes),
which is exactly the kind of overclaim PHASE4_SCOPING.md's existing
sections (C2PA, style-cloak LoRA validation) are careful not to make.

Expect the same image-dependent, noisy-per-image, real-in-aggregate result
`lora_validation`'s history already found (starting single-image result
didn't replicate cleanly until n grew to 4, then 6, then 8) -- this is why
`prepare_dataset.py` starts at n=5 images x 3 seeds = 15 runs per
condition, not n=1, and why the verdict logic requires the CI to exclude
zero rather than just checking the mean.
