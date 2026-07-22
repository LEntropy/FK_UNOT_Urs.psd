# LoRA validation experiment

Answers a question this project had never actually tested: does
`style_cloak.py`'s cloak degrade real LoRA training, not just the VGG19
Gram-matrix proxy metric every other measurement in `ml-engine/README.md`
reports? Full narrative, results, and honest caveats live in
**`ml-engine/README.md`'s "LoRA validation experiment" section** — this
file is a map of the scripts, not a second copy of the findings.

## Requires

Everything here runs against this project's GPU PC's existing kohya_ss +
SD1.5/SDXL setup (`C:\Users\mello\Desktop\Develop\kohya_ss`,
`C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\`)
— nothing here installs a new training stack or downloads a new
checkpoint. Two Python environments matter:

- **ml-engine's own `.venv`** (plain torch/Pillow, has `style_cloak.cloak()`)
  — runs the `prepare_*.py` scripts (cloaking, dataset assembly). CPU-only
  works but is slow; the GPU PC's copy of this same venv has CUDA torch
  too.
- **kohya_ss's `venv`** (diffusers/transformers/peft/accelerate) — runs
  training (via `accelerate launch .../train_network.py` or
  `sdxl_train_network.py`) and the `score_*.py` / `generate_and_score.py`
  scripts (diffusers pipeline + CLIP scoring).

## Scripts, in the order a full run touches them

| Script | Venv | What it does |
|---|---|---|
| `prepare_dataset.py` | ml-engine | Builds baseline + cloaked kohya-style datasets for all images in `IMAGE_CONFIGS` (currently 10 paintings, cross-cloaked in pairs). Writes `out/manifest.json`. |
| `generate_and_score.py` | kohya | Given trained baseline+cloaked LoRAs per (image, seed), generates samples via diffusers `StableDiffusionPipeline` and scores CLIP similarity to the true painting. Prints the aggregate report (mean/stdev/95% CI). |
| `pairwise_similarity.py` | ml-engine or kohya (CUDA optional) | Standalone utility: Gram-matrix cosine similarity from one base image to a list of candidates, sorted low→high. Used to pick a controlled similarity spread for the dissimilarity experiment below. |
| `correlate_drift.py` | ml-engine or kohya | Follow-up analysis (no retraining): does the cloak's own VGG19 "drift" metric, or the pre-cloak similarity between original and target, predict the real per-image CLIP delta already measured? Two Pearson correlations. |
| `prepare_target_dissimilarity.py` + `score_target_dissimilarity.py` | ml-engine / kohya | Dedicated confirmation: fixes one base image (`starry_night`), cloaks it toward 4 targets at controlled similarity levels, reuses the existing baseline LoRA (doesn't depend on target choice). |
| `prepare_l2_preset.py` + `score_l2_preset.py` | ml-engine / kohya | Does the effect scale with epsilon? Re-cloaks 4 representative images at `L2_PORTFOLIO` instead of `L3_ANTI_TRAIN`, reuses their existing baseline LoRAs. |
| `prepare_sdxl.py` + `score_sdxl.py` | ml-engine / kohya | Does the effect reproduce on SDXL? Cloaks + trains from scratch (both conditions) on Illustrious-XL at 1024px for 2 images. Uses `StableDiffusionXLPipeline`, not the SD1.5 script's `StableDiffusionPipeline`. |

`remote/run_lora_validation.ps1`, `remote/run_target_dissimilarity.ps1`,
`remote/run_l2_preset.ps1`, and `remote/run_sdxl.ps1` (in
`apps/protection-svc/ml-engine/remote/`) orchestrate each pair of
prepare/score scripts end-to-end on the GPU PC — sync this repo there,
then run the relevant `.ps1` file directly (see each one's header comment
for what it assumes already exists, e.g. an existing baseline LoRA).

## Known gotchas hit while building this (fixed, left as comments in the code)

- Windows PowerShell + `$ErrorActionPreference = "Stop"` wraps benign
  native-process stderr (e.g. "triton not found") in a terminating error
  even on exit code 0 — every training call switches to `"Continue"`
  around just that call and checks `$LASTEXITCODE` explicitly instead.
- `train_network.py` (SD1.x) rejects `--cache_text_encoder_outputs` — that
  flag is SDXL-only (`sdxl_train_network.py` accepts it).
- sd-scripts logs some strings in Japanese; Windows' Korean (cp949)
  console codepage can't encode them and crashes `accelerate.print()`
  mid-training — fixed via `PYTHONUTF8=0` / `PYTHONIOENCODING=utf-8:replace`.
- `diffusers`' `pipe.load_lora_weights()` needs `peft` installed — not
  part of kohya_ss's original venv setup, installed as a one-time step.
- A LoRA output folder name that omits the training seed silently lets
  every seed's run overwrite the same file — the scoring script's later
  failure (file not found → diffusers falls back to treating the path as
  a HuggingFace repo id → a confusing `HFValidationError`) is the
  symptom, not the cause; check the actual output directory naming first
  if a `score_*.py` script can't find a LoRA file that training claimed
  to produce.
- SDXL at 1024px on an 8GB GPU runs roughly 250x slower per diffusion
  step than SD1.5 at 512px (~27s/step vs ~0.1s/step) when VRAM is nearly
  saturated — `score_sdxl.py` defaults to fewer samples/steps than the
  SD1.5 scripts specifically because of this, not for consistency with
  them.

## Results: 2026-07-23 re-validation against current `main` preset config

The n=30 result this experiment originally produced (+0.0130 mean delta,
cited in `PHASE4_SCOPING.md`) was measured on the `ai-engine` branch
against a preset config that no longer matches `main`: `style_cloak.py`
picked up 11 commits changing L2/L3's epsilon (roughly halved), plus
`color_weight`, `perceptual_mask`, `clip_transfer_weight`, and AMP, none
of which the original validation ever saw. This re-run answers "does the
*current* preset still degrade real LoRA training" -- reduced to 5 images
x 2 seeds (n=10, GPU-time feasibility) from the original 10 x 3, using the
first 5 images in `prepare_dataset.py`'s own list order (not cherry-picked
for a favorable result -- includes both the two strongest original
positive-delta images and the one image whose effect was inconsistent/
negative before).

```
image           seed   baseline    cloaked    delta
starry_night       1     0.8687     0.8300  +0.0387
starry_night       2     0.8852     0.8350  +0.0503
great_wave         1     0.9089     0.8983  +0.0106
great_wave         2     0.9180     0.8945  +0.0235
mona_lisa          1     0.7527     0.7465  +0.0062
mona_lisa          2     0.7472     0.7455  +0.0017
the_scream         1     0.8030     0.7850  +0.0180
the_scream         2     0.7693     0.8168  -0.0475
composition_vii     1     0.8365     0.8366  -0.0001
composition_vii     2     0.8511     0.8290  +0.0221

n = 10, mean delta = +0.0123, stdev = 0.0264
95% CI (t-approx): [-0.0066, +0.0312]
Verdict (script's own conservative bar, mean > 0.03 AND CI excludes zero): WEAK/FAIL
```

**Reading this honestly**: the mean delta (+0.0123) lands almost exactly
on the original n=30 measurement (+0.0130) despite the preset's epsilon
being roughly halved and three new competing loss terms added on top --
the real, actual evidence is that the preset drift did **not** measurably
degrade this defense. But the 95% CI includes zero at this reduced sample
size (n=10 vs. the original's n=30, which itself needed the full 30 to
get a CI excluding zero) and `the_scream`'s second seed flips sign
entirely (-0.0475) -- the same "small, real, but inconsistent and
image-dependent" effect this experiment already found the first time, not
a stronger or weaker one. Neither the original nor this re-run supports
representing style-cloak's LoRA-training defense as a strong or reliable
guarantee; both support "a small, real, measured effect that sometimes
doesn't show up for a given image."

Raw training/generation logs and the full `generate_and_score.py` report
are on the GPU PC at `C:\dontai-protection-svc\ml-engine\experiments\
lora_validation\out\` (not copied into this repo -- large, low-level logs,
same treatment as every other GPU-experiment run this project keeps out
of version control).
