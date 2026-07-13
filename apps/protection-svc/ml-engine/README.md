# protection-svc / ml-engine — style-cloak PoC

A from-scratch, minimal reimplementation of the *mechanism* Glaze uses
("style confusion" via adversarial perturbation against a feature
extractor's style representation), proving out the optimization framing in
`PROJECT_DESIGN.md` §3-3/§8:

```
maximize   Feature_Drift          (style embedding measurably moves)
subject to Perceptual_Distance < epsilon   (pixels barely change to a human)
```

This is a PoC, not a production port of Glaze/Nightshade — see
`PROJECT_DESIGN.md` §12 for the project-wide caveat that no cloaking method
gives 100% protection.

## Files

```
src/
  model.py           VGG19 feature extractor -> Gram matrices at 5 layers
                      (same layer convention as Gatys et al. neural style transfer)
  style_cloak.py      the cloaking optimization (PGD/Adam, L-infinity bounded);
                      --eot optimizes against random resize round-trips too
  evaluate.py         quantifies style drift + perceptual preservation
  robustness_test.py  re-measures style drift after JPEG recompression /
                      resize round-trips, to see how much survives a real
                      upload pipeline instead of only the exact original file
  perceptual_hash.py  standard DCT pHash -> the bytes32 `perceptualHash`
                      blockchain-svc's on-chain content hash needs (NOT the
                      same thing as the Gram-matrix similarity above)
scripts/
  generate_test_images.py   synthetic original/style-target images
                             (no copyrighted art needed to run this PoC)
remote/                GPU-offload workflow (see remote/README.md) — runs the
                        same scripts on a second PC with an NVIDIA GPU over SSH
out/                   generated images (gitignored except .gitkeep)
```

## Quick start

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
./.venv/Scripts/python.exe -m pip install Pillow numpy

./.venv/Scripts/python.exe scripts/generate_test_images.py
./.venv/Scripts/python.exe src/style_cloak.py --preset L3_ANTI_TRAIN
./.venv/Scripts/python.exe src/evaluate.py
```

Swap `--index-url .../cpu` for a CUDA build (see `remote/README.md` for the
GPU-specific gotchas we hit) if you have an NVIDIA GPU locally.

## Protection strength presets

Maps to `PROJECT_DESIGN.md` §3-4. Only the style-confusion layer is
implemented here; concept-misalignment (Nightshade-style) is Phase 4, not in
this PoC.

| Preset | epsilon (L-inf) | steps |
|---|---|---|
| `L1_PREVIEW` | 0.02 | 150 |
| `L2_PORTFOLIO` | 0.04 | 300 |
| `L3_ANTI_TRAIN` | 0.08 | 500 |

**These epsilon values, and every robustness number in this README, were
originally measured at `cloak()`'s default `size=256`.** `size` is a real,
adjustable parameter (`--size` on `style_cloak.py`/`evaluate.py`/
`robustness_test.py`, not hardcoded to 256 in any way that blocks changing
it). This mattered in practice: `apps/protection-svc/orchestrate.py` (which
wires this together with rust-core's watermark/variants) defaults to 256
for exactly this reason, and that default is why Delivery Gateway's
1280px/2048px variant tiers turned out to be unreachable in the first full
pipeline run — see `apps/protection-svc/INTEGRATION.md`'s "256px processing
size" section for the concrete consequence.

## Re-validated at 1024x1024 — the resize breakpoint is about absolute pixels, not scale ratio

Re-ran the same painting pair (`starry_night_1920.jpg` / `great_wave_1920.jpg`,
downloaded at higher resolution for this) at `--size 1024`,
`L3_ANTI_TRAIN --eot --eot-samples 1` (reduced from `--eot-samples 3` —
see why below):

```bash
./.venv/Scripts/python.exe src/style_cloak.py \
  --original out/real/starry_night_1920.jpg --style-target out/real/great_wave_1920.jpg \
  --output out/real/cloaked_1024.png --preset L3_ANTI_TRAIN --eot --eot-samples 1 --size 1024
./.venv/Scripts/python.exe src/evaluate.py --size 1024 \
  --original out/real/starry_night_1920.jpg --cloaked out/real/cloaked_1024.png --style-target out/real/great_wave_1920.jpg
./.venv/Scripts/python.exe src/robustness_test.py --size 1024 \
  --original out/real/starry_night_1920.jpg --cloaked out/real/cloaked_1024.png --style-target out/real/great_wave_1920.jpg
```

**Raw effectiveness barely changed** — style drift +0.183 (identical to the
256px result), PSNR 23.98 dB (vs 23.6 dB at 256px). The cloaking mechanism
itself generalizes fine to higher resolution.

**Resize robustness changed a lot, in the direction the information-floor
hypothesis predicts:**

| Transform | 256px retained | 1024px retained |
|---|---|---|
| resize 0.5x | +11% | **+34%** |
| resize 0.25x | **-144%** (total collapse) | **-14%** (barely a dent) |
| sns_pipeline | -6% | **+24%** |

At 256px, resizing to 0.25x lands at 64x64 -- deep in the information floor
this project already identified. At 1024px, resizing to 0.25x lands at
256x256 -- the *same absolute resolution* this whole project was validated
at, which turns out to still have "enough room" for the perturbation to
mostly survive. **This confirms the breakpoint was never really about a
0.25x/0.5x scale ratio -- it's about the absolute pixel count the image gets
resampled down to.** A product implication: `variants.rs`'s `Safe`/`Unknown`/
`Unsafe` bands (scale relative to the *protected* image) should really be
thought of in terms of the *resulting* resolution, not the ratio alone --
worth revisiting once this generalizes past a single painting pair.

Why `--eot-samples 1` instead of the usual `3`: the first attempt at
`--eot-samples 3` at this size pushed the GPU (RTX 5060 Ti, 8GB VRAM) to
~94.5% VRAM usage and became dramatically slower than linear scaling would
predict (over 2 hours without finishing, vs. an expected 16-32 minutes by
naive 16x-pixel-count scaling from the 256px timing) -- consistent with
memory-pressure-induced allocator overhead, not a hang (confirmed via
`nvidia-smi` showing 100% GPU utilization and steadily increasing process
CPU time throughout). Killed and re-ran at `--eot-samples 1`, which
completed at a comfortable ~5.6GB VRAM. **Capacity-planning implication**:
`eot_samples` and processing `size` both consume VRAM, and their product
matters more than either alone -- a production GPU worker sizing decision
needs to budget for this combination, not just the larger of the two.

## Measured results (synthetic test image, 256x256)

`avg similarity to style_target` is the mean cosine similarity of Gram-matrix
style vectors across 5 VGG19 layers (1.0 = identical style representation).

| Preset | drift (style sim, before -> after) | PSNR vs original | Verdict |
|---|---|---|---|
| `L1_PREVIEW` | 0.233 -> 0.397 (+0.164) | 35.2 dB | style PASS, perceptual PASS |
| `L3_ANTI_TRAIN` | 0.233 -> 0.630 (+0.397) | 25.3 dB | style PASS, perceptual WEAK |

This is the real trade-off `PROJECT_DESIGN.md` §8 warns about, now measured
instead of asserted: `L3` roughly **2.4x the style drift** of `L1`, at the
cost of dropping ~10 dB of perceptual quality (visible texture noise in flat
color regions — see `out/cloaked_gpu.png` vs `out/cloaked_l1.png`). Neither
number is "correct" in isolation; which preset to ship depends on how much
visible quality loss a creator will accept for stronger protection, which is
exactly why this is an opt-in per-artwork choice in the product design, not
a single global setting.

Reproduce:
```bash
./.venv/Scripts/python.exe src/style_cloak.py --preset L1_PREVIEW --output out/cloaked_l1.png
./.venv/Scripts/python.exe src/evaluate.py --cloaked out/cloaked_l1.png
./.venv/Scripts/python.exe src/evaluate.py --cloaked out/cloaked_gpu.png   # or out/cloaked.png
```

## Robustness to real-world re-encoding

A cloak that only works on the exact original file isn't worth much — most
platforms downscale thumbnails and re-compress to JPEG on upload. Measured
with `src/robustness_test.py` on the `L3_ANTI_TRAIN` cloak:

| Transform | style drift retained |
|---|---|
| none (baseline) | 100% |
| JPEG q95 | 73% |
| JPEG q75 | 60% |
| JPEG q50 | 54% |
| resize 0.5x round-trip | 63% |
| resize 0.25x round-trip | 49% |
| `sns_pipeline` (resize 0.5x + JPEG q75, approximates a typical upload pipeline) | 52% |

Reproduce:
```bash
./.venv/Scripts/python.exe src/robustness_test.py --cloaked out/cloaked_gpu.png
```

Roughly half the effect survives a realistic upload pipeline, not all of it.
This PoC's perturbation is optimized only against the clean image (no
transform-augmented / EOT-style training during optimization) — Glaze's
actual published approach specifically trains against expected
transformations to survive them better, which is real future work here, not
implemented in this PoC.

## Validation on real artwork

The synthetic images (flat shapes vs. random noise) are a clean signal for
debugging the algorithm, but they don't prove anything about real art —
real paintings have dense, correlated texture everywhere, which behaves
differently in a VGG feature extractor than flat color next to noise.
Tested with two public-domain (Wikimedia Commons) paintings with very
different styles: Van Gogh's *Starry Night* (`original`) cloaked toward
Hokusai's *The Great Wave* (`style_target`), `L3_ANTI_TRAIN`:

```bash
./.venv/Scripts/python.exe src/style_cloak.py \
  --original out/real/starry_night.jpg --style-target out/real/great_wave.jpg \
  --output out/real/cloaked.png --preset L3_ANTI_TRAIN
./.venv/Scripts/python.exe src/evaluate.py \
  --original out/real/starry_night.jpg --cloaked out/real/cloaked.png \
  --style-target out/real/great_wave.jpg
./.venv/Scripts/python.exe src/robustness_test.py \
  --original out/real/starry_night.jpg --cloaked out/real/cloaked.png \
  --style-target out/real/great_wave.jpg
```

**Style drift still works, but the baseline is very different from the
synthetic test**: two real paintings already share 0.752 average Gram-cosine
similarity (real art has rich texture everywhere; our synthetic pair was an
intentionally extreme 0.233 baseline), so there's much less "room" to drift
toward the target. Absolute drift is smaller (+0.183 vs +0.397) even though
the mechanism works the same way. Perceptual cost was similar (23.6 dB PSNR,
close to the synthetic test's 25.3 dB for the same preset) — the image is
still recognizable as *Starry Night* to a human, with faint added texture.

**Robustness is dramatically worse on real art than the synthetic test
suggested — this is the most important finding from this test:**

| Transform | style drift retained (synthetic) | style drift retained (real painting) |
|---|---|---|
| none | 100% | 100% |
| JPEG q95 | 73% | 95% |
| JPEG q75 | 60% | 71% |
| JPEG q50 | 54% | 53% |
| resize 0.5x round-trip | 63% | **-26%** |
| resize 0.25x round-trip | 49% | **-148%** |
| sns_pipeline (resize+JPEG) | 52% | **-38%** |

JPEG-only degradation is comparable to the synthetic test. But any resize
round-trip on the real painting doesn't just weaken the cloak — it **reverses**
it: the resized cloaked image ends up *less* similar to the target style than
the plain resized original is. The synthetic test's flat color regions
apparently let the perturbation survive bicubic downsampling far better than
it does against dense real brushwork, where resizing blurs away the
high-frequency adversarial signal faster than it blurs the painting's own
texture.

## Fixing the resize vulnerability with EOT (Expectation over Transformation)

`style_cloak.py --eot` optimizes the *average* style loss over several random
differentiable resize round-trips per step (`random_resize_round_trip` in
`style_cloak.py`), not just the clean image — so the perturbation has to
survive being resized *during training*, not only at inference. Scale can be
sampled either from a continuous range (`--eot-min-scale`/`--eot-max-scale`,
default `[0.3, 1.0]`) or a discrete set (`--eot-scales 0.25,0.5,1.0`) — see
the experiment below for why the discrete option exists. Same preset, same
painting pair:

```bash
./.venv/Scripts/python.exe src/style_cloak.py \
  --original out/real/starry_night.jpg --style-target out/real/great_wave.jpg \
  --output out/real/cloaked_eot.png --preset L3_ANTI_TRAIN --eot --eot-samples 3
./.venv/Scripts/python.exe src/robustness_test.py \
  --original out/real/starry_night.jpg --cloaked out/real/cloaked_eot.png \
  --style-target out/real/great_wave.jpg
```

| Transform | no-EOT retained | EOT `[0.3,1.0]` | EOT `[0.15,1.0]` | EOT discrete `{0.25,0.5,1.0}` |
|---|---|---|---|---|
| resize 0.5x round-trip | **-26%** | **+11%** | +8% | **+18%** |
| resize 0.25x round-trip | **-148%** | -144% | -139% | -139% |
| sns_pipeline (resize 0.5x + JPEG) | **-38%** | **-6%** | -9% | **-3%** |

EOT fixes the sign flip and mostly recovers `sns_pipeline` — for transforms
*inside* the trained scale range, which covers 0.5x resizing. **0.25x never
improves, across three different attempts to fix it:**

1. **Widen the continuous range to `[0.15, 1.0]`** (so 0.25 is technically
   inside it) — no improvement (-139% vs -144%). Hypothesis: maybe uniform
   sampling over a wide range rarely actually draws scales near 0.25, so
   training barely sees it (a *dilution* problem).
2. **Test the dilution hypothesis directly**: switch to a *discrete* scale
   set `{0.25, 0.5, 1.0}`, so scale=0.25 is now guaranteed 1/3 of all EOT
   samples across all 500 steps — no dilution possible. **Still no
   improvement** (-139%). This rules out dilution as the cause.
3. **What did improve**: 0.5x retention got *better* under the discrete set
   (+18% vs +8-11%) — consistent with 0.5 also being sampled more often
   (1/3 of draws vs a small slice of a continuous range). So EOT training
   *does* work exactly as expected for 0.5x; it just never works for 0.25x
   regardless of how it's trained.

**Conclusion: this isn't a training/sampling problem, it's an information
floor.** At `256x256 -> 0.25x`, the image is downsampled to `64x64` before
being upsampled back — at that resolution there may simply not be enough
pixels left to encode the adversarial perturbation, no matter how much
training exposure it gets. EOT can teach a perturbation to survive a
transform it has "room" to survive; it can't manufacture information-space
that the transform destroys outright. If 0.25x-scale robustness matters in
production, the fix isn't more EOT training — it's more research into what,
if anything, functions at that level of information loss (possibly nothing
does, which would itself be a useful thing to know before promising it to
users).

Trade-off, holding scale strategy aside: EOT drift at the identity transform
is consistently a bit lower than non-EOT (+0.17 vs +0.18) and costs roughly
`eot_samples + 1`x the compute per step (every step now runs the feature
extractor once per transform sample, not once). Robustness against a fixed
transform budget isn't free, and — per above — isn't unlimited either.

## perceptualHash (the content-identity fingerprint blockchain-svc needs)

`evaluate.py`'s Gram-matrix cosine similarity answers "does this look like a
different *style*" — useful for grading the cloak, useless as a content
fingerprint (it's not hashable/comparable the way `blockchain-svc`'s
`computeContentHash` needs; see `apps/protection-svc/INTEGRATION.md`, which
previously flagged this as unimplemented). `perceptual_hash.py` closes that
gap with a standard DCT-based pHash (via the `imagehash` library, not a
from-scratch reimplementation — nothing to gain from hand-rolling an
algorithm with no research judgment calls in it), at `hash_size=16` so the
output is exactly a 256-bit / 32-byte hash — no padding or truncation needed
to match blockchain-svc's `bytes32`.

```bash
./.venv/Scripts/python.exe -m pip install ImageHash scipy
./.venv/Scripts/python.exe src/perceptual_hash.py out/real/starry_night.jpg out/real/cloaked.png out/real/great_wave.jpg
```

Validated against exactly the properties a content-identity hash needs
(Hamming distance out of 256 bits — lower means "more likely the same
underlying image"):

| Comparison | Hamming distance | Expected? |
|---|---|---|
| original vs. JPEG q75 recompress of itself | 0 | yes — pHash is *designed* to be robust to this, unlike our adversarial cloak |
| original vs. resize 0.5x round-trip of itself | 0 | yes, same reason |
| original vs. our own cloaked output (`L3_ANTI_TRAIN`) | 6 | yes — epsilon-bounded perturbation is small, should register as "same image, lightly modified," not a different image |
| original vs. `cloaked_eot.png` | 8 | yes, similarly small |
| original vs. an unrelated painting (*The Great Wave*) | 138-142 | yes — close to the ~128 expected for unrelated hashes, correctly reads as "different image" |

This is exactly the behavior both consumers of `perceptualHash` need: a
small Hamming distance from the original is fine (even expected) for the
on-chain registration use case (the protected image is still identifiably
"this artwork"), while the future Monitoring & Detection service
(`PROJECT_DESIGN.md` §3-7) can use a Hamming-distance threshold to flag
near-duplicates/derivatives found elsewhere on the web, distinct from
completely unrelated images.

## LoRA validation experiment (real training, not a proxy)

Every number elsewhere in this README — style drift, PSNR, robustness
percentages — comes from **VGG19 Gram-matrix cosine similarity**, the same
feature space `cloak()` optimizes its perturbation against (`model.py`'s
`StyleFeatureExtractor`). That's a legitimate way to check "did the
optimization do what it was told to," but it says nothing about whether
the perturbation transfers to what a real LoRA fine-tune actually learns
from — a diffusion U-Net's latent/cross-attention representations are
architecturally unrelated to a VGG19 classifier's features. Despite LoRA
degradation being a stated goal (`PROJECT_DESIGN.md` §3-3), **no version of
this project had ever actually trained a LoRA and checked**, cloaked or
not, until this experiment.

**Method**: SD1.5 LoRA training (`network_dim=32`, `network_alpha=16`,
`AdamW8bit`, `lr=5e-5`, 10 epochs / 200 steps, resolution 512 —
kohya_ss's `sd-scripts/train_network.py`), single-image style-overfit on
20 repeats of a real painting, the only variable per pair being whether
that image was cloaked first (`L3_ANTI_TRAIN`, cloaked *at the actual
training resolution*, 512px, not the 256px this project's epsilon/EOT
numbers were otherwise tuned at). Each trained LoRA then generated 6
samples, scored by **CLIP image-image cosine similarity** against the
true uncloaked painting — CLIP is architecturally independent of the
VGG19 space the cloak optimizes against, so unlike this README's other
metrics, this one isn't circular with the attack's own objective. Scripts:
`experiments/lora_validation/{prepare_dataset,generate_and_score}.py`,
orchestrated by `remote/run_lora_validation.ps1`.

**First pass** (one image, one seed) found delta +0.0315 and reported it
as a "PASS." **That result did not replicate at n=6** (2 paintings × 3
seeds, mean delta +0.0137, 95% CI included zero). Expanded further to 4
real paintings spanning very different subjects/styles — `starry_night.jpg`
(post-impressionist landscape), `great_wave.jpg` (ukiyo-e woodblock
landscape), `mona_lisa.jpg` (renaissance portrait), `the_scream.jpg`
(expressionist portrait/figure), each cross-cloaked toward one of the
others — × 3 training seeds each (n=12 image×seed combinations).

**Bug found along the way**: the first 4-image run used one hardcoded
generation prompt suffix ("oil painting, landscape") for every image. That
actively fights a portrait subject — with the text encoder untrained
(`--network_train_unet_only`), the LoRA's trigger word is weak signal
next to a strong, contradictory subject word already in the prompt.
Generated samples for `mona_lisa`/`the_scream` came out as unrelated
landscapes for *both* conditions, not because of cloaking — confirmed by
eye, and by CLIP-similarity scores sitting oddly low (~0.60-0.64) for
those two images versus ~0.85-0.92 for the other two. Fixed by giving
each image its own subject-matching prompt suffix
(`prepare_dataset.py`'s `IMAGE_CONFIGS`) and re-running generation+scoring
only (no retraining needed — the LoRA weights themselves were unaffected).
Post-fix, `mona_lisa`/`the_scream` scores moved into the same ~0.74-0.83
range as the other two images, and a spot-check generation actually looks
like a portrait now (gilded frame, three-quarter profile) rather than a
random landscape — this is the corrected, trustworthy run:

| image | seed | baseline CLIP sim | cloaked CLIP sim | delta |
|---|---|---|---|---|
| starry_night | 1 | 0.8734 | 0.8625 | +0.0109 |
| starry_night | 2 | 0.8875 | 0.8374 | +0.0501 |
| starry_night | 3 | 0.8523 | 0.8477 | +0.0046 |
| great_wave | 1 | 0.9050 | 0.8718 | +0.0332 |
| great_wave | 2 | 0.9207 | 0.8992 | +0.0215 |
| great_wave | 3 | 0.9229 | 0.8801 | +0.0428 |
| mona_lisa | 1 | 0.7483 | 0.7475 | +0.0009 |
| mona_lisa | 2 | 0.7409 | 0.7466 | **-0.0057** |
| mona_lisa | 3 | 0.7573 | 0.7523 | +0.0050 |
| the_scream | 1 | 0.8062 | 0.7616 | +0.0445 |
| the_scream | 2 | 0.7630 | 0.7820 | **-0.0189** |
| the_scream | 3 | 0.8024 | 0.8251 | **-0.0227** |

**mean delta: +0.0138, stdev 0.0246, 95% CI (t-approx): [-0.0001, +0.0278]**
— the interval still just barely includes zero. 8 of 12 runs were
positive (cloak reduced fidelity); the 4 negative runs are split across
`mona_lisa` and `the_scream` specifically, while `starry_night` and
`great_wave` were positive in all 6 of their combined runs. That split by
image (not randomly scattered) suggests the cloak's effect on real LoRA
training may depend on the image/style pair, not be a fixed universal
degradation — plausible, since the cloak targets whatever Gram-matrix
distance exists between the specific original/style-target pair, which
differs per image, but not something this experiment can confirm with 2
images per pattern.

**Honest reading of this, after two rounds of correction**: the original
single-run "PASS" was a real number, correctly measured, non-circular —
and still wrong to generalize from, exactly the failure mode multi-seed/
multi-image validation exists to catch. Even after fixing a real
methodology bug (the prompt mismatch) and doubling the image count, the
aggregate 95% CI still touches zero. The point estimate (+0.0138) and the
8-of-12 positive split are consistent with a real small effect that this
sample size can't yet confirm at 95% confidence — not with "no effect,"
not with "reliable protection" either. Composition/subject learning
survived cloaking in every run regardless of condition (expected for
single-image overfit training). **Do not cite either the +0.0315 or the
+0.0137 numbers as this project's validated LoRA-degradation result** —
+0.0138 at n=12 is the current number, and even that needs more images/
seeds (and ideally a proper paired statistical test, not raw CLIP deltas)
before treating it as more than "a plausible small effect, not yet
statistically confirmed."

## What this PoC does not do (see PROJECT_DESIGN.md §12)

- No concept-misalignment (Nightshade-style) layer — style confusion only.
- EOT here only covers resize; JPEG recompression isn't part of the training
  loop (real JPEG encoding isn't differentiable — would need a differentiable
  JPEG approximation to include it in EOT, not implemented here).
- EOT has a real, measured ceiling: it recovers robustness for moderate
  resizes (0.5x) but three different training strategies all failed to
  recover 0.25x — see the information-floor conclusion above. Any product
  claim about "surviving thumbnailing" needs to specify how aggressive a
  thumbnail, not just assert robustness in general.
