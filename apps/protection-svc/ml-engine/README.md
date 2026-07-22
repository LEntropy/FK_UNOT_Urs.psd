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
                      (same layer convention as Gatys et al. neural style transfer);
                      also ConceptFeatureExtractor, a CLIP image-encoder wrapper
                      for concept_misalign.py below
  style_cloak.py      the cloaking optimization (PGD/Adam, L-infinity bounded);
                      --eot optimizes against random resize round-trips too
  concept_misalign.py EXPERIMENTAL/opt-in, validated negative (real LoRA-training
                      tests, single- and multi-image, found no effect) -- see below
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

## Resolution preservation and L3 color-preservation (real user report, fixed)

Two real problems reported after real uploads went through the pipeline in
production, both traced to the same root area and fixed together.

**Resolution was silently capped at `size` (256px), and non-square uploads
were stretched.** `style_cloak.py`'s `image_to_tensor` used to do a plain
`img.resize((size, size))` -- every upload, regardless of its real
resolution or aspect ratio, got squashed into a `size x size` square before
watermarking. This wasn't just a resolution ceiling: a landscape or portrait
photo came out with the *wrong proportions* in the published result, and
`delivery-gateway`'s larger variants (`public_preview_1280`/`2048`) could
structurally never be generated, since the source was never bigger than 256px
on its long edge (`rust-core` never upscales -- see its own README).

Fixed with `letterbox_resize`/`letterbox_content_box`: fit the real image
into the `size x size` canvas without stretching (pad with neutral gray
instead), crop the padding back out after `cloak()` finishes, then restore
the original resolution with a real super-resolution model (EDSR, via the
`super-image` package -- `src/upscale.py`) rather than a naive resize.
Verified against a real 960x645 test image end-to-end through
`orchestrate.py`: the published `watermarked.png` came out at exactly
960x645, and a `grid_thumbnail_512` variant (previously impossible) was
generated at the correct aspect ratio.

**L3_ANTI_TRAIN's color balance visibly shifted ("색감이 이상해").** Already
honestly flagged above as "perceptual WEAK" (25.3 dB, below the 30 dB
"visually near-identical" rule of thumb) -- a real user hit this in
practice. Root-caused with a real sweep on the GPU PC (RTX 5060 Ti), not
guessed:

| Change | styleDriftScore | PSNR | Note |
|---|---|---|---|
| baseline (epsilon=0.08, no color term) | 0.267 | 24.3 dB | original L3 |
| + color_weight=8 (epsilon unchanged) | 0.240 | 25.3 dB | ~1 dB gain, plateaus past weight=4 |
| + epsilon=0.04 (= L2's own epsilon) | 0.221 | 29.3 dB | epsilon is the dominant lever, not color weight alone |
| **final: epsilon=0.05, color_weight=8** | **0.223** | **27.1 dB** | *measured with EOT on, matching production's actual default for this preset* |

A low-pass (16x16 average-pooled) color-preservation loss term
(`color_preservation_loss` in `style_cloak.py`) alone only bought about 1 dB
and stopped improving past `color_weight=4` -- it penalizes a large-scale
tint shift but does nothing about the sheer magnitude of high-frequency
pixel noise an epsilon=0.08 budget allows over 500 steps, which is what
PSNR (a global per-pixel metric) actually responds to. Lowering epsilon
did the real work: `L3_ANTI_TRAIN` now runs at `epsilon=0.05` (was 0.08,
kept above L2_PORTFOLIO's 0.04 so L3 stays the strongest preset) plus
`color_weight=8`. Under EOT (this preset's real production default):
PSNR 23.95 dB -> 27.10 dB (+3.15 dB), while styleDriftScore only dropped
0.249 -> 0.223 (~10%, still far above the 0.05 threshold this file already
treats as a real effect) -- a real, measured improvement to the reported
complaint, not a full fix to "PASS" territory, and not free (a further
epsilon cut would help perceptual quality more but cut into the actual
protection effect this preset exists for -- see the trade-off table above).

**L1/L2 checked too, same real GPU measurement, not assumed fine because
L3 was the one reported.** Same test image, each preset's actual
production EOT default:

| Preset | epsilon | styleDriftScore | PSNR | Verdict |
|---|---|---|---|---|
| `L1_PREVIEW` | 0.02 | 0.191 | 34.5 dB | comfortably PASS, left unchanged |
| `L2_PORTFOLIO` (before) | 0.04 | 0.216 | 29.0 dB | borderline -- just under the 30 dB rule of thumb |
| `L2_PORTFOLIO` (after) | 0.03 | 0.199 | 30.9 dB | color_weight alone only bought +0.13 dB; epsilon was the lever again |
| `L3_ANTI_TRAIN` (after) | 0.05 | 0.223 | 27.1 dB | see above |

`L2_PORTFOLIO` now runs at `epsilon=0.03` (was 0.04) plus `color_weight=8`
(was 0). Its styleDriftScore (0.199) ends up close to L1's (0.191), but L2
still trains for `steps=300` vs L1's 150 -- this isn't "L2 quietly became
L1," it's the same epsilon-was-the-real-lever finding as L3, applied once
epsilon stopped being the *largest* budget in the three-preset lineup and
started being the thing making L2's own output visibly noisier than it
needed to be for a "portfolio" tier.

## Follow-up: the 256px processing cap itself was the bigger problem

Real user report after the fixes above shipped: L3's noise was *still*
too visible, and a high-resolution upload came back looking "뭉개져서"
(mushy/blocky). Root cause: the letterbox fix above stopped the *stretch*
distortion, but `cloak()` was still always processing at a fixed
`size=256` regardless of the real upload's resolution, then relying on
`upscale.py`'s EDSR pass to reconstruct the rest -- for a real 2835x4289
upload, that's an ~11x upscale of a 256px result. EDSR can't recover
detail that never reached the optimizer in the first place, and its job
(produce natural-looking output) partially smooths the adversarial
signal back out on the way up.

Measured directly on the GPU PC against the same real 2835x4289 image
(both compared to a real 1024px-long-edge LANCZOS downsample of the true
original, at `L3_ANTI_TRAIN`'s tuned settings):

| Strategy | styleDriftScore | PSNR | Time |
|---|---|---|---|
| cloak@256 + EDSR upscale to 1024 (the old pipeline) | 0.084 | 27.7 dB | 16 s |
| **cloak directly@1024 (native-ish resolution)** | **0.142** | **32.7 dB** | 129 s |

Not a trade-off -- native-resolution processing won on *both* axes at
once: PSNR crosses into "visually near-identical" territory (+4.9 dB),
**and** styleDriftScore is 69% *higher* (stronger protection, not
weaker), for about 8x more compute. Downsampling to 256 first was
throwing away real image structure the optimizer needed to work against.

Fixed: `server.py`'s `ProtectRequest.size` now defaults to
`orchestrate.py`'s `choose_processing_size()` (the real upload's own long
edge, capped at `MAX_PROCESSING_SIZE = 1024`, matching this project's
own prior "1024px re-validation" precedent) instead of a fixed 256 --
most uploads now need no upscale step at all.

**A second real problem surfaced live while measuring this**: `size=1024`
at the usual `eot_samples=2` pushed the GPU PC to ~96% VRAM and got
dramatically slower than linear scaling predicted -- 2+ hours without
finishing (killed mid-run), the *exact* VRAM-pressure slowdown this
file's own "1024px re-validation" section already documented once before
at `eot_samples=3`. The table above is the *re-run* at `eot_samples=1`,
which is what actually finished in 129s with no quality regression that
mattered. `orchestrate.py`'s new `choose_eot_samples()` applies this
automatically: `eot_samples=2` only for sizes still inside the originally
-validated 256px envelope, `1` for anything above it -- every real upload
big enough to need this size fix in the first place. Both `remote_gpu.py`
(the Pi's actual production path -- `USE_REMOTE_GPU=1`) and the local
`cloak()` call now receive this instead of silently defaulting to 2
regardless of size.

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

## Concept Misalignment Layer (`concept_misalign.py`) — EXPERIMENTAL, validated negative

`PROJECT_DESIGN.md` §3-3's layer [3] (Nightshade's actual mechanism,
distinct from the style-confusion layer above -- see
`PHASE4_SCOPING.md` §1 for the full design). `concept_misalign.py` runs
the same epsilon-bounded optimization shape as `style_cloak.py`, but
against `model.py`'s new `ConceptFeatureExtractor` (a CLIP image encoder,
`open_clip` `ViT-B-32/openai`) instead of VGG19 Gram matrices, pulling the
image's CLIP embedding toward a decoy concept image's embedding.

```bash
./.venv/Scripts/python.exe src/concept_misalign.py \
  --original out/original.png --concept-target out/some_other_concept.png \
  --preset L3_ANTI_TRAIN
```

Wired into `orchestrate.py`'s `protect()` as a fully opt-in
`concept_misalign_target_path` parameter / `--concept-misalign-target` CLI
flag, `None`/unset by default -- every existing caller's behavior is
unchanged unless this is explicitly passed. **Not exposed through
`server.py`'s HTTP API** (same reasoning as `select_style_target.py`'s
env-var-only wiring: no curated decoy-concept pool exists in this repo,
and see the next two points).

**Two real, stated-plainly gaps** (full detail in `PHASE4_SCOPING.md` §1):

1. Targets a decoy *image's* CLIP embedding, not a mismatched *caption's*
   CLIP text embedding — this project's pipeline has no real training
   caption to target against, only `title`/`creator_id`. A reasonable
   proxy given CLIP's joint space, but not the literal text-alignment
   mechanism Nightshade describes.
2. **Validated, and the effect isn't there.** `PHASE4_SCOPING.md` §1's
   recommended methodology (a real GPU LoRA-training run measuring
   generation drift) has now been executed --
   `experiments/concept_misalignment_validation/` on the GPU PC, 30 real
   SD1.5 LoRA trainings (5 images × 3 seeds × 2 conditions), then
   generation + CLIP scoring against both the true and decoy concepts.
   **Result: WEAK/FAIL.** Mean delta_true = -0.0058 (95% CI includes
   zero), mean delta_decoy = -0.0044 (95% CI includes zero) — training on
   the misaligned image did not measurably drift generation away from the
   true concept or toward the decoy one. The CLIP-embedding optimization
   itself does work (concept loss converges from ~0.3 to <0.001 within
   the epsilon budget, confirmed live during this run) — the gap is that
   a single-image LoRA's training dynamics don't propagate that
   pixel-level perturbation into a text/cross-attention shift large
   enough to redirect generation, plausibly because it's closer to
   memorizing the (image, trigger) pair than learning a generalizable
   association a small perturbation could bend.

   **Follow-up to rule that out: also run, also negative.** A second
   experiment (`experiments/concept_misalignment_validation/
   prepare_multiimage.py`) trained one shared LoRA per condition per seed
   across all 5 images jointly instead of 5 isolated single-image LoRAs --
   closer to a real scraper's actual training set. Result (n=15): mean
   delta_true = -0.0020 (95% CI includes zero), mean delta_decoy = +0.0027
   (95% CI includes zero) -- still WEAK/FAIL. **Combining both: the null
   result is not an artifact of the single-image setup** -- a more
   realistic joint training regime shows the same lack of effect. Full
   numbers in `PHASE4_SCOPING.md` §1's "Update" note and
   `experiments/concept_misalignment_validation/out/report.txt` +
   `out_multiimage/report_multiimage.txt` (GPU PC, not committed). Stays
   strictly opt-in with no default-on path, now because the effect wasn't
   measured under two different training regimes, not just because it
   wasn't checked.

## CLIP-transfer chat-AI-editing defense — real-world test, negative result

`style_cloak.py`'s `clip_transfer_weight` (see `PRESETS`' `L1_PREVIEW`/
`L2_PORTFOLIO`/`L3_ANTI_TRAIN` entries) targets a different threat than
everything else in this file: someone pasting a protected image into a
closed commercial multimodal AI (ChatGPT-4o, Gemini, Grok) and asking it
to edit/redraw it, an inference-time request none of the training-defense
mechanisms above were built to resist. It pulls the image's CLIP
embedding toward a decoy target (mirroring `concept_misalign.py`'s own
targeted mechanism, see above) using this project's own small open CLIP
checkpoint (`ViT-B-32/openai`) as a white-box surrogate, on the premise
(also PhotoGuard's) that adversarial perturbations against one open model
sometimes transfer to other, different, unseen models.

**Real-world test (2026-07-22): the transfer did not work, and arguably
backfired.** A real image was cloaked with `L3_ANTI_TRAIN` (`clip_transfer_
weight=1.0`), then uploaded to actual ChatGPT (GPT-4o image edit) and
actual Gemini through their real consumer web UIs, asked to redraw it in
a different style, and compared side by side against the same request
against the unprotected original:

- **ChatGPT**: barely touched the original at all (near-identical
  output), but substantially redrew the *protected* version.
- **Gemini**: redrew both, but added more new elements to the *protected*
  version's redraw than to the original's.

Both platforms produced a *more* different re-render for the protected
image than for the original — the opposite of the intended effect. Not
"the edit is blocked/degraded," but "the edit happens more freely."

**Working theory, not confirmed**: single-surrogate adversarial
perturbations (this project used exactly one open CLIP checkpoint) are
well documented in the adversarial-ML literature to transfer weakly to
different, larger, differently-trained vision encoders — almost
certainly what GPT-4o/Gemini actually run internally, not this project's
small open CLIP model. The perturbation may still be doing *something*
to those models' image understanding (one plausible explanation for
"edited more, not less": whatever recognition/caution behavior kept the
original edit conservative didn't fire on the perturbed version) without
doing the intended thing.

**Bottom line**: `clip_transfer_weight` stays enabled (it's real, cheap,
measured protection-preservation against this project's own metrics —
see the numbers in each preset's comment) but should not be represented
to end users as an effective defense against real GPT-4o/Gemini/Grok
editing. The "unvalidated against actual closed systems" caveat this
mechanism has carried since its first draft turned out to matter in
practice, not just as a disclaimer.

### Ensemble follow-up and its own VRAM ceiling (2026-07-22)

The most plausible fix for weak single-surrogate transfer is the standard
adversarial-ML lever: attack several architecturally/training-data-diverse
CLIP checkpoints at once (`CLIP_ENSEMBLE_CHECKPOINTS` in `style_cloak.py`),
on the premise that a perturbation fooling multiple different models is
less likely to be exploiting just one model's own idiosyncratic decision
boundary. `L3_ANTI_TRAIN` now attacks 2 models (`ViT-B-32/openai` +
`ViT-B-32/laion2b_s34b_b79k`, same architecture, different training data)
at `clip_transfer_weight=0.5` — real GPU sweep: -2.2% styleDriftScore
(0.1608 → 0.1573) for near-saturated similarity to the decoy target on
both models (0.9992, 0.9991), completing in minutes like every other
preset. Whether this ensemble actually improves real-world transfer to
ChatGPT/Gemini is **not yet re-tested** — the negative result above was
found with the single-model version; confirming or refuting the ensemble
needs a repeat of the same manual live-upload test.

A 3rd checkpoint (`ViT-L-14/openai`, real depth/width diversity) was also
tried, twice. The first attempt combined all 3 models' losses into one
scalar before a single `.backward()` call, which required every model's
activation graph to stay resident in VRAM simultaneously — this hung the
project's 8GB GPU PC at ~96% VRAM for 24+ minutes with no progress
(killed). `cloak()`'s training loop was then restructured to call
`.backward()` separately per ensemble member right after that member's own
forward pass (gradients still accumulate correctly into the same `delta`
tensor — that's `.backward()`'s default behavior), bounding activation-
graph memory to whichever single model is largest instead of the sum of
all three. The second attempt with this fix **did stop hanging** — the
sweep completed instead of stalling forever — but real numbers show it's
still not practical: torch's own peak-allocated counter hit 8672MB,
*exceeding* the card's 8151MB physical VRAM, meaning the frozen models'
static weights (unrelated to the activation-graph problem the backward
restructuring fixed) pushed it into Windows/NVIDIA's slow system-RAM-
backed "shared GPU memory" fallback. Consequence: the full 4-config sweep
took 7h24m wall-clock instead of the few minutes every other preset
measurement in this file takes — a different, still-impractical failure
mode (technically completes, unusable for any real upload pipeline) rather
than a real fix. Local quality was comparable to the 2-model result
(weight=0.5: styleDriftScore=0.1562, -2.1%, clipSim=[0.997, 0.9952,
0.9975]), so the 3rd model isn't excluded for a quality reason — its
frozen weights simply don't leave enough headroom on 8GB hardware.

A third attempt tried fp16-casting the frozen CLIP/VGG models' weights
(halving their static footprint — distinct from `torch.autocast`'s
op-level-only mixed precision, which doesn't change static storage dtype)
on top of the sequential-backward fix, on the theory that the remaining
~500MB overage was the weights themselves. Real result: `cloak()` did
complete without crashing this time, but VRAM stayed pegged near the same
~95%+ ceiling and a single config (`weight=0.5`) alone ran 90+ minutes
without finishing before being killed — fp16-casting the weights clearly
wasn't sufficient by itself. Working theory: activation-graph and
optimizer-state memory (which weight-halving doesn't touch), plus the
overhead of keeping 3 architecturally distinct model graphs resident at
once (VGG19 + LPIPS + 3 separate CLIP models, even at fp16), dominates on
this 8GB card more than the frozen weights alone. **2 models is this
hardware's practical ceiling for now** — a real fix for 3+ would need
something bigger (gradient checkpointing on the CLIP forward passes, or
simply a GPU with more VRAM), not attempted further in this session.

### Ensemble re-tested against real ChatGPT/Gemini — still negative, defense deprioritized (2026-07-22)

The 2-model ensemble (deployed to production, see above) was re-tested the
same way as the original single-model version: a real image cloaked with
production `L3_ANTI_TRAIN` settings (fp16-cast weights, watermarked, the
exact artifact a real user's browser would receive) was uploaded to actual
ChatGPT and actual Gemini through their real consumer web UIs, alongside
the unprotected original, both asked to redraw it in a different style.

**Result: still no evidence of a protective effect.** ChatGPT kept the
same overall composition for both but shifted the *protected* version's
sky color/background more than the original's redraw — a real, visible
difference, in the wrong direction (more altered, not less). Gemini
redrew both images into a substantially different illustrated style
regardless of protection status; the original-vs-protected difference in
that case was within the range of normal generation-to-generation
variance for the same prompt, giving no clear signal either way. Ensemble
diversity (the standard adversarial-ML lever for better black-box
transfer) did not produce a measurably different real-world outcome than
the single-model version.

**Decision: stop escalating the adversarial-perturbation approach for
this specific threat.** The underlying limitation isn't a tuning problem
this project can fix with a bigger ensemble or higher weight — it's the
well-documented fact that white-box attacks against small open models
transfer unreliably to large closed ones, and this project has no access
to GPT-4o/Gemini's actual vision encoder to attack directly. Escalating
further (bigger ensembles, higher perturbation weight, more aggressive
noise) was considered and rejected: it runs into the same "analog hole"
every image-protection scheme in this space eventually hits — any pixel
pattern visible enough for a *human* to still recognize the artwork can
be screenshotted, and a screenshot carries none of the original file's
adversarial structure through to the re-upload, so no perturbation
survives that path regardless of how strong it's made. Making the noise
strong enough to survive a screenshot would also make it visible/
damaging to the human viewer it's supposed to look normal to — pushing
harder buys no reliable defense, only guaranteed quality loss.

**Repositioning**: `clip_transfer_weight` stays in the codebase as a
real, cheap, measured (against this project's own metrics), low-cost
mechanism grounded in published research — but is no longer this
project's primary answer to "can I stop someone from feeding my art to
ChatGPT and asking it to redraw it." That threat is better addressed by
this project's mechanisms that don't depend on fooling a specific closed
model's internals: invisible watermarking + `detection-svc`'s evidence
pipeline (real, verified end-to-end against the live production
deployment — recovers the exact embedded payload, builds a signed
evidence bundle) and on-chain registration, which together answer "can
this be proven, after the fact, to be a copy of a specific registered
work" — a claim this project can actually back with real, repeatable
measurements, unlike inference-time editing prevention against an
unknown closed model.

## What this PoC does not do (see PROJECT_DESIGN.md §12)

- Concept-misalignment exists as opt-in code (`concept_misalign.py`,
  above) — a real GPU LoRA-training validation found no measurable
  protection effect, not just "not proven yet." Not on by default
  anywhere.
- `clip_transfer_weight`'s chat-AI-editing defense (see the sections
  above) is enabled by default for all three presets but real tests
  against actual ChatGPT/Gemini, with both a single-model and a 2-model
  ensemble surrogate, found no evidence it blocks editing — one platform's
  redraw diverged *more* from the original when protected, the other
  showed no clear signal either way. Deprioritized as of 2026-07-22: not
  represented to end users as a chat-AI-editing defense; this project's
  actual answer to that threat is the watermark + evidence-pipeline +
  on-chain-registration combination (post-hoc provable copying), not
  pre-emptive editing prevention against a closed model this project
  can't inspect.
- EOT here only covers resize; JPEG recompression isn't part of the training
  loop (real JPEG encoding isn't differentiable — would need a differentiable
  JPEG approximation to include it in EOT, not implemented here).
- EOT has a real, measured ceiling: it recovers robustness for moderate
  resizes (0.5x) but three different training strategies all failed to
  recover 0.25x — see the information-floor conclusion above. Any product
  claim about "surviving thumbnailing" needs to specify how aggressive a
  thumbnail, not just assert robustness in general.
