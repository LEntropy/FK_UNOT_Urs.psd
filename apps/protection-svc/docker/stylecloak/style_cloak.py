"""Glaze-style "style confusion" cloaking PoC.

Implements the optimization described in PROJECT_DESIGN.md §3-3 / §8:

    maximize   Feature_Drift (toward a different style's Gram-matrix
               representation)
    subject to Perceptual_Distance < epsilon (bounded pixel-space
               perturbation, so the image looks unchanged to a human)

This is a simplified, from-scratch reimplementation of the *mechanism*
Glaze/Nightshade-style tools use (adversarial perturbation optimized against
a feature extractor's style representation) — not a copy of their published
implementation, and not tuned to their published fidelity. It's a PoC to
prove the pipeline end-to-end: load image -> optimize -> bounded perturbation
-> style embedding measurably drifts while pixels stay visually unchanged.

Usage:
    python src/style_cloak.py --original out/original.png \\
        --style-target out/style_target.png --preset L3_ANTI_TRAIN
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from model import StyleFeatureExtractor, ConceptFeatureExtractor, DiffusionVAEExtractor

STYLE_LOSS_WEIGHTS = {
    "relu1_1": 1.0,
    "relu2_1": 1.0,
    "relu3_1": 1.0,
    "relu4_1": 1.0,
    "relu5_1": 1.0,
}


@dataclass
class Preset:
    """Maps to the protection strength presets in PROJECT_DESIGN.md §3-4.
    epsilon is the L-infinity pixel-space perturbation budget (in [0,1]
    image scale); steps is the optimization iteration count. color_weight
    is a second loss term's weight (see color_preservation_loss) -- the
    epsilon clamp alone only bounds the *worst single pixel-channel*
    deviation, which doesn't stop the optimizer from pushing the *overall*
    color balance in one direction across the whole image (every pixel's
    red channel nudged up a little, say) -- individually within budget,
    but visible as a real tint shift once summed across the image. Real
    user report on L3 specifically: "색감이 이상해" (colors look wrong,
    the protected image no longer looks close enough to the original).
    """

    epsilon: float
    steps: int
    lr: float
    color_weight: float = 0.0
    tv_weight: float = 0.0
    lpips_weight: float = 0.0
    # compute_perceptual_mask's [low, high] range for this preset -- kept
    # per-preset (not a single global default) because it's only actually
    # been measured for L3_ANTI_TRAIN so far; see PRESETS' L3 entry below
    # for the real GPU numbers.
    mask_low: float = 0.3
    mask_high: float = 1.7
    # Both target the "someone pastes this into ChatGPT/Gemini/Grok and asks
    # it to edit/redraw it" threat -- an inference-time request against a
    # closed commercial model, which style_loss (targets *training*) was
    # never built to resist. See model.py's ConceptFeatureExtractor and
    # DiffusionVAEExtractor doc comments for the mechanism and honesty
    # caveats (best-effort transfer, not validated against any specific
    # closed system).
    clip_transfer_weight: float = 0.0
    vae_transfer_weight: float = 0.0


PRESETS = {
    # L1_PREVIEW measured PSNR 34.5dB, styleDriftScore 0.19 with real GPU
    # numbers (see ml-engine/README.md's "L1/L2/L3 measured" section) --
    # already comfortably above the 30dB "visually near-identical" rule of
    # thumb, left unchanged.
    #
    # clip_transfer_weight=0.1: same targeted chat-AI-editing defense as
    # L2/L3 (see L3_ANTI_TRAIN's comment for the full mechanism rationale),
    # measured separately for L1's own production settings -- no EOT, no
    # perceptual_mask (orchestrate.py: `eot = preset_name != "L1_PREVIEW"`;
    # choose_perceptual_mask only returns True for L2/L3), smallest epsilon
    # (0.02) and fewest steps (150) of the three tiers. weight=0.1 already
    # reached clipSimToDecoyTarget=0.9911 (near the 1.0 max) for only -3.7%
    # styleDriftScore (0.1492 -> 0.1437). weight=0.25 crosses the ~5% bar
    # (-5.4%) for a marginal effect gain (0.9985); weight=0.5 costs even
    # more (-7.5%) with a slightly *worse* effect (0.9953) -- weight=0.1 is
    # the clean pick, completing chat-AI-defense measurement across all
    # three presets (L1=0.1, L2=0.5, L3=1.0).
    "L1_PREVIEW": Preset(epsilon=0.02, steps=150, lr=0.01, color_weight=0.0, clip_transfer_weight=0.1),
    # epsilon and color_weight tuned the same way as L3 below, after a real
    # measurement found L2 borderline (PSNR 28.99dB at the original
    # epsilon=0.04, just under the 30dB rule of thumb). color_weight alone
    # barely moved it (+0.13dB) -- epsilon was the lever again. epsilon=0.03
    # (down from 0.04) crosses to 30.88dB while styleDriftScore stays 0.199
    # (comparable to L1's own 0.19, but L2 still trains 2x the steps, so
    # this isn't just "L2 became L1").
    # clip_transfer_weight=0.5: same targeted chat-AI-editing defense as
    # L3_ANTI_TRAIN (see that entry's comment for the full mechanism
    # rationale -- pulls toward style_target's CLIP embedding instead of
    # pushing away from x_adv's own original), separately measured for L2
    # since it has a smaller epsilon budget (0.03 vs L3's 0.05) and fewer
    # steps (300 vs 500). weight=0.5 reached clipSimToDecoyTarget=0.9969
    # (near the 1.0 max, on par with L3's own saturated result) for only
    # -3.4% styleDriftScore (0.1513 -> 0.1462) -- inside the tolerance bar.
    # weight=1 already crosses it (-5.0%) for slightly *worse* effect
    # (0.9883, EOT-sampling noise) and weight=2 costs more (-6.8%) for a
    # marginal effect gain (0.9986) -- weight=0.5 is the clean pick.
    "L2_PORTFOLIO": Preset(epsilon=0.03, steps=300, lr=0.01, color_weight=8.0, clip_transfer_weight=0.5),
    # epsilon and color_weight both tuned empirically against a real test
    # image on real GPU hardware, with EOT on (matching orchestrate.py's
    # actual production default for this preset) -- see ml-engine/README.md's
    # "L3 color-preservation" section for the full sweep. Measured effect,
    # EOT on: PSNR 23.95dB -> 27.10dB (+3.15dB) while styleDriftScore only
    # dropped 0.249 -> 0.223 (~10%, still far above the 0.05 threshold
    # evaluate.py treats as a real effect). color_weight alone (at the
    # original epsilon=0.08) only bought ~1dB and plateaued past weight=4 --
    # epsilon was the dominant lever for the reported "looks too different"
    # complaint, not color balance specifically; lowering it to 0.05 (down
    # from 0.08, still above L2_PORTFOLIO's 0.03 so L3 stays the strongest
    # preset) did the real work.
    #
    # mask_low=0.15 (down from 0.3): real GPU noise-reduction sweep on top
    # of the native-resolution + perceptual_mask pipeline, comparing three
    # candidate techniques for making the noise even less visible while
    # holding protection steady. Two of the three (TV/total-variation
    # regularization on delta, and an LPIPS perceptual loss term) were
    # tested and REJECTED: both collapsed styleDriftScore by 83-98% (TV
    # weight=8: 0.1617->0.0161; TV weight=24: ->0.0036; LPIPS weight=20:
    # ->0.0276) because minimizing "distance from the original image" has a
    # trivial global optimum at delta=0 -- the optimizer happily converges
    # there, sacrificing the entire adversarial objective. Only tightening
    # the perceptual mask's low bound (0.3 -> 0.15, pushing even less noise
    # into flat regions) was a clean win: styleDriftScore 0.1617 -> 0.1603
    # (-0.9%, noise), PSNR 28.90 -> 29.40dB (+0.5dB), LPIPS distance to
    # original 0.4390 -> 0.3772 (~14% better). Only measured for L3 so far
    # -- L1/L2 keep mask_low=0.3 (the Preset dataclass default) until
    # separately measured.
    # clip_transfer_weight=1.0: targets a different threat than everything
    # above -- someone pasting this image into a closed commercial
    # multimodal AI (ChatGPT-4o/Gemini/Grok) and asking it to edit/redraw
    # it, an inference-time request none of this preset's other terms were
    # built to resist. First attempt (clip_transfer_loss pushing the CLIP
    # embedding away from x_adv's own original, untargeted, no decoy) had
    # no usable low-cost regime: even the smallest weight tested (2.0)
    # already cost -12.6% styleDriftScore (0.1609 -> 0.1406), and cost kept
    # climbing with weight (weight=10: -20.2%) with no sign of a cheap
    # option -- "maximize distance from a moving, ever-more-different
    # point" has no natural stopping point, so it keeps producing a large
    # gradient competing with the style objective for as long as training
    # runs. Redesigned to be *targeted* instead (pull toward style_target's
    # own CLIP embedding, the same decoy image already driving the
    # Gram-matrix loss -- mirrors concept_misalign.py's existing
    # Nightshade-style mechanism exactly) and re-measured: weight=1 reached
    # clipSimToDecoyTarget=0.9965 (near the theoretical max of 1.0) for
    # only -2.9% styleDriftScore (0.1604 -> 0.1557) -- comfortably inside
    # the "same or negligible difference" bar this project holds
    # protection strength to. Higher weights (2/4/8) saturate at
    # essentially the same effect (~0.99-0.999) for steadily worse cost
    # (3.7%/5.5%/7.3%) -- weight=1 already captures nearly all of the
    # available effect, so there's no reason to pay more.
    #
    # vae_transfer_weight stays 0 -- PhotoGuard's own SD-VAE-encoder attack
    # (see model.py's DiffusionVAEExtractor), also redesigned to be
    # targeted the same way, but unlike CLIP the targeted redesign did NOT
    # fix its cost problem: even the smallest weight tested (0.5, at a
    # reduced 512px scale since 1024px still hits this project's 8GB GPU's
    # VRAM ceiling even under AMP -- confirmed hung twice) collapsed
    # styleDriftScore by -87.6% (0.1405 -> 0.0174), saturating at
    # essentially the same damage by weight=2.0 (-92.9%). Raw MSE in VAE
    # latent space apparently doesn't have the same gentle-saturation
    # property cosine similarity in CLIP space does -- a real, different
    # negative result, not just an infrastructure limitation this time.
    #
    # REAL-WORLD TEST, NEGATIVE RESULT (2026-07-22): the "unvalidated
    # against actual closed systems" caveat below stopped being a
    # hypothetical. A real image was cloaked with this exact preset,
    # uploaded to actual ChatGPT (GPT-4o image edit) and actual Gemini
    # through their real consumer web UIs, and asked to redraw it in a
    # different style -- compared side by side against the same request
    # against the unprotected original. Both platforms produced a
    # noticeably MORE different re-render for the *protected* image than
    # for the original (ChatGPT barely touched the original at all but
    # substantially redrew the protected version; Gemini's redraw of the
    # protected version added more new elements than its redraw of the
    # original). This is the opposite of the intended effect -- not "the
    # edit is blocked/degraded" but "the edit happens more freely."
    #
    # Working theory, not confirmed: clip_transfer_weight was tuned as a
    # white-box attack against this project's own small open CLIP
    # checkpoint (ViT-B-32/openai, single model) -- single-surrogate
    # adversarial perturbations are well documented in the literature to
    # transfer weakly to different, larger, differently-trained vision
    # encoders, which is almost certainly what GPT-4o/Gemini actually run
    # internally. The perturbation may still be doing *something* to
    # those models' image understanding (plausible explanation for the
    # "edited more, not less" result: whatever recognition/caution
    # mechanism kept the original edit conservative didn't fire on the
    # perturbed version) without doing the intended thing.
    #
    # Bottom line: do not represent clip_transfer_weight as an effective
    # defense against real GPT-4o/Gemini/Grok editing to end users. It
    # remains enabled here as a measured, low-cost (see numbers above)
    # experimental mechanism grounded in published research, not a proven
    # one -- the honesty caveat this project has carried since this
    # mechanism's first draft turned out to matter in practice, not just
    # as a disclaimer.
    #
    # ENSEMBLE ADOPTION (2026-07-22): switched from the single-model
    # measurement above to CLIP_ENSEMBLE_CHECKPOINTS' 2-model ensemble
    # (ViT-B-32/openai + ViT-B-32/laion2b_s34b_b79k), the standard
    # adversarial-ML lever for better black-box transfer -- a perturbation
    # fooling two differently-trained models at once is less likely to be
    # exploiting one model's own idiosyncratic decision boundary, which is
    # the most plausible fix for the real-world negative result above.
    # Real GPU sweep at 2 models: weight=0.5 reached
    # clipSimToDecoyTarget=[0.9992, 0.9991] (both near-saturated) for only
    # -2.2% styleDriftScore (0.1608 -> 0.1573); weight=1/2 cost more
    # (-3.4%/-4.5%) for no meaningful similarity gain -- weight=0.5 is the
    # clean pick, same saturating-cost pattern as every other targeted-decoy
    # measurement in this file. peakVramMB=5751, well inside this project's
    # 8GB GPU PC's budget, completes in minutes like every other preset.
    #
    # A 3rd checkpoint (ViT-L-14/openai, real depth/width diversity, not
    # just training-data diversity) was also tried, using cloak()'s
    # sequential-per-model-backward training loop (see that function's
    # comment) specifically built to keep this from recreating the VRAM
    # hang the first 3-model attempt hit. It did stop hanging -- the sweep
    # completed instead of stalling forever -- but real numbers show why
    # it's not adopted: torch's own peak-allocated counter hit 8672MB,
    # *exceeding* this GPU's 8151MB physical VRAM, meaning it was spilling
    # into Windows/NVIDIA's slow system-RAM-backed "shared GPU memory"
    # fallback rather than genuinely fitting. Consequence: the full 4-config
    # sweep (500 steps each) took 7h24m wall-clock, vs. minutes for the
    # 2-model version -- the hang became a different, still-impractical
    # failure mode (technically completes, too slow for any real upload
    # pipeline) rather than a real fix. Local quality numbers were
    # comparable to the 2-model result (weight=0.5: styleDriftScore=0.1562,
    # -2.1%, clipSim=[0.997, 0.9952, 0.9975]), so the 3rd model isn't being
    # left out for a quality reason -- it's purely that its frozen weights'
    # static VRAM footprint doesn't leave enough headroom on 8GB hardware.
    # fp16-casting the frozen CLIP/VGG models (halving their static
    # footprint, distinct from autocast's op-level-only mixed precision)
    # was tried next, on top of the sequential-backward fix -- also not
    # enough. cloak() itself completed without crashing, but VRAM stayed
    # pegged near the same ~95%+ ceiling and one config alone ran 90+ min
    # without finishing before being killed (see CLIP_ENSEMBLE_CHECKPOINTS'
    # doc for the full writeup). The frozen weights weren't the dominant
    # cost after all -- activation-graph/optimizer-state memory and the
    # sheer number of distinct model graphs resident at once (VGG19 +
    # LPIPS + 3 CLIP models) apparently matter more on this 8GB card. 2
    # models is this hardware's practical ceiling for now; a real fix for
    # 3+ would need something bigger (gradient checkpointing, or a GPU with
    # more VRAM), not attempted further.
    "L3_ANTI_TRAIN": Preset(epsilon=0.05, steps=500, lr=0.01, color_weight=8.0, mask_low=0.15, clip_transfer_weight=0.5),
}


def letterbox_content_box(orig_w: int, orig_h: int, size: int) -> tuple[int, int, int, int]:
    """Where the real (non-padding) content sits inside a size x size
    letterboxed canvas for an orig_w x orig_h source -- shared between
    letterbox_resize (building the canvas) and cloak() (cropping the
    padding back out of the final output). Kept as its own function so
    both sides compute the identical box from the same two numbers,
    instead of risking the resize math drifting apart from the crop math.
    """
    scale = size / max(orig_w, orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    return (left, top, left + new_w, top + new_h)


def letterbox_resize(img: Image.Image, size: int) -> Image.Image:
    """Resizes preserving aspect ratio to fit within size x size, then pads
    with neutral gray to reach exactly size x size.

    Replaces a plain img.resize((size, size)), which silently stretched
    every non-square upload into a square -- a real, user-visible bug: a
    portrait or landscape photo came out with the wrong proportions in the
    published, watermarked result, not just a resolution problem. VGG's
    Gram-matrix style loss doesn't care about the padding (both the
    original and the adversarial image carry the same gray border in the
    same place through the whole optimization, so it doesn't bias the
    *difference* being optimized) -- the padding only needs to be cropped
    back out once, after cloak() finishes writing pixels, which is what
    letterbox_content_box is for.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top, _, _ = letterbox_content_box(w, h, size)
    canvas.paste(resized, (left, top))
    return canvas


def image_to_tensor(img: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    """Converts a PIL image (any mode/size) to the 1x3xHxW [0,1] tensor the
    model expects. Split out from load_image_tensor so callers that already
    have a PIL image in memory (e.g. robustness_test.py, after simulating a
    JPEG re-encode) don't have to round-trip through a file.
    """
    img = letterbox_resize(img.convert("RGB"), size)
    arr = torch.from_numpy(
        __import__("numpy").array(img).astype("float32") / 255.0
    )  # HWC in [0,1]
    return arr.permute(2, 0, 1).unsqueeze(0).to(device)  # 1x3xHxW


def load_image_tensor(path: str, size: int, device: torch.device) -> torch.Tensor:
    return image_to_tensor(Image.open(path), size, device)


def save_tensor_image(x: torch.Tensor, path: str) -> None:
    import numpy as np

    arr = x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


def style_loss(grams_a: dict[str, torch.Tensor], grams_b: dict[str, torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=next(iter(grams_a.values())).device)
    for layer, weight in STYLE_LOSS_WEIGHTS.items():
        total = total + weight * F.mse_loss(grams_a[layer], grams_b[layer])
    return total


def color_preservation_loss(original: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """MSE between heavily-blurred (16x16 average-pooled) versions of the
    original and adversarial image -- penalizes a large-scale, low-frequency
    shift in overall color/tone (a visible "tint" across the whole image)
    without penalizing the high-frequency pixel noise the epsilon-bounded
    perturbation actually needs room to move in. The epsilon clamp alone
    only bounds the worst single pixel-channel deviation; it says nothing
    about every pixel's red channel drifting the same direction at once,
    which sums to a real, visible color cast even though each individual
    pixel stayed inside budget.
    """
    pooled_original = F.avg_pool2d(original, kernel_size=16, stride=16)
    pooled_adv = F.avg_pool2d(x_adv, kernel_size=16, stride=16)
    return F.mse_loss(pooled_adv, pooled_original)


def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between horizontally/vertically adjacent
    pixels of the perturbation itself (not the image) -- penalizes a noisy,
    speckled delta in favor of a smoother one that still fits the same
    epsilon budget. This is what makes adversarial noise look like grain
    or a texture instead of static: two neighboring pixels pushed in wildly
    different directions is what reads as "visible noise" to a human eye,
    even when both are individually within the epsilon clamp. Classic
    adversarial-perturbation smoothing technique (same idea as TV
    regularization in image denoising, applied to the perturbation instead
    of the image).
    """
    dh = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    dw = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return dh + dw


_lpips_model = None


def lpips_loss(original: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """Learned Perceptual Image Patch Similarity -- a network trained to
    match human perceptual judgments of "do these two images look the
    same", used here as a direct optimization target for imperceptibility
    (unlike color_preservation_loss's blur-MSE heuristic, this is trained
    specifically to track human visual similarity, including texture and
    structure, not just low-frequency color). Complementary to, not a
    replacement for, color_preservation_loss and the perceptual mask --
    those shape *where* and *how* noise gets clamped; this one gives the
    optimizer a *direct* incentive to minimize the noise's visible impact
    in the first place, alongside chasing style drift.

    lpips expects [-1,1]-scaled inputs (not this codebase's [0,1] tensor
    convention) -- see lpips's own README.
    """
    global _lpips_model
    if _lpips_model is None:
        import lpips as _lpips_pkg

        _lpips_model = _lpips_pkg.LPIPS(net="vgg").to(original.device)
        for p in _lpips_model.parameters():
            p.requires_grad_(False)
    return _lpips_model(original * 2 - 1, x_adv * 2 - 1).mean()


_clip_ensemble: list | None = None
_vae_extractor = None

# Real-world negative result (see ml-engine/README.md's "CLIP-transfer
# chat-AI-editing defense" section): a single-checkpoint white-box attack
# (ViT-B-32/openai alone) did not transfer to actual ChatGPT/Gemini in a
# live test -- both re-rendered the *protected* image MORE differently
# than the original, the opposite of the intended effect. Attacking
# several architecturally/training-data-diverse checkpoints at once (the
# standard adversarial-ML lever for improving black-box transfer -- a
# perturbation fooling multiple different models simultaneously is less
# likely to be exploiting one model's own idiosyncratic decision boundary)
# is the follow-up this constant enables. Picked for diversity, not just
# more of the same: ViT-B-32/openai (the original, already-measured
# checkpoint), ViT-B-32/laion2b_s34b_b79k (same architecture, completely
# different training data/procedure -- LAION vs OpenAI's own proprietary
# dataset).
#
# A third checkpoint (ViT-L-14/openai, same training data as the first but
# deeper/wider -- real depth/width diversity, not just training-data
# diversity) was tried twice and dropped both times, for two different
# reasons. First attempt: with all 3 models' losses combined into one
# scalar and backprop'd in a single .backward() call, every ensemble
# member's activation graph had to stay alive at once, which pushed the
# project's 8GB GPU PC to ~96% VRAM and hung for 24+ min with no progress
# (same allocator-thrashing pattern this project has hit before with
# VAE-transfer and size=1536 -- see choose_use_amp's doc), killed. Second
# attempt, after cloak()'s training loop was restructured to call
# .backward() separately per ensemble member right after that member's own
# forward pass (bounding activation-graph memory to whichever single model
# is largest, not the sum of all three): the hang was gone, but the 3
# models' frozen *weights* (always resident, unrelated to the activation
# graph the backward restructuring fixed) still didn't leave enough
# headroom -- torch's own peak-allocated counter hit 8672MB, exceeding this
# card's 8151MB physical VRAM, meaning it was spilling into Windows/
# NVIDIA's slow system-RAM-backed "shared GPU memory" fallback rather than
# genuinely fitting. Real cost: a 4-config sweep that normally takes
# minutes took 7h24m wall-clock (see PRESETS' L3_ANTI_TRAIN comment for the
# full writeup) -- technically completes, not practical for any real
# upload pipeline.
#
# Third attempt: fp16-cast the frozen extractor weights themselves (see
# cloak()'s `extractor_dtype` cast, distinct from autocast's op-level-only
# mixed precision) on top of the sequential-backward fix, on the theory
# that halving the static weight footprint would close the remaining
# ~500MB overage. Real result: cloak() itself did complete this time (no
# crash), but VRAM stayed pegged near the same ~95%+ ceiling and one
# config (weight=0.5) alone ran 90+ min without finishing before being
# killed -- fp16-casting the weights clearly wasn't enough by itself.
# Working theory: activation-graph and optimizer-state memory (which the
# weight-halving doesn't touch) plus the surviving overhead of loading 3
# separate model architectures simultaneously (even at fp16, VGG19 + LPIPS
# + 3 CLIP models is still a lot of *distinct* graphs to keep resident)
# dominates on an 8GB card, not the weights alone. A real fix would need
# something bigger -- gradient checkpointing on the CLIP forward passes,
# or simply accepting that 2 models is this hardware's ceiling until a
# GPU with more VRAM is available -- not attempted further for now. 2
# models (same architecture, different training data -- still real
# diversity, just not also varying depth/width) is what actually fits
# comfortably in real VRAM on this hardware and is what's shipped.
CLIP_ENSEMBLE_CHECKPOINTS: list[tuple[str, str]] = [
    ("ViT-B-32", "openai"),
    ("ViT-B-32", "laion2b_s34b_b79k"),
]


def get_clip_ensemble(device: torch.device) -> list:
    global _clip_ensemble
    if _clip_ensemble is None:
        _clip_ensemble = [ConceptFeatureExtractor(device, model_name, pretrained) for model_name, pretrained in CLIP_ENSEMBLE_CHECKPOINTS]
    return _clip_ensemble


def clip_transfer_loss(clip_ensemble: list, target_embeds: list[torch.Tensor], x_adv: torch.Tensor) -> torch.Tensor:
    """Mean, across every model in the ensemble, of (1 - cosine similarity
    to that model's own embedding of the fixed decoy) -- minimizing this
    pulls x_adv's embedding *toward* a specific, different image's
    embedding (the same style_target already used for the Gram-matrix
    objective) simultaneously in every ensemble member's own embedding
    space, mirroring concept_misalign.py's concept_loss exactly instead of
    inventing a new formulation, just summed over several surrogate models
    instead of one.

    Superseded the original untargeted single-model version (maximize
    distance from x_adv's own original embedding, no decoy) after a real
    GPU sweep found it had no usable low-cost regime -- see PRESETS'
    L3_ANTI_TRAIN comment for that history. The targeted single-model
    version that replaced it was then real-world-tested against actual
    ChatGPT/Gemini and found not to transfer (ml-engine/README.md) --
    this ensemble version is the follow-up attempt at better black-box
    transfer, itself not yet re-validated against real closed systems
    (see CLIP_ENSEMBLE_CHECKPOINTS' doc).

    target_embeds is a list precomputed once per cloak() call (one
    embedding per ensemble member, from style_target, with no_grad) by
    the caller, not recomputed here every step -- same list order and
    length as clip_ensemble.
    """
    losses = [
        1.0 - F.cosine_similarity(clip_ext.embed(x_adv), target_embed).mean()
        for clip_ext, target_embed in zip(clip_ensemble, target_embeds)
    ]
    return torch.stack(losses).mean()


def vae_transfer_loss(vae_extractor, target_latent: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """MSE to a fixed decoy Stable Diffusion VAE latent -- minimizing this
    pulls x_adv's encoded latent *toward* style_target's latent, the same
    targeted-vs-untargeted reasoning as clip_transfer_loss above (this
    project's own real GPU testing never got past confirming the untargeted
    version was too VRAM-heavy to even measure on an 8GB card, with or
    without AMP -- this targeted version wasn't separately re-measured
    against that same hardware wall, since the constraint is the VAE
    encoder's activation memory at real processing resolutions, not the
    loss target choice; PhotoGuard's own published attack is untargeted,
    included here for symmetry with clip_transfer_loss and in case a
    future GPU upgrade makes measuring it practical).

    target_latent is precomputed once per cloak() call (from style_target,
    with no_grad) by the caller, not recomputed here every step.
    """
    return F.mse_loss(vae_extractor.encode_latent(x_adv), target_latent)


def compute_perceptual_mask(original: torch.Tensor, low: float = 0.3, high: float = 1.7) -> torch.Tensor:
    """Per-pixel multiplier on the epsilon clamp, in [low, high] -- the
    real fix real steganography/Glaze-style tools use for "noise looks
    too visible": a human eye is far more sensitive to noise in smooth,
    flat regions (sky, skin, a plain background) than in already-textured,
    high-detail regions (brushwork, foliage, hair). A *uniform* epsilon
    clamp (what this file did before) spends the same noise budget
    everywhere, which is the worst place to spend it evenly -- it's
    exactly as visible as the flattest region in the image can tolerate.

    Built from a Sobel gradient-magnitude map (local edge/texture
    strength), normalized to [0, 1] per image, then rescaled to [low,
    high] so the *average* multiplier stays close to 1.0 -- redistributing
    where the epsilon budget goes rather than changing the total budget,
    which is what keeps the protection effect (style drift) close to
    unchanged while the perturbation becomes far less visible in the
    regions a human actually looks at first.
    """
    device = original.device
    gray = original.mean(dim=1, keepdim=True)  # 1x1xHxW, luminance proxy

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    # replicate padding, not conv2d's default zero-padding -- padding with
    # 0 creates a fake high-contrast "edge" between real content and the
    # artificial black border on every side, which the mask would then
    # (wrongly) read as "this border region is highly textured."
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(gray_padded, sobel_x)
    gy = F.conv2d(gray_padded, sobel_y)
    edge_strength = torch.sqrt(gx**2 + gy**2)  # exactly 0 for genuinely flat regions, no floor offset

    # Widen each edge's influence -- a pixel a few steps away from a strong
    # edge is still in a "textured neighborhood" a human won't scrutinize
    # as closely as a truly flat region, and a single-pixel-wide mask would
    # leave razor-thin safe strips the optimizer can't meaningfully use.
    edge_strength = F.max_pool2d(F.pad(edge_strength, (4, 4, 4, 4), mode="replicate"), kernel_size=9, stride=1)

    spread = edge_strength.max() - edge_strength.min()
    if spread < 1e-4:
        # Degenerate case: a genuinely (near-)flat image has no texture
        # signal to redistribute toward -- normalizing near-equal values
        # by a near-zero range is numerically unstable (floating-point
        # noise around the floor can dominate the ratio and swing the
        # result to either end). Fall back to `low` uniformly, the
        # conservative choice for "nothing here is safer to perturb than
        # anything else."
        return torch.full_like(gray, low).expand(-1, 3, -1, -1)

    normalized = (edge_strength - edge_strength.min()) / spread
    mask = low + normalized * (high - low)
    return mask.expand(-1, 3, -1, -1)  # broadcast the 1-channel mask across RGB


def random_resize_round_trip(
    x: torch.Tensor,
    min_scale: float = 0.3,
    max_scale: float = 1.0,
    scales: list[float] | None = None,
) -> torch.Tensor:
    """Differentiable stand-in for "someone's upload pipeline downscaled this
    image" — downsamples to a random scale then back to the original size,
    using torch's own (differentiable) interpolation so gradients can flow
    back through it into `delta`. This is what EOT training actually needs:
    F.interpolate, not PIL, because PIL round-trips aren't part of the
    autograd graph.

    If `scales` is given, sample uniformly from that discrete set instead of
    a continuous [min_scale, max_scale) range — widening the continuous
    range dilutes how often training actually hits a specific troublesome
    scale (e.g. 0.25x), since most draws land elsewhere in the range. A
    discrete set guarantees every listed scale gets trained against roughly
    equally often.
    """
    _, _, h, w = x.shape
    if scales:
        scale = scales[torch.randint(0, len(scales), (1,)).item()]
    else:
        scale = torch.empty(1).uniform_(min_scale, max_scale).item()
    small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
    down = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def cloak(
    original_path: str,
    style_target_path: str,
    output_path: str,
    preset_name: str,
    size: int = 256,
    eot: bool = False,
    eot_samples: int = 2,
    eot_min_scale: float = 0.3,
    eot_max_scale: float = 1.0,
    eot_scales: list[float] | None = None,
    perceptual_mask: bool = False,
    mask_low: float | None = None,
    mask_high: float | None = None,
    use_amp: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = PRESETS[preset_name]
    # None (the default) -> use this preset's own measured mask range;
    # an explicit value (CLI/experiment override) takes precedence.
    mask_low = preset.mask_low if mask_low is None else mask_low
    mask_high = preset.mask_high if mask_high is None else mask_high
    extractor = StyleFeatureExtractor(device)

    # Opt-in mixed precision (fp16 forward/backward through VGG, fp32 the
    # optimized `delta` itself) -- roughly halves the VGG activation memory
    # that dominates VRAM usage at real processing resolutions. Investigated
    # because the GPU PC's card only has 8GB VRAM total (confirmed via
    # nvidia-smi) and MAX_PROCESSING_SIZE is already capped at 1024 in
    # orchestrate.py specifically because eot_samples=2 at that size pushed
    # VRAM to ~96% and hung for 2+ hours -- this is the lever to raise that
    # cap without hitting the same wall again, not a general speed
    # optimization. GradScaler guards against the gradient-underflow risk
    # fp16 introduces (this project's loss values run small, ~0.004-0.02).
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # Lazily instantiated (module-level cache, mirrors _lpips_model) --
    # only load the CLIP ensemble/the SD VAE when a caller actually opts
    # into the chat-AI-editing defense, not on every cloak() call.
    global _vae_extractor
    clip_ensemble = None
    if preset.clip_transfer_weight > 0:
        clip_ensemble = get_clip_ensemble(device)
    vae_extractor = None
    if preset.vae_transfer_weight > 0:
        if _vae_extractor is None:
            _vae_extractor = DiffusionVAEExtractor(device)
        vae_extractor = _vae_extractor

    # fp16-cast these frozen models' *weights* (not just autocast's
    # op-level-only mixed precision, which leaves static storage dtype
    # unchanged) when AMP is on for this call -- halves their resident
    # VRAM footprint, the real fix CLIP_ENSEMBLE_CHECKPOINTS' doc promised
    # after the 2-vs-3-model ensemble measurement found the 3rd model's
    # frozen weights (not activation-graph memory, already fixed by the
    # sequential-backward restructuring below) were what pushed peak VRAM
    # past this project's 8GB GPU's physical capacity. Re-cast on every
    # call rather than once at load time because these models are cached
    # module-level singletons (get_clip_ensemble, _vae_extractor) reused
    # across calls that may have different use_amp settings (e.g.
    # orchestrate.py's choose_use_amp turns AMP off for small images) --
    # a stale fp16 cast from an earlier AMP call would crash a later
    # non-AMP call's un-autocasted forward passes (dtype mismatch, no
    # autocast context to reconcile fp32 activations against fp16
    # weights). The cast itself is just a dtype conversion of
    # already-loaded weights (no recomputation), effectively free compared
    # to the steps it runs before.
    extractor_dtype = torch.float16 if amp_enabled else torch.float32
    if clip_ensemble is not None:
        for clip_ext in clip_ensemble:
            clip_ext.to(extractor_dtype)
    if vae_extractor is not None:
        vae_extractor.to(extractor_dtype)

    original = load_image_tensor(original_path, size, device)
    style_target = load_image_tensor(style_target_path, size, device)
    target_grams = extractor.gram_matrices(style_target)

    # Precomputed once (not recomputed every step) -- style_target doubles
    # as the decoy for clip_transfer_loss/vae_transfer_loss, the same image
    # already driving the Gram-matrix style objective, so a single target
    # image coherently drives all three feature spaces this preset opts
    # into instead of needing a separate decoy parameter. Wrapped in the
    # same autocast context the training loop uses -- required now that
    # the extractors above may be fp16 while style_target itself stays
    # fp32 (autocast is what reconciles that, same as every per-step
    # forward pass already does).
    clip_target_embeds = None
    if clip_ensemble is not None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            clip_target_embeds = [clip_ext.embed(style_target) for clip_ext in clip_ensemble]
    vae_target_latent = None
    if vae_extractor is not None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            vae_target_latent = vae_extractor.encode_latent(style_target)

    # Opt-in (default off -- see this parameter's callers/README before
    # flipping the default): redistributes the same epsilon budget toward
    # already-textured regions and away from flat ones instead of a
    # uniform clamp everywhere -- see compute_perceptual_mask's doc.
    epsilon_mask = compute_perceptual_mask(original, mask_low, mask_high) * preset.epsilon if perceptual_mask else preset.epsilon

    delta = torch.zeros_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=preset.lr)

    scale_desc = f"discrete{eot_scales}" if eot_scales else f"[{eot_min_scale},{eot_max_scale}]"
    mode = f"EOT(resize, samples={eot_samples}, scale={scale_desc})" if eot else "no-EOT (clean image only)"
    mask_desc = " perceptual_mask=on" if perceptual_mask else ""
    amp_desc = " amp=on" if amp_enabled else ""
    print(f"[cloak] preset={preset_name} epsilon={preset.epsilon} steps={preset.steps} mode={mode}{mask_desc}{amp_desc}")
    for step in range(preset.steps):
        optimizer.zero_grad()

        # Core objective (VGG style loss + EOT + color/TV/LPIPS regularizers)
        # stays one combined forward+backward, same as always -- this group
        # has never been the VRAM problem (measured fine at 1024px+AMP long
        # before any of the chat-AI-defense terms existed).
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            x_adv = (original + delta).clamp(0, 1)

            if eot:
                # Expectation over transformation: average the loss over
                # several random resize round-trips (plus the clean image)
                # instead of optimizing the clean image alone, so the
                # perturbation survives in expectation across the transform
                # distribution, not just at exact original resolution.
                loss = style_loss(extractor.gram_matrices(x_adv), target_grams)
                for _ in range(eot_samples):
                    transformed = random_resize_round_trip(x_adv, eot_min_scale, eot_max_scale, eot_scales)
                    loss = loss + style_loss(extractor.gram_matrices(transformed), target_grams)
                loss = loss / (eot_samples + 1)
            else:
                loss = style_loss(extractor.gram_matrices(x_adv), target_grams)

            if preset.color_weight > 0:
                color_loss = color_preservation_loss(original, x_adv)
                loss = loss + preset.color_weight * color_loss

            if preset.tv_weight > 0:
                loss = loss + preset.tv_weight * total_variation_loss(delta)

            if preset.lpips_weight > 0:
                loss = loss + preset.lpips_weight * lpips_loss(original, x_adv)

        scaler.scale(loss).backward()

        # Chat-AI-defense terms: each surrogate model gets its own fresh
        # x_adv, forward pass, and backward() call instead of joining one
        # combined graph -- real GPU measurement found the combined-graph
        # version needed every ensemble member's activations alive
        # simultaneously (VRAM scales with model count), which hung a
        # 3-model CLIP ensemble at ~96% VRAM for 24+ min on this project's
        # 8GB GPU PC. backward() accumulates into delta.grad by default (it
        # doesn't overwrite), and PyTorch frees a call's own activation
        # graph once that call returns -- so peak memory for this whole
        # block is bounded by whichever single surrogate is largest, not
        # the sum of all of them, regardless of how many are in the
        # ensemble. Weighted so the accumulated gradient exactly matches
        # what the old combined-mean-then-one-backward version would have
        # produced (weight/N per model, summed over N models, equals
        # weight * mean(N models)).
        if clip_ensemble is not None:
            per_model_weight = preset.clip_transfer_weight / len(clip_ensemble)
            for clip_ext, target_embed in zip(clip_ensemble, clip_target_embeds):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    x_adv_clip = (original + delta).clamp(0, 1)
                    clip_loss = per_model_weight * clip_transfer_loss([clip_ext], [target_embed], x_adv_clip)
                scaler.scale(clip_loss).backward()

        if vae_extractor is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                x_adv_vae = (original + delta).clamp(0, 1)
                vae_loss = preset.vae_transfer_weight * vae_transfer_loss(vae_extractor, vae_target_latent, x_adv_vae)
            scaler.scale(vae_loss).backward()

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            delta.clamp_(-epsilon_mask, epsilon_mask)
            delta.copy_(((original + delta).clamp(0, 1) - original))  # keep x_adv in [0,1]

        if step % 50 == 0 or step == preset.steps - 1:
            print(f"  step {step:4d}  style_loss={loss.item():.6f}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[cloak] wrote {output_path}")

    # Restore these cached singletons to fp32 before returning -- real bug,
    # found live: leaving them in the fp16 state this call may have cast
    # them to (see the cast above) meant any *other* code touching the same
    # cached clip_ensemble/vae_extractor after this call returns (e.g. an
    # evaluation script calling clip_ext.embed() directly, with no
    # autocast context to reconcile fp32 activations against fp16 weights)
    # crashed with a dtype-mismatch RuntimeError. fp32 is the safe default
    # any caller should be able to assume; the next cloak() call re-casts
    # to fp16 itself if it needs to (cheap, see that cast's own comment).
    if clip_ensemble is not None:
        for clip_ext in clip_ensemble:
            clip_ext.to(torch.float32)
    if vae_extractor is not None:
        vae_extractor.to(torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="out/original.png")
    parser.add_argument("--style-target", default="out/style_target.png")
    parser.add_argument("--output", default="out/cloaked.png")
    parser.add_argument("--preset", choices=list(PRESETS), default="L3_ANTI_TRAIN")
    parser.add_argument(
        "--size", type=int, default=256, help="square processing resolution (all presets were tuned at 256)"
    )
    parser.add_argument("--eot", action="store_true", help="optimize against random resize round-trips too")
    parser.add_argument("--eot-samples", type=int, default=2)
    parser.add_argument("--eot-min-scale", type=float, default=0.3)
    parser.add_argument("--eot-max-scale", type=float, default=1.0)
    parser.add_argument(
        "--eot-scales",
        type=str,
        default=None,
        help="comma-separated discrete scales, e.g. '0.25,0.5,1.0' (overrides --eot-min/max-scale)",
    )
    parser.add_argument(
        "--perceptual-mask",
        action="store_true",
        help="redistribute the epsilon clamp toward already-textured regions (JND-style) instead of "
        "a uniform clamp -- see compute_perceptual_mask's doc. GPU-measured real win: +1.37dB PSNR "
        "for -1.9% styleDriftScore on a real high-res L3 upload.",
    )
    parser.add_argument("--mask-low", type=float, default=None, help="defaults to the chosen preset's own measured value")
    parser.add_argument("--mask-high", type=float, default=None, help="defaults to the chosen preset's own measured value")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="mixed precision (fp16) forward/backward through VGG -- roughly halves VRAM usage, "
        "the lever for raising MAX_PROCESSING_SIZE past 1024 on an 8GB GPU without hitting the "
        "eot_samples=2-at-1024 VRAM wall documented in orchestrate.py's choose_eot_samples.",
    )
    args = parser.parse_args()

    eot_scales = [float(s) for s in args.eot_scales.split(",")] if args.eot_scales else None

    cloak(
        args.original,
        args.style_target,
        args.output,
        args.preset,
        size=args.size,
        eot=args.eot,
        eot_samples=args.eot_samples,
        eot_min_scale=args.eot_min_scale,
        eot_max_scale=args.eot_max_scale,
        eot_scales=eot_scales,
        perceptual_mask=args.perceptual_mask,
        mask_low=args.mask_low,
        mask_high=args.mask_high,
        use_amp=args.amp,
    )
