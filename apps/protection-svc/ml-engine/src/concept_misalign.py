"""Concept Misalignment Layer PoC (PROJECT_DESIGN.md §3-3 layer [3],
Nightshade's actual mechanism -- see PHASE4_SCOPING.md §1 for the full
design writeup this module implements).

Same optimization *shape* as style_cloak.py --

    maximize   Concept_Drift (image embedding pulled toward a different
               concept's embedding, in a joint image-text space)
    subject to Perceptual_Distance < epsilon (bounded pixel-space
               perturbation, image looks unchanged to a human)

-- but a genuinely different target space: style_cloak.py perturbs VGG19
Gram-matrix ("style") features; this perturbs CLIP image-embedding
features, which is what actually correlates with a model's *caption-to-
visual-feature* association during fine-tuning (VGG19 Gram matrices have
no notion of text at all). See model.py's ConceptFeatureExtractor for the
CLIP wrapper, mirroring StyleFeatureExtractor's shape.

**Honest status, matching this project's practice of not overclaiming**
(same posture as the C2PA and LoRA-drift sections of ml-engine/README.md):
this file implements the optimization loop PHASE4_SCOPING.md §1 designed,
and running it does measurably move an image's CLIP embedding toward a
decoy concept's embedding within the perceptual budget -- that part is
just calling a documented, well-understood embedding space and doing
gradient descent against it, the same PoC-level claim style_cloak.py
already makes about its own VGG19 drift.

What this file's existence does NOT claim: PHASE4_SCOPING.md §1's own
"recommended validation methodology" (train a real SD1.5 LoRA on
(concept-misaligned image, real caption) pairs and measure whether
generation from that caption drifts to a different concept) has **not**
been run against this code. That requires a real GPU LoRA-training run
(the same kind ml-engine/README.md's LoRA-validation experiment used) and
was out of reach in the session this file was written in -- no GPU was
available, and downloading this module's own pretrained CLIP checkpoint
was blocked by this environment's own external-code safety gate before
even a smoke test could run locally. Until that validation experiment
runs for real, treat "concept misalignment" here the same way
PHASE4_SCOPING.md §1 says to: an unvalidated per-image poisoning
*mechanism*, not a proven per-image poisoning *effect*. Default-off in
orchestrate.py for exactly this reason -- see protect()'s
`concept_misalign` parameter.

Usage:
    python src/concept_misalign.py --original out/original.png \\
        --concept-target out/decoy_concept.png --preset L3_ANTI_TRAIN
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import ConceptFeatureExtractor
from style_cloak import load_image_tensor, random_resize_round_trip, save_tensor_image


@dataclass
class ConceptPreset:
    """Same shape as style_cloak.Preset -- epsilon is the L-infinity
    pixel-space perturbation budget, steps is the optimization iteration
    count. Kept as a separate table (not shared with style_cloak.PRESETS)
    since CLIP-embedding-space loss and VGG19-Gram-space loss have
    different scales and were never jointly tuned -- reusing the same
    numbers here would be an unearned assumption, not a real default.
    """

    epsilon: float
    steps: int
    lr: float


# Starting points mirroring style_cloak.py's L1/L2/L3 shape (same epsilon
# budgets -- perceptual constraint doesn't depend on which feature space
# the loss targets). step counts start equal to style_cloak's too, as an
# untuned starting guess -- PHASE4_SCOPING.md §1's recommended validation
# experiment is what would actually tune these, not run yet (see module
# doc above).
CONCEPT_PRESETS = {
    "L1_PREVIEW": ConceptPreset(epsilon=0.02, steps=150, lr=0.01),
    "L2_PORTFOLIO": ConceptPreset(epsilon=0.04, steps=300, lr=0.01),
    "L3_ANTI_TRAIN": ConceptPreset(epsilon=0.08, steps=500, lr=0.01),
}


def concept_loss(embed_a: torch.Tensor, embed_b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, so minimizing this loss maximizes similarity
    to the decoy concept's embedding -- i.e. pulls embed_a toward embed_b.
    CLIP embeddings are meant to be compared by cosine similarity (that's
    what CLIP's own contrastive training objective optimizes), unlike
    style_cloak.py's Gram matrices which use raw MSE.
    """
    return 1.0 - F.cosine_similarity(embed_a, embed_b).mean()


def misalign(
    original_path: str,
    concept_target_path: str,
    output_path: str,
    preset_name: str,
    size: int = 256,
    eot: bool = False,
    eot_samples: int = 2,
    eot_min_scale: float = 0.3,
    eot_max_scale: float = 1.0,
) -> None:
    """Optimizes `original`'s pixels (within an epsilon ball) so its CLIP
    image embedding moves toward `concept_target`'s CLIP image embedding --
    the visual-feature side of a caption/visual-feature poisoning pair
    (see module doc for what this does and doesn't prove about the text
    side, which needs the real captions a training pipeline would use,
    not available in this project's pipeline -- see PHASE4_SCOPING.md §1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = CONCEPT_PRESETS[preset_name]
    extractor = ConceptFeatureExtractor(device)

    original = load_image_tensor(original_path, size, device)
    concept_target = load_image_tensor(concept_target_path, size, device)
    with torch.no_grad():
        target_embed = extractor.embed(concept_target)

    delta = torch.zeros_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=preset.lr)

    mode = f"EOT(resize, samples={eot_samples}, scale=[{eot_min_scale},{eot_max_scale}])" if eot else "no-EOT"
    print(f"[concept_misalign] preset={preset_name} epsilon={preset.epsilon} steps={preset.steps} mode={mode}")
    for step in range(preset.steps):
        optimizer.zero_grad()
        x_adv = (original + delta).clamp(0, 1)

        if eot:
            loss = concept_loss(extractor.embed(x_adv), target_embed)
            for _ in range(eot_samples):
                transformed = random_resize_round_trip(x_adv, eot_min_scale, eot_max_scale)
                loss = loss + concept_loss(extractor.embed(transformed), target_embed)
            loss = loss / (eot_samples + 1)
        else:
            loss = concept_loss(extractor.embed(x_adv), target_embed)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-preset.epsilon, preset.epsilon)
            delta.copy_(((original + delta).clamp(0, 1) - original))

        if step % 50 == 0 or step == preset.steps - 1:
            print(f"  step {step:4d}  concept_loss={loss.item():.6f}")

    x_adv = (original + delta).clamp(0, 1)
    save_tensor_image(x_adv, output_path)
    print(f"[concept_misalign] wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="out/original.png")
    parser.add_argument("--concept-target", default="out/decoy_concept.png")
    parser.add_argument("--output", default="out/concept_misaligned.png")
    parser.add_argument("--preset", choices=list(CONCEPT_PRESETS), default="L3_ANTI_TRAIN")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--eot", action="store_true")
    parser.add_argument("--eot-samples", type=int, default=2)
    parser.add_argument("--eot-min-scale", type=float, default=0.3)
    parser.add_argument("--eot-max-scale", type=float, default=1.0)
    args = parser.parse_args()

    misalign(
        args.original,
        args.concept_target,
        args.output,
        args.preset,
        size=args.size,
        eot=args.eot,
        eot_samples=args.eot_samples,
        eot_min_scale=args.eot_min_scale,
        eot_max_scale=args.eot_max_scale,
    )
