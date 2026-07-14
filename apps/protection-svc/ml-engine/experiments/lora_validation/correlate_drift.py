"""Follow-up analysis to the LoRA validation experiment (see
ml-engine/README.md's "Final report" section, next-steps item 1): does the
cloak's own VGG19-space optimization target ("Feature_Drift" toward the
cloak-target style, PROJECT_DESIGN.md §3-3) actually predict its
real-world effect on LoRA training (the CLIP-measured delta already
recorded per image at n=30)? If it doesn't correlate, that's direct
evidence the VGG19 proxy metric this project has always reported doesn't
transfer to real training outcomes even in relative/ranking terms, not
just in absolute magnitude -- a stronger claim than "the metrics are
different," which is all prior README sections established.

Runs entirely on CPU with the ml-engine venv (no GPU/training needed) --
reuses cloak() and evaluate.py's gram_cosine_similarity exactly as already
validated elsewhere in this project, just applied to all 10 experiment
images instead of one.
"""

import sys
from pathlib import Path

import torch

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from evaluate import gram_cosine_similarity, mean_sim  # noqa: E402
from model import StyleFeatureExtractor  # noqa: E402
from style_cloak import cloak, load_image_tensor  # noqa: E402
from prepare_dataset import IMAGE_CONFIGS  # noqa: E402

# Per-image mean CLIP delta across 3 seeds, from the n=30 run recorded in
# ml-engine/README.md's stage-6 table (baseline - cloaked CLIP similarity
# to the true painting; positive = cloak reduced LoRA style fidelity).
MEAN_CLIP_DELTA = {
    "starry_night": (0.0062 + 0.0415 + 0.0116) / 3,
    "great_wave": (0.0339 + 0.0209 + 0.0318) / 3,
    "mona_lisa": (0.0061 - 0.0128 - 0.0006) / 3,
    "the_scream": (0.0251 + 0.0011 - 0.0376) / 3,
    "composition_vii": (-0.0008 + 0.0094 + 0.0101) / 3,
    "water_lilies": (-0.0017 + 0.0052 - 0.0120) / 3,
    "girl_pearl_earring": (0.0088 + 0.0259 + 0.0129) / 3,
    "birth_of_venus": (0.0060 + 0.0133 + 0.0140) / 3,
    "night_watch": (0.0115 + 0.0364 + 0.0264) / 3,
    "the_kiss": (0.0340 + 0.0463 + 0.0153) / 3,
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    extractor = StyleFeatureExtractor(device)
    size = 512  # matches the actual LoRA-training resolution used throughout

    results = []
    for cfg in IMAGE_CONFIGS:
        name = cfg["name"]
        print(f"=== [{name}] computing Gram-matrix drift ===")

        original_tensor = load_image_tensor(str(cfg["image"]), size, device)
        target_tensor = load_image_tensor(str(cfg["cloak_target"]), size, device)

        grams_original = extractor.gram_matrices(original_tensor)
        grams_target = extractor.gram_matrices(target_tensor)
        sim_orig_to_target = mean_sim(gram_cosine_similarity(grams_original, grams_target))

        cloaked_path = Path(__file__).parent / "out" / f"_corr_{name}.png"
        cloaked_path.parent.mkdir(parents=True, exist_ok=True)
        cloak(
            original_path=str(cfg["image"]),
            style_target_path=str(cfg["cloak_target"]),
            output_path=str(cloaked_path),
            preset_name="L3_ANTI_TRAIN",
            size=size,
            eot=False,
        )
        cloaked_tensor = load_image_tensor(str(cloaked_path), size, device)
        grams_cloaked = extractor.gram_matrices(cloaked_tensor)
        sim_cloaked_to_target = mean_sim(gram_cosine_similarity(grams_cloaked, grams_target))

        vgg_drift = sim_cloaked_to_target - sim_orig_to_target
        clip_delta = MEAN_CLIP_DELTA[name]
        results.append((name, sim_orig_to_target, vgg_drift, clip_delta))
        cloaked_path.unlink()

    print()
    print("=== drift vs. real-world effect, per image ===")
    print(f"{'image':<20} {'orig->target sim':>17} {'VGG19 drift':>13} {'CLIP delta (n=3)':>18}")
    for name, sim, drift, clip_delta in results:
        print(f"{name:<20} {sim:>17.4f} {drift:>+13.4f} {clip_delta:>+18.4f}")

    # Pearson correlation between VGG19 drift (the cloak's own optimization
    # signal) and the real CLIP-measured LoRA effect -- the actual question.
    drifts = [r[2] for r in results]
    deltas = [r[3] for r in results]
    n = len(drifts)
    mean_drift, mean_delta = sum(drifts) / n, sum(deltas) / n
    cov = sum((d - mean_drift) * (c - mean_delta) for d, c in zip(drifts, deltas))
    var_drift = sum((d - mean_drift) ** 2 for d in drifts)
    var_delta = sum((c - mean_delta) ** 2 for c in deltas)
    r = cov / (var_drift * var_delta) ** 0.5 if var_drift > 0 and var_delta > 0 else float("nan")

    print()
    print(f"Pearson r (VGG19 drift vs. real CLIP-measured LoRA delta): {r:+.3f}")
    print(f"  n={n} images (not seeds -- one drift number per image, matched to its 3-seed-mean CLIP delta)")
    if abs(r) < 0.3:
        verdict = "weak/no correlation -- VGG19 drift does not predict real LoRA impact"
    elif r > 0:
        verdict = "positive correlation -- images the cloak moves further in VGG19 space also show more real LoRA degradation"
    else:
        verdict = "negative correlation -- images the cloak moves further in VGG19 space show LESS real LoRA degradation (counter to the proxy's assumption)"
    print(f"  => {verdict}")

    # Second, more actionable question: does the PRE-cloak similarity
    # between an image and its chosen cloak-target predict the effect,
    # independent of how much the cloak itself moved things? If cloaking
    # toward a more dissimilar style produces a bigger real effect, that's
    # a concrete, applicable recommendation (choose a dissimilar target),
    # not just a diagnostic about the proxy metric.
    sims = [r[1] for r in results]
    mean_sim_val = sum(sims) / n
    cov2 = sum((s - mean_sim_val) * (c - mean_delta) for s, c in zip(sims, deltas))
    var_sim = sum((s - mean_sim_val) ** 2 for s in sims)
    r2 = cov2 / (var_sim * var_delta) ** 0.5 if var_sim > 0 and var_delta > 0 else float("nan")

    print()
    print(f"Pearson r (pre-cloak orig->target similarity vs. real CLIP-measured LoRA delta): {r2:+.3f}")
    if r2 < -0.5:
        print("  => strong negative correlation: cloaking toward a MORE DISSIMILAR style target")
        print("     produces a bigger real LoRA-degradation effect. Actionable: pick style-targets")
        print("     that are maximally different from the original, not a fixed default image.")
    elif abs(r2) < 0.3:
        print("  => weak/no correlation -- target dissimilarity doesn't predict the effect either")


if __name__ == "__main__":
    main()
