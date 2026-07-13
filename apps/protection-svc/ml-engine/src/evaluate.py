"""Quantifies whether cloaking actually did anything, per the two axes in
PROJECT_DESIGN.md §3-3's optimization framing:

    maximize   Feature_Drift        (style embedding measurably moves)
    subject to Perceptual_Distance < epsilon   (pixels barely change)

Without numbers here, "the image looks the same but the AI sees something
different" is just an assertion. This prints both sides so the PoC's claim
is checked, not taken on faith.

Usage:
    python src/evaluate.py --original out/original.png \\
        --cloaked out/cloaked.png --style-target out/style_target.png
"""

import argparse

import torch
import torch.nn.functional as F

from model import StyleFeatureExtractor
from style_cloak import load_image_tensor


def gram_cosine_similarity(
    grams_a: dict[str, torch.Tensor], grams_b: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Per-layer cosine similarity between two images' Gram-matrix style
    representations, flattened to vectors. 1.0 = identical style
    representation, 0.0 = orthogonal/unrelated.
    """
    sims: dict[str, float] = {}
    for layer in grams_a:
        a = grams_a[layer].flatten()
        b = grams_b[layer].flatten()
        sims[layer] = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return sims


def mean_sim(sims: dict[str, float]) -> float:
    return sum(sims.values()) / len(sims)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()  # images in [0,1]


def evaluate(original_path: str, cloaked_path: str, style_target_path: str, size: int = 256) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StyleFeatureExtractor(device)

    original = load_image_tensor(original_path, size, device)
    cloaked = load_image_tensor(cloaked_path, size, device)
    style_target = load_image_tensor(style_target_path, size, device)

    grams_original = extractor.gram_matrices(original)
    grams_cloaked = extractor.gram_matrices(cloaked)
    grams_target = extractor.gram_matrices(style_target)

    sim_orig_to_target = gram_cosine_similarity(grams_original, grams_target)
    sim_cloaked_to_target = gram_cosine_similarity(grams_cloaked, grams_target)
    sim_cloaked_to_original = gram_cosine_similarity(grams_cloaked, grams_original)

    # --- perceptual preservation (pixel space) ---
    diff = (cloaked - original).abs()
    l_inf = diff.max().item()
    rmse = torch.sqrt(F.mse_loss(cloaked, original)).item()
    psnr_db = psnr(cloaked, original)

    print("=== Style drift (Gram-matrix cosine similarity, per VGG19 layer) ===")
    print(f"{'layer':<10} {'orig->target':>12} {'cloaked->target':>16} {'cloaked->orig':>14}")
    for layer in sim_orig_to_target:
        print(
            f"{layer:<10} {sim_orig_to_target[layer]:>12.4f} "
            f"{sim_cloaked_to_target[layer]:>16.4f} {sim_cloaked_to_original[layer]:>14.4f}"
        )

    avg_orig_to_target = mean_sim(sim_orig_to_target)
    avg_cloaked_to_target = mean_sim(sim_cloaked_to_target)
    avg_cloaked_to_original = mean_sim(sim_cloaked_to_original)
    drift = avg_cloaked_to_target - avg_orig_to_target

    print()
    print(f"avg similarity to style_target   before cloak: {avg_orig_to_target:.4f}")
    print(f"avg similarity to style_target   after  cloak: {avg_cloaked_to_target:.4f}")
    print(f"  -> style drift toward target:                {drift:+.4f} "
          f"({'moved toward target' if drift > 0 else 'no drift / moved away'})")
    print(f"avg similarity cloaked vs its own original:     {avg_cloaked_to_original:.4f} "
          f"(1.0 = unchanged style, lower = own style disrupted)")

    print()
    print("=== Perceptual preservation (pixel space, what a human sees) ===")
    print(f"PSNR:              {psnr_db:.2f} dB  (>30 dB is generally considered visually near-identical)")
    print(f"L-infinity diff:   {l_inf:.4f}  ({l_inf * 255:.1f} / 255 max single-pixel-channel change)")
    print(f"RMSE:              {rmse:.4f}  ({rmse * 255:.1f} / 255)")

    print()
    verdict_style = "PASS" if drift > 0.05 else "WEAK/FAIL"
    verdict_perceptual = "PASS" if psnr_db > 30 else "WEAK/FAIL"
    print(f"=== Verdict: style drift {verdict_style}, perceptual preservation {verdict_perceptual} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="out/original.png")
    parser.add_argument("--cloaked", default="out/cloaked.png")
    parser.add_argument("--style-target", default="out/style_target.png")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    evaluate(args.original, args.cloaked, args.style_target, args.size)
