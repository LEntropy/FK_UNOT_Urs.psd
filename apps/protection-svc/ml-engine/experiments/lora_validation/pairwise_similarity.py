"""Quick utility: prints Gram-matrix cosine similarity from one fixed base
image to every other candidate, so a target-dissimilarity experiment can
pick a good spread (low/med/high similarity) instead of guessing. No
cloak() call, no training -- just forward passes through VGG19, fast even
on CPU.
"""

import argparse
import sys
from pathlib import Path

import torch

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from evaluate import gram_cosine_similarity, mean_sim  # noqa: E402
from model import StyleFeatureExtractor  # noqa: E402
from style_cloak import load_image_tensor  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StyleFeatureExtractor(device)
    base_grams = extractor.gram_matrices(load_image_tensor(args.base, args.size, device))

    results = []
    for c in args.candidates:
        grams = extractor.gram_matrices(load_image_tensor(c, args.size, device))
        sim = mean_sim(gram_cosine_similarity(base_grams, grams))
        results.append((c, sim))

    results.sort(key=lambda r: r[1])
    print(f"=== similarity to {args.base} (sorted low -> high) ===")
    for path, sim in results:
        print(f"{sim:.4f}  {path}")
