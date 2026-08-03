"""Scores the diffusion training-loss attack's real LoRA-validation
experiment -- same CLIP-similarity method and verdict logic as
experiments/lora_validation/generate_and_score.py (imports its
generate_samples/clip_similarity rather than duplicating them, same
convention experiments/concept_misalignment_validation/
generate_and_score_multiimage.py already uses), just scoped to this
experiment's own attacked_v1.safetensors output naming instead of
cloaked_v1.safetensors.

Must run with kohya_ss's venv (diffusers/transformers/peft/accelerate).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lora_validation"))
from generate_and_score import clip_similarity, generate_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lora-root", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out" / "generated"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    lora_root = Path(args.lora_root)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pipe_cache: dict = {}

    runs = []
    for entry in manifest:
        name, trigger, true_image_path = entry["name"], entry["trigger"], entry["true_image"]
        true_image = Image.open(true_image_path).convert("RGB")
        prompt = f"{trigger}, {entry['prompt_suffix']}"

        for seed in seeds:
            print(f"=== [{name} / seed {seed}] baseline ===")
            baseline_lora = lora_root / f"lora_{name}_{seed}_baseline" / f"baseline_{args.run_name}.safetensors"
            baseline_images = generate_samples(
                args.checkpoint, str(baseline_lora), prompt,
                out_dir / name / str(seed) / "baseline", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            baseline_scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in baseline_images]

            print(f"=== [{name} / seed {seed}] attacked ===")
            attacked_lora = lora_root / f"lora_{name}_{seed}_attacked" / f"attacked_{args.run_name}.safetensors"
            attacked_images = generate_samples(
                args.checkpoint, str(attacked_lora), prompt,
                out_dir / name / str(seed) / "attacked", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            attacked_scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in attacked_images]

            avg_baseline = statistics.mean(baseline_scores)
            avg_attacked = statistics.mean(attacked_scores)
            runs.append(
                {
                    "name": name,
                    "seed": seed,
                    "avg_baseline": avg_baseline,
                    "avg_attacked": avg_attacked,
                    "delta": avg_baseline - avg_attacked,
                }
            )

    del pipe_cache["pipe"]
    import torch

    torch.cuda.empty_cache()

    print()
    print("=== per-run results (diffusion training-loss attack) ===")
    print(f"{'image':<16} {'seed':>5} {'baseline':>10} {'attacked':>10} {'delta':>8}")
    for r in runs:
        print(f"{r['name']:<16} {r['seed']:>5} {r['avg_baseline']:>10.4f} {r['avg_attacked']:>10.4f} {r['delta']:>+8.4f}")

    deltas = [r["delta"] for r in runs]
    mean_delta = statistics.mean(deltas)
    stdev_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    n = len(deltas)
    if n > 1:
        t_table = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        t_crit = t_table.get(n - 1, 1.96)
        margin = t_crit * (stdev_delta / (n ** 0.5))
    else:
        margin = float("nan")

    print()
    print(f"n = {n} (image x seed combinations)")
    print(f"mean delta (baseline - attacked): {mean_delta:+.4f}")
    print(f"stdev: {stdev_delta:.4f}")
    print(f"95% CI (t-approx): [{mean_delta - margin:+.4f}, {mean_delta + margin:+.4f}]")
    print(f"individual deltas: {[round(d, 4) for d in deltas]}")
    print()

    threshold = 0.03
    ci_excludes_zero_and_threshold = (mean_delta - margin) > 0
    if ci_excludes_zero_and_threshold and mean_delta > threshold:
        verdict = "PASS (attack measurably degrades LoRA fidelity, 95% CI excludes zero)"
    elif mean_delta > threshold:
        verdict = f"WEAK PASS (mean above threshold but 95% CI includes zero -- not statistically reliable at n={n})"
    else:
        verdict = "WEAK/FAIL (mean at or below threshold)"
    print(f"=== Verdict: {verdict} (mean_delta={mean_delta:+.4f}, threshold={threshold}) ===")
    print()
    print("compare to style_cloak (VGG19 Gram-matrix, n=10): mean delta +0.0113 (95% CI [-0.0038, +0.0264])")
    print("compare to concept_misalign (CLIP embedding, n=15): mean delta_true -0.0058, delta_decoy -0.0044 (both CI include zero)")


if __name__ == "__main__":
    main()
