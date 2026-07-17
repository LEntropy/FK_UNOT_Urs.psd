"""Scores the multi-image follow-up (prepare_multiimage.py) -- same CLIP
similarity-to-true-and-decoy measurement as generate_and_score.py, but
each seed has only ONE baseline LoRA and ONE misaligned LoRA (trained
jointly on all 5 images), reused across all 5 images' prompts, instead of
5 separate per-image LoRAs.

Expects trained LoRA weights at
{lora-root}/lora_multi_{seed}_{condition}/{condition}_v{run}.safetensors
for every seed in --seeds, matching
run_concept_misalignment_multiimage_validation.ps1's naming convention.

Must run with kohya_ss's venv python (diffusers/transformers/peft/torch+
cuda) -- NOT ml-engine's own .venv.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_and_score import ci_margin, clip_similarity, generate_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SD1.5 base checkpoint path")
    parser.add_argument("--manifest", required=True, help="manifest_multiimage.json written by prepare_multiimage.py")
    parser.add_argument("--lora-root", required=True, help="dir containing lora_multi_{seed}_{condition}/ subfolders")
    parser.add_argument("--seeds", required=True, help="comma-separated training seeds, e.g. '1,2,3'")
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out_multiimage" / "generated"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    images = manifest["images"]
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    lora_root = Path(args.lora_root)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pipe_cache: dict = {}

    runs = []
    for seed in seeds:
        baseline_lora = lora_root / f"lora_multi_{seed}_baseline" / f"baseline_{args.run_name}.safetensors"
        misaligned_lora = lora_root / f"lora_multi_{seed}_misaligned" / f"misaligned_{args.run_name}.safetensors"

        for entry in images:
            name, trigger = entry["name"], entry["trigger"]
            true_image = Image.open(entry["true_image"]).convert("RGB")
            decoy_image = Image.open(entry["decoy_concept_image"]).convert("RGB")
            prompt = f"{trigger}, {entry['prompt_suffix']}"

            print(f"=== [{name} / seed {seed}] baseline (shared multi-image LoRA) ===")
            baseline_images_out = generate_samples(
                args.checkpoint, str(baseline_lora), prompt,
                out_dir / name / str(seed) / "baseline", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            baseline_true = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in baseline_images_out]
            baseline_decoy = [clip_similarity(model, processor, decoy_image, Image.open(p).convert("RGB")) for p in baseline_images_out]

            print(f"=== [{name} / seed {seed}] misaligned (shared multi-image LoRA) ===")
            misaligned_images_out = generate_samples(
                args.checkpoint, str(misaligned_lora), prompt,
                out_dir / name / str(seed) / "misaligned", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            misaligned_true = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in misaligned_images_out]
            misaligned_decoy = [clip_similarity(model, processor, decoy_image, Image.open(p).convert("RGB")) for p in misaligned_images_out]

            avg_baseline_true = statistics.mean(baseline_true)
            avg_baseline_decoy = statistics.mean(baseline_decoy)
            avg_misaligned_true = statistics.mean(misaligned_true)
            avg_misaligned_decoy = statistics.mean(misaligned_decoy)
            runs.append(
                {
                    "name": name,
                    "seed": seed,
                    "avg_baseline_true": avg_baseline_true,
                    "avg_baseline_decoy": avg_baseline_decoy,
                    "avg_misaligned_true": avg_misaligned_true,
                    "avg_misaligned_decoy": avg_misaligned_decoy,
                    "delta_true": avg_baseline_true - avg_misaligned_true,
                    "delta_decoy": avg_misaligned_decoy - avg_baseline_decoy,
                }
            )

    del pipe_cache["pipe"]
    torch.cuda.empty_cache()

    print()
    print("=== per-run results (multi-image LoRA) ===")
    header = f"{'image':<20} {'seed':>5} {'base->true':>11} {'mis->true':>10} {'base->decoy':>12} {'mis->decoy':>11} {'d_true':>8} {'d_decoy':>8}"
    print(header)
    for r in runs:
        print(
            f"{r['name']:<20} {r['seed']:>5} {r['avg_baseline_true']:>11.4f} {r['avg_misaligned_true']:>10.4f} "
            f"{r['avg_baseline_decoy']:>12.4f} {r['avg_misaligned_decoy']:>11.4f} "
            f"{r['delta_true']:>+8.4f} {r['delta_decoy']:>+8.4f}"
        )

    n = len(runs)
    delta_true_vals = [r["delta_true"] for r in runs]
    delta_decoy_vals = [r["delta_decoy"] for r in runs]
    mean_delta_true = statistics.mean(delta_true_vals)
    mean_delta_decoy = statistics.mean(delta_decoy_vals)
    margin_true = ci_margin(delta_true_vals)
    margin_decoy = ci_margin(delta_decoy_vals)

    print()
    print(f"n = {n} (image x seed combinations, {len(seeds)} shared LoRAs per condition)")
    print(f"mean delta_true (baseline_sim_to_true - misaligned_sim_to_true): {mean_delta_true:+.4f}")
    print(f"95% CI (t-approx): [{mean_delta_true - margin_true:+.4f}, {mean_delta_true + margin_true:+.4f}]")
    print(f"mean delta_decoy (misaligned_sim_to_decoy - baseline_sim_to_decoy): {mean_delta_decoy:+.4f}")
    print(f"95% CI (t-approx): [{mean_delta_decoy - margin_decoy:+.4f}, {mean_delta_decoy + margin_decoy:+.4f}]")
    print()

    threshold = 0.03
    true_ci_excludes_zero = (mean_delta_true - margin_true) > 0
    decoy_ci_excludes_zero = (mean_delta_decoy - margin_decoy) > 0
    if true_ci_excludes_zero and decoy_ci_excludes_zero and mean_delta_true > threshold and mean_delta_decoy > threshold:
        verdict = "PASS (misalignment measurably drifts generation away from the true concept and toward the decoy, 95% CIs exclude zero on both)"
    elif mean_delta_true > threshold and mean_delta_decoy > threshold:
        verdict = f"WEAK PASS (both means above threshold but at least one 95% CI includes zero -- not statistically reliable at n={n})"
    else:
        verdict = "WEAK/FAIL (at least one mean at or below threshold)"
    print(f"=== Verdict (multi-image LoRA): {verdict} ===")
    print(f"    (mean_delta_true={mean_delta_true:+.4f}, mean_delta_decoy={mean_delta_decoy:+.4f}, threshold={threshold})")


if __name__ == "__main__":
    main()
