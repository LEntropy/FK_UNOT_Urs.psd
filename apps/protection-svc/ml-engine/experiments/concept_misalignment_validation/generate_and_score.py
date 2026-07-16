"""Generates sample images from each trained LoRA (baseline vs
misaligned) and scores them against BOTH the true concept image and the
decoy concept image using CLIP image-image cosine similarity --
PHASE4_SCOPING.md §1's exact recommended measurement: "CLIP-similarity
between generated samples and both the true concept and the decoy
concept, same shape as the existing baseline-vs-cloaked delta
measurement."

This is the actual test of the claim concept_misalign.py makes: if
misalignment worked, training on (misaligned image, true caption) should
make generation from that caption drift toward the decoy concept's
features, not the true image's -- i.e. sim_to_true should drop and
sim_to_decoy should rise, relative to the baseline (unmisaligned)
condition trained on the identical caption.

Structurally mirrors experiments/lora_validation/generate_and_score.py
(same diffusers-direct generation approach, same reasoning for using CLIP
instead of this project's own Gram-matrix/embedding metrics -- see that
file's docstring) but reports two deltas per run instead of one.

Expects trained LoRA weights at
{lora-root}/lora_{name}_{seed}_{condition}/{condition}_v{run}.safetensors
for every (name, seed) in the manifest x --seeds, matching
run_concept_misalignment_validation.ps1's naming convention exactly.

Must run with kohya_ss's venv python (has diffusers/transformers/peft/
torch+cuda already) -- NOT ml-engine's own .venv, which only has plain
torch/Pillow.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def generate_samples(
    checkpoint: str,
    lora_weights: str,
    prompt: str,
    out_dir: Path,
    num_samples: int,
    seed: int,
    resolution: int,
    pipe_cache: dict,
) -> list[Path]:
    from diffusers import StableDiffusionPipeline

    out_dir.mkdir(parents=True, exist_ok=True)

    if "pipe" not in pipe_cache:
        pipe_cache["pipe"] = StableDiffusionPipeline.from_single_file(
            checkpoint, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")
    pipe = pipe_cache["pipe"]
    pipe.unload_lora_weights()
    pipe.load_lora_weights(lora_weights)

    images: list[Path] = []
    for i in range(num_samples):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        result = pipe(
            prompt,
            num_inference_steps=30,
            width=resolution,
            height=resolution,
            generator=generator,
        )
        image_path = out_dir / f"sample_{i:02d}.png"
        result.images[0].save(image_path)
        images.append(image_path)

    return images


def clip_similarity(model: CLIPModel, processor: CLIPProcessor, image_a: Image.Image, image_b: Image.Image) -> float:
    inputs = processor(images=[image_a, image_b], return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return float((features[0] @ features[1]).item())


def ci_margin(values: list[float]) -> float:
    """Same t-approximation as lora_validation/generate_and_score.py --
    see that file for why (small n, report the raw spread alongside it,
    don't lean on this alone)."""
    n = len(values)
    if n <= 1:
        return float("nan")
    stdev = statistics.stdev(values)
    t_table = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    t_crit = t_table.get(n - 1, 1.96)
    return t_crit * (stdev / (n**0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SD1.5 base checkpoint path")
    parser.add_argument("--manifest", required=True, help="manifest.json written by prepare_dataset.py")
    parser.add_argument("--lora-root", required=True, help="dir containing lora_{name}_{seed}_{condition}/ subfolders")
    parser.add_argument("--seeds", required=True, help="comma-separated training seeds, e.g. '1,2,3'")
    parser.add_argument("--run-name", default="v1", help="output_name suffix used when training (e.g. baseline_v1.safetensors -> 'v1')")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=42, help="base seed for image generation (independent of training seed)")
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
        name, trigger = entry["name"], entry["trigger"]
        true_image = Image.open(entry["true_image"]).convert("RGB")
        decoy_image = Image.open(entry["decoy_concept_image"]).convert("RGB")
        # Same trigger + prompt for both conditions -- unlike the style-
        # cloak cross-cloaking experiment, the caption is meant to
        # correctly describe the true image throughout (see this file's
        # module doc for why that's the point).
        prompt = f"{trigger}, {entry['prompt_suffix']}"

        for seed in seeds:
            print(f"=== [{name} / seed {seed}] baseline ===")
            baseline_lora = lora_root / f"lora_{name}_{seed}_baseline" / f"baseline_{args.run_name}.safetensors"
            baseline_images = generate_samples(
                args.checkpoint, str(baseline_lora), prompt,
                out_dir / name / str(seed) / "baseline", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            baseline_true = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in baseline_images]
            baseline_decoy = [clip_similarity(model, processor, decoy_image, Image.open(p).convert("RGB")) for p in baseline_images]

            print(f"=== [{name} / seed {seed}] misaligned ===")
            misaligned_lora = lora_root / f"lora_{name}_{seed}_misaligned" / f"misaligned_{args.run_name}.safetensors"
            misaligned_images = generate_samples(
                args.checkpoint, str(misaligned_lora), prompt,
                out_dir / name / str(seed) / "misaligned", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            misaligned_true = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in misaligned_images]
            misaligned_decoy = [clip_similarity(model, processor, decoy_image, Image.open(p).convert("RGB")) for p in misaligned_images]

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
                    # Protection working = true similarity drops (positive
                    # delta_true) AND decoy similarity rises (positive
                    # delta_decoy) when trained on the misaligned condition.
                    "delta_true": avg_baseline_true - avg_misaligned_true,
                    "delta_decoy": avg_misaligned_decoy - avg_baseline_decoy,
                }
            )

    del pipe_cache["pipe"]
    torch.cuda.empty_cache()

    print()
    print("=== per-run results ===")
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
    print(f"n = {n} (image x seed combinations)")
    print(f"mean delta_true (baseline_sim_to_true - misaligned_sim_to_true): {mean_delta_true:+.4f}")
    print(f"95% CI (t-approx): [{mean_delta_true - margin_true:+.4f}, {mean_delta_true + margin_true:+.4f}]")
    print(f"mean delta_decoy (misaligned_sim_to_decoy - baseline_sim_to_decoy): {mean_delta_decoy:+.4f}")
    print(f"95% CI (t-approx): [{mean_delta_decoy - margin_decoy:+.4f}, {mean_delta_decoy + margin_decoy:+.4f}]")
    print()

    # Same conservative, not-independently-calibrated threshold as
    # lora_validation's single-metric verdict -- see ml-engine/README.md's
    # caveats on why 0.03 specifically and why it isn't recalibrated per
    # experiment.
    threshold = 0.03
    true_ci_excludes_zero = (mean_delta_true - margin_true) > 0
    decoy_ci_excludes_zero = (mean_delta_decoy - margin_decoy) > 0
    if true_ci_excludes_zero and decoy_ci_excludes_zero and mean_delta_true > threshold and mean_delta_decoy > threshold:
        verdict = "PASS (misalignment measurably drifts generation away from the true concept and toward the decoy, 95% CIs exclude zero on both)"
    elif mean_delta_true > threshold and mean_delta_decoy > threshold:
        verdict = f"WEAK PASS (both means above threshold but at least one 95% CI includes zero -- not statistically reliable at n={n})"
    else:
        verdict = "WEAK/FAIL (at least one mean at or below threshold)"
    print(f"=== Verdict: {verdict} ===")
    print(f"    (mean_delta_true={mean_delta_true:+.4f}, mean_delta_decoy={mean_delta_decoy:+.4f}, threshold={threshold})")


if __name__ == "__main__":
    main()
