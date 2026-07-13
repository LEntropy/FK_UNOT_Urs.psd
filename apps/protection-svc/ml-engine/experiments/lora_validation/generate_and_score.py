"""Generates sample images from each trained LoRA and scores them against
the true, uncloaked style reference using CLIP image-image cosine
similarity -- batched across every (image, seed) run the orchestrator
trained, so the result is a distribution (mean/std/individual deltas),
not a single data point.

Generation uses `diffusers`' StableDiffusionPipeline directly (loading the
checkpoint via from_single_file + the LoRA via load_lora_weights) rather
than kohya_ss's sd-scripts/gen_img.py -- gen_img.py in this sd-scripts
checkout has an internal API mismatch (`library.train_util` here has no
`load_tokenizer`) that isn't this experiment's concern to debug; diffusers
loading the exact same .safetensors files is standard and already
installed in this venv (plus `peft`, needed for load_lora_weights).

Why CLIP and not the project's existing Gram-matrix similarity
(perceptual_hash.py/model.py's StyleFeatureExtractor): style_cloak.py's
perturbation was optimized against VGG19 Gram matrices specifically --
reusing that same metric here to judge whether cloaking worked would be
circular. CLIP embeds images through a completely different architecture
never touched by the cloak's optimization, making it a real, independent
check.

Expects trained LoRA weights at
{lora-root}/lora_{name}_{seed}_{condition}/{condition}_v{run}.safetensors
for every (name, seed) in the manifest x --seeds, matching
run_lora_validation.ps1's naming convention exactly.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SD1.5 base checkpoint path")
    parser.add_argument("--manifest", required=True, help="manifest.json written by prepare_dataset.py")
    parser.add_argument("--lora-root", required=True, help="dir containing lora_{name}_{seed}_{condition}/ subfolders")
    parser.add_argument("--seeds", required=True, help="comma-separated training seeds, e.g. '1,2,3'")
    parser.add_argument("--run-name", default="v1", help="output_name suffix used when training (e.g. baseline_v1.safetensors -> 'v1')")
    parser.add_argument("--prompt-suffix", default="oil painting, landscape")
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
        name, trigger, true_image_path = entry["name"], entry["trigger"], entry["true_image"]
        true_image = Image.open(true_image_path).convert("RGB")
        prompt = f"{trigger}, {args.prompt_suffix}"

        for seed in seeds:
            print(f"=== [{name} / seed {seed}] baseline ===")
            baseline_lora = lora_root / f"lora_{name}_{seed}_baseline" / f"baseline_{args.run_name}.safetensors"
            baseline_images = generate_samples(
                args.checkpoint, str(baseline_lora), prompt,
                out_dir / name / str(seed) / "baseline", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            baseline_scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in baseline_images]

            print(f"=== [{name} / seed {seed}] cloaked ===")
            cloaked_lora = lora_root / f"lora_{name}_{seed}_cloaked" / f"cloaked_{args.run_name}.safetensors"
            cloaked_images = generate_samples(
                args.checkpoint, str(cloaked_lora), prompt,
                out_dir / name / str(seed) / "cloaked", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            cloaked_scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in cloaked_images]

            avg_baseline = statistics.mean(baseline_scores)
            avg_cloaked = statistics.mean(cloaked_scores)
            runs.append(
                {
                    "name": name,
                    "seed": seed,
                    "avg_baseline": avg_baseline,
                    "avg_cloaked": avg_cloaked,
                    "delta": avg_baseline - avg_cloaked,
                }
            )

    del pipe_cache["pipe"]
    torch.cuda.empty_cache()

    print()
    print("=== per-run results ===")
    print(f"{'image':<14} {'seed':>5} {'baseline':>10} {'cloaked':>10} {'delta':>8}")
    for r in runs:
        print(f"{r['name']:<14} {r['seed']:>5} {r['avg_baseline']:>10.4f} {r['avg_cloaked']:>10.4f} {r['delta']:>+8.4f}")

    deltas = [r["delta"] for r in runs]
    mean_delta = statistics.mean(deltas)
    stdev_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    n = len(deltas)
    # 95% CI via t-distribution approximation (n is small -- report the
    # raw spread too, don't lean on this alone). For n<=1 no interval is
    # meaningful.
    if n > 1:
        # t-critical for common small-n df, else fall back to normal z=1.96
        t_table = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        t_crit = t_table.get(n - 1, 1.96)
        margin = t_crit * (stdev_delta / (n ** 0.5))
    else:
        margin = float("nan")

    print()
    print(f"n = {n} (image x seed combinations)")
    print(f"mean delta (baseline - cloaked): {mean_delta:+.4f}")
    print(f"stdev: {stdev_delta:.4f}")
    print(f"95% CI (t-approx): [{mean_delta - margin:+.4f}, {mean_delta + margin:+.4f}]")
    print(f"individual deltas: {[round(d, 4) for d in deltas]}")
    print()

    # Same conservative, not-independently-calibrated threshold as the
    # single-run experiment -- see ml-engine/README.md's caveats.
    threshold = 0.03
    ci_excludes_zero_and_threshold = (mean_delta - margin) > 0
    if ci_excludes_zero_and_threshold and mean_delta > threshold:
        verdict = "PASS (cloak measurably degrades LoRA style fidelity, 95% CI excludes zero)"
    elif mean_delta > threshold:
        verdict = "WEAK PASS (mean above threshold but 95% CI includes zero -- not statistically reliable at n={})".format(n)
    else:
        verdict = "WEAK/FAIL (mean at or below threshold)"
    print(f"=== Verdict: {verdict} (mean_delta={mean_delta:+.4f}, threshold={threshold}) ===")


if __name__ == "__main__":
    main()
