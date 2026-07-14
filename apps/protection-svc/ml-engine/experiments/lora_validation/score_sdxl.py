"""Scores the SDXL confirmation experiment. Same CLIP-similarity method as
the main SD1.5 experiment's generate_and_score.py, but generation uses
diffusers' StableDiffusionXLPipeline (Illustrious-XL is a genuine SDXL
checkpoint) instead of StableDiffusionPipeline, and both conditions
(baseline and cloaked) are freshly trained here -- unlike the
target-dissimilarity/L2-preset follow-ups, there's no existing SDXL
baseline to reuse.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def generate_samples(checkpoint, lora_weights, prompt, out_dir, num_samples, seed, resolution, pipe_cache):
    from diffusers import StableDiffusionXLPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    if "pipe" not in pipe_cache:
        pipe_cache["pipe"] = StableDiffusionXLPipeline.from_single_file(
            checkpoint, torch_dtype=torch.float16
        ).to("cuda")
    pipe = pipe_cache["pipe"]
    pipe.unload_lora_weights()
    pipe.load_lora_weights(lora_weights)

    images = []
    for i in range(num_samples):
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        result = pipe(prompt, num_inference_steps=30, width=resolution, height=resolution, generator=generator)
        image_path = out_dir / f"sample_{i:02d}.png"
        result.images[0].save(image_path)
        images.append(image_path)
    return images


def clip_similarity(model, processor, image_a, image_b):
    inputs = processor(images=[image_a, image_b], return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return float((features[0] @ features[1]).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lora-root", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out" / "sdxl" / "generated"))
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
            print(f"=== [sdxl {name} / seed {seed}] baseline ===")
            baseline_lora = lora_root / f"lora_sdxl_{name}_{seed}_baseline" / f"baseline_{args.run_name}.safetensors"
            baseline_images = generate_samples(
                args.checkpoint, str(baseline_lora), prompt,
                out_dir / name / str(seed) / "baseline", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            baseline_scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in baseline_images]

            print(f"=== [sdxl {name} / seed {seed}] cloaked ===")
            cloaked_lora = lora_root / f"lora_sdxl_{name}_{seed}_cloaked" / f"cloaked_{args.run_name}.safetensors"
            cloaked_images = generate_samples(
                args.checkpoint, str(cloaked_lora), prompt,
                out_dir / name / str(seed) / "cloaked", args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            cloaked_scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in cloaked_images]

            avg_baseline = statistics.mean(baseline_scores)
            avg_cloaked = statistics.mean(cloaked_scores)
            runs.append({"name": name, "seed": seed, "avg_baseline": avg_baseline, "avg_cloaked": avg_cloaked, "delta": avg_baseline - avg_cloaked})

    del pipe_cache["pipe"]
    torch.cuda.empty_cache()

    print()
    print("=== per-run results (SDXL / Illustrious-XL) ===")
    print(f"{'image':<14} {'seed':>5} {'baseline':>10} {'cloaked':>10} {'delta':>8}")
    for r in runs:
        print(f"{r['name']:<14} {r['seed']:>5} {r['avg_baseline']:>10.4f} {r['avg_cloaked']:>10.4f} {r['delta']:>+8.4f}")

    deltas = [r["delta"] for r in runs]
    n = len(deltas)
    mean_delta = statistics.mean(deltas)
    stdev_delta = statistics.stdev(deltas) if n > 1 else 0.0
    t_table = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    t_crit = t_table.get(n - 1, 1.96)
    margin = t_crit * (stdev_delta / (n ** 0.5)) if n > 1 else float("nan")

    print()
    print(f"n = {n}")
    print(f"mean delta (baseline - cloaked): {mean_delta:+.4f}")
    print(f"stdev: {stdev_delta:.4f}")
    print(f"95% CI (t-approx): [{mean_delta - margin:+.4f}, {mean_delta + margin:+.4f}]")
    print()
    print("compare to SD1.5 main experiment: mean delta +0.0130 (95% CI [+0.0066, +0.0193], n=30)")


if __name__ == "__main__":
    main()
