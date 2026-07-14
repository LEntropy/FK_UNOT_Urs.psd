"""Scores the L2_PORTFOLIO preset-scaling experiment. Reuses each image's
existing baseline LoRA (already trained in the main experiment) and its
already-recorded baseline CLIP score per seed -- only the 4 new
L2_PORTFOLIO-cloaked LoRAs need fresh generation+scoring.

Baseline CLIP-similarity-to-true-image scores per (image, seed), from the
main experiment's n=30 report (ml-engine/README.md's stage-6 table):
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASELINE_CLIP_SIM = {
    "great_wave": {1: 0.9092, 2: 0.9205, 3: 0.9208},
    "starry_night": {1: 0.8695, 2: 0.8868, 3: 0.8535},
    "night_watch": {1: 0.8175, 2: 0.8027, 3: 0.8321},
    "mona_lisa": {1: 0.7487, 2: 0.7424, 3: 0.7550},
}

# L3_ANTI_TRAIN mean deltas for the same 4 images, from the main n=30
# experiment (for direct comparison, not recomputed here).
L3_MEAN_DELTA = {
    "great_wave": (0.0339 + 0.0209 + 0.0318) / 3,
    "starry_night": (0.0062 + 0.0415 + 0.0116) / 3,
    "night_watch": (0.0115 + 0.0364 + 0.0264) / 3,
    "mona_lisa": (0.0061 - 0.0128 - 0.0006) / 3,
}


def generate_samples(checkpoint, lora_weights, prompt, out_dir, num_samples, seed, resolution, pipe_cache):
    from diffusers import StableDiffusionPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    if "pipe" not in pipe_cache:
        pipe_cache["pipe"] = StableDiffusionPipeline.from_single_file(
            checkpoint, torch_dtype=torch.float16, safety_checker=None
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
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out" / "generated_l2"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    lora_root = Path(args.lora_root)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pipe_cache: dict = {}

    results = []
    for entry in manifest:
        name = entry["name"]
        true_image = Image.open(entry["true_image"]).convert("RGB")
        prompt = f"{entry['trigger']}, {entry['prompt_suffix']}"

        for seed in seeds:
            print(f"=== [{name} L2 / seed {seed}] ===")
            lora_path = lora_root / f"lora_l2_{name}_{seed}_cloaked" / "cloaked_v1.safetensors"
            images = generate_samples(
                args.checkpoint, str(lora_path), prompt,
                out_dir / name / str(seed), args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in images]
            avg_cloaked = statistics.mean(scores)
            baseline = BASELINE_CLIP_SIM[name][seed]
            delta = baseline - avg_cloaked
            results.append({"name": name, "seed": seed, "l2_cloaked_sim": avg_cloaked, "l2_delta": delta})

    del pipe_cache["pipe"]
    torch.cuda.empty_cache()

    print()
    print("=== per-run results (L2_PORTFOLIO) ===")
    print(f"{'image':<16} {'seed':>5} {'L2 cloaked CLIP':>16} {'L2 delta':>10} {'L3 delta (ref)':>15}")
    for r in results:
        print(f"{r['name']:<16} {r['seed']:>5} {r['l2_cloaked_sim']:>16.4f} {r['l2_delta']:>+10.4f} {L3_MEAN_DELTA[r['name']]:>+15.4f}")

    by_name: dict[str, list[float]] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r["l2_delta"])

    print()
    print("=== L2 vs L3 mean delta comparison ===")
    print(f"{'image':<16} {'L2 mean delta':>14} {'L3 mean delta':>14} {'L2/L3 ratio':>12}")
    all_l2, all_l3 = [], []
    for name in by_name:
        l2_mean = statistics.mean(by_name[name])
        l3_mean = L3_MEAN_DELTA[name]
        ratio = l2_mean / l3_mean if l3_mean != 0 else float("nan")
        print(f"{name:<16} {l2_mean:>+14.4f} {l3_mean:>+14.4f} {ratio:>12.2f}")
        all_l2.append(l2_mean)
        all_l3.append(l3_mean)

    overall_l2 = statistics.mean(all_l2)
    overall_l3 = statistics.mean(all_l3)
    print()
    print(f"overall mean delta: L2_PORTFOLIO={overall_l2:+.4f}  L3_ANTI_TRAIN={overall_l3:+.4f}")
    if overall_l2 < overall_l3 * 0.7:
        print("=> effect scales down substantially with weaker preset (epsilon) -- consistent with VGG-space metrics")
    elif overall_l2 > overall_l3 * 1.3:
        print("=> effect scales UP with weaker preset -- inconsistent with VGG-space metrics, worth double-checking")
    else:
        print("=> effect is roughly similar across presets -- real LoRA impact may not scale with epsilon the way VGG-space drift does")


if __name__ == "__main__":
    main()
