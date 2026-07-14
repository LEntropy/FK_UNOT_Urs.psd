"""Scores the target-dissimilarity confirmation experiment. Reuses the
main experiment's existing starry_night baseline LoRA weights and their
already-recorded CLIP scores (deterministic given the same LoRA + same
generation seed, so regenerating them would just reproduce the same
numbers) -- only the 4 new cloaked-toward-a-controlled-target LoRAs
actually need fresh generation+scoring here.

Baseline CLIP-similarity-to-true-starry_night scores per seed, from the
main experiment's n=30 report (ml-engine/README.md's stage-6 table):
    seed 1: 0.8695
    seed 2: 0.8868
    seed 3: 0.8535
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASELINE_CLIP_SIM = {1: 0.8695, 2: 0.8868, 3: 0.8535}


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
    parser.add_argument("--manifest", required=True, help="target_dissimilarity_manifest.json from prepare_target_dissimilarity.py")
    parser.add_argument("--lora-root", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--true-image", required=True)
    parser.add_argument("--trigger", default="starrynighttest")
    parser.add_argument("--prompt-suffix", default="oil painting, landscape, night sky")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out" / "generated_dissimilarity"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    lora_root = Path(args.lora_root)
    prompt = f"{args.trigger}, {args.prompt_suffix}"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    true_image = Image.open(args.true_image).convert("RGB")
    pipe_cache: dict = {}

    results = []
    for entry in manifest:
        target_name, similarity = entry["target_name"], entry["similarity"]
        for seed in seeds:
            print(f"=== [vs {target_name}, sim={similarity} / seed {seed}] ===")
            lora_path = lora_root / f"lora_starry_night_vs_{target_name}_{seed}_cloaked" / "cloaked_v1.safetensors"
            images = generate_samples(
                args.checkpoint, str(lora_path), prompt,
                out_dir / target_name / str(seed), args.num_samples, args.gen_seed, args.resolution, pipe_cache,
            )
            scores = [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in images]
            avg_cloaked = statistics.mean(scores)
            baseline = BASELINE_CLIP_SIM[seed]
            delta = baseline - avg_cloaked
            results.append({"target": target_name, "similarity": similarity, "seed": seed, "cloaked_sim": avg_cloaked, "delta": delta})

    del pipe_cache["pipe"]
    torch.cuda.empty_cache()

    print()
    print("=== per-run results ===")
    print(f"{'target':<20} {'orig_sim':>9} {'seed':>5} {'cloaked_clip':>13} {'delta':>8}")
    for r in results:
        print(f"{r['target']:<20} {r['similarity']:>9.4f} {r['seed']:>5} {r['cloaked_sim']:>13.4f} {r['delta']:>+8.4f}")

    # Per-target mean delta, then correlate against similarity (the whole
    # point: is this a real dose-response relationship, isolated to a
    # single base image, not an artifact of which 5 pairs got cross-cloaked
    # in the main experiment).
    by_target: dict[str, list[float]] = {}
    for r in results:
        by_target.setdefault(r["target"], []).append(r["delta"])

    print()
    print("=== per-target mean delta ===")
    sims, deltas = [], []
    for entry in manifest:
        name, sim = entry["target_name"], entry["similarity"]
        mean_delta = statistics.mean(by_target[name])
        print(f"{name:<20} sim={sim:.4f}  mean_delta={mean_delta:+.4f}")
        sims.append(sim)
        deltas.append(mean_delta)

    # Include the main experiment's already-measured great_wave point
    # (similarity 0.7445, mean CLIP delta +0.0198 for starry_night) to
    # complete the 5-point spectrum without retraining it.
    sims.append(0.7445)
    deltas.append((0.0062 + 0.0415 + 0.0116) / 3)
    print(f"{'great_wave (reused)':<20} sim=0.7445  mean_delta={deltas[-1]:+.4f}")

    n = len(sims)
    ms, md = sum(sims) / n, sum(deltas) / n
    cov = sum((s - ms) * (d - md) for s, d in zip(sims, deltas))
    vs = sum((s - ms) ** 2 for s in sims)
    vd = sum((d - md) ** 2 for d in deltas)
    r = cov / (vs * vd) ** 0.5 if vs > 0 and vd > 0 else float("nan")

    print()
    print(f"Pearson r (similarity vs. mean delta), single base image, n={n} controlled target points: {r:+.3f}")
    print("  (compare to r=-0.929 from the main experiment's 5 incidental cross-cloaked pairs)")


if __name__ == "__main__":
    main()
