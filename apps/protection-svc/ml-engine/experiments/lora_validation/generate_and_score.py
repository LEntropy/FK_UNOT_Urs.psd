"""Generates sample images from each trained LoRA and scores them against
the true, uncloaked style reference using CLIP image-image cosine
similarity.

Generation uses `diffusers`' StableDiffusionPipeline directly (loading the
checkpoint via from_single_file + the LoRA via load_lora_weights) rather
than kohya_ss's sd-scripts/gen_img.py -- gen_img.py in this sd-scripts
checkout has an internal API mismatch (`library.train_util` here has no
`load_tokenizer`, likely a version skew between gen_img.py and the rest of
this sd-scripts install) that isn't this experiment's concern to debug;
diffusers loading the exact same .safetensors files is standard, stable,
and already installed in this venv.

Why CLIP and not the project's existing Gram-matrix similarity
(perceptual_hash.py/model.py's StyleFeatureExtractor): style_cloak.py's
perturbation was optimized against VGG19 Gram matrices specifically --
reusing that same metric here to judge whether cloaking worked would be
circular (of course a VGG19-space attack shows up in a VGG19-space
metric; that says nothing about whether it transfers to what a diffusion
LoRA actually learns from). CLIP embeds images through a completely
different architecture never touched by the cloak's optimization, making
it a real, independent check.

Must run with kohya_ss's venv python (has diffusers/transformers/torch+cuda
already) -- NOT ml-engine's own .venv, which only has plain torch/Pillow.
"""

import argparse
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
) -> list[Path]:
    from diffusers import StableDiffusionPipeline

    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_single_file(
        checkpoint, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
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

    del pipe
    torch.cuda.empty_cache()
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
    parser.add_argument("--lora-baseline", required=True, help="LoRA trained on the uncloaked image")
    parser.add_argument("--lora-cloaked", required=True, help="LoRA trained on the cloaked image")
    parser.add_argument("--true-style-image", required=True, help="the real, uncloaked style_target.png-equivalent reference")
    parser.add_argument("--trigger", default="starrynighttest")
    parser.add_argument("--prompt-suffix", default="oil painting, landscape")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out" / "generated"))
    args = parser.parse_args()

    prompt = f"{args.trigger}, {args.prompt_suffix}"
    out_dir = Path(args.out_dir)

    print("=== generating samples: baseline (uncloaked) LoRA ===")
    baseline_images = generate_samples(
        args.checkpoint, args.lora_baseline, prompt,
        out_dir / "baseline", args.num_samples, args.seed, args.resolution,
    )
    print(f"  wrote {len(baseline_images)} images to {out_dir / 'baseline'}")

    print("=== generating samples: cloaked LoRA ===")
    cloaked_images = generate_samples(
        args.checkpoint, args.lora_cloaked, prompt,
        out_dir / "cloaked", args.num_samples, args.seed, args.resolution,
    )
    print(f"  wrote {len(cloaked_images)} images to {out_dir / 'cloaked'}")

    print()
    print("=== scoring: CLIP image-image cosine similarity vs true style reference ===")
    print("(architecturally independent of the VGG19 space style_cloak.py optimizes against -- not circular)")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    true_image = Image.open(args.true_style_image).convert("RGB")

    def score_all(images: list[Path]) -> list[float]:
        return [clip_similarity(model, processor, true_image, Image.open(p).convert("RGB")) for p in images]

    baseline_scores = score_all(baseline_images)
    cloaked_scores = score_all(cloaked_images)

    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_cloaked = sum(cloaked_scores) / len(cloaked_scores)
    delta = avg_baseline - avg_cloaked

    print()
    print(f"{'condition':<12} {'n':>3} {'avg CLIP sim':>14} {'min':>8} {'max':>8}")
    print(f"{'baseline':<12} {len(baseline_scores):>3} {avg_baseline:>14.4f} {min(baseline_scores):>8.4f} {max(baseline_scores):>8.4f}")
    print(f"{'cloaked':<12} {len(cloaked_scores):>3} {avg_cloaked:>14.4f} {min(cloaked_scores):>8.4f} {max(cloaked_scores):>8.4f}")
    print()
    print(f"delta (baseline - cloaked): {delta:+.4f} "
          f"({'cloak reduced style fidelity' if delta > 0 else 'cloak did NOT reduce style fidelity'})")
    print()

    # Threshold is deliberately conservative and stated explicitly, same
    # pattern as evaluate.py's PASS/WEAK-FAIL thresholds -- 0.03 is roughly
    # one order of magnitude above typical CLIP-similarity noise between
    # independent generations from the same LoRA (not derived from a large
    # calibration study; a real product decision should re-derive this from
    # more runs/seeds before trusting it at face value).
    threshold = 0.03
    verdict = "PASS (cloak measurably degrades LoRA style fidelity)" if delta > threshold else "WEAK/FAIL (no measurable degradation vs baseline LoRA)"
    print(f"=== Verdict: {verdict} (delta={delta:+.4f}, threshold={threshold}) ===")


if __name__ == "__main__":
    main()
