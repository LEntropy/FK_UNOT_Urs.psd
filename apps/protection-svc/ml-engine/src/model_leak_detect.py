"""Detection-side reuse of this project's own LoRA-training-protection
validation methodology (experiments/lora_validation/generate_and_score.py)
as a LEAK-detection signal, not a protection-effect measurement: given a
suspect LoRA file someone else trained and possibly published, does
generating images from it come out anomalously close to a specific
registered artwork -- evidence that artwork was used as training data
without authorization?

Same baseline-vs-condition delta methodology every protection-mechanism
validation in this project used (style_cloak/concept_misalign/
diffusion_attack/aspl_attack/hybrid_attack/ensemble_attack -- see
PHASE4_SCOPING.md and the lora-protection-research memory), just pointed
at the opposite question: instead of measuring whether OUR perturbation
lowered a LoRA's fidelity to the original, this measures whether a THIRD
PARTY's LoRA has anomalously HIGH fidelity to it.

    delta = clip_similarity(suspect_lora_samples, original)
          - clip_similarity(base_checkpoint_samples, original)

A large POSITIVE delta (the suspect LoRA is much more similar to the
original than the bare base checkpoint is, under the exact same prompts)
is the leak signal. Comparing against the same base checkpoint controls
for prompt/subject overlap that would otherwise inflate similarity even
with no LoRA at all (a generic "oil painting, landscape" prompt is
somewhat close to any landscape painting by chance).

No knowledge of the suspect LoRA's actual training trigger word is
assumed (a real infringer's trigger is unknown to us) -- callers should
pass prompts built from the artwork's own generic subject description
(title/tags), deliberately trigger-word-free, so this measures the LoRA's
*learned bias* toward the artwork's style/subject even under prompts that
invoke no special token, rather than depending on guessing whatever
bespoke trigger the infringer happened to pick.

Honesty note, same posture as every other mechanism in this project: the
threshold below (0.03) is the same magnitude this project's own protection
experiments treated as "a real effect, not noise" -- reused here for the
mirror-image question, not independently calibrated against real leaked-
model incidents (none exist yet to calibrate against). Treat a
SUSPECTED_LEAK verdict as a strong lead to investigate further, not
standalone proof.

Must run under kohya_ss's venv (diffusers/transformers/peft/torch+cuda) --
same requirement as diffusion_attack.py/aspl_attack.py, for the same
reason (a full StableDiffusionPipeline + load_lora_weights).
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

THRESHOLD = 0.03


def _generate_samples(pipe, prompt: str, num_samples: int, seed: int, resolution: int, out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pipe.device.type
    paths = []
    for i in range(num_samples):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        result = pipe(prompt, num_inference_steps=30, width=resolution, height=resolution, generator=generator)
        path = out_dir / f"{prefix}_{i:02d}.png"
        result.images[0].save(path)
        paths.append(path)
    return paths


def clip_similarity(model: CLIPModel, processor: CLIPProcessor, image_a: Image.Image, image_b: Image.Image) -> float:
    inputs = processor(images=[image_a, image_b], return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return float((features[0] @ features[1]).item())


def detect_model_leak(
    checkpoint_path: str,
    suspect_lora_path: str,
    original_image_path: str,
    prompts: list[str],
    num_samples: int = 4,
    resolution: int = 512,
    gen_seed: int = 42,
    out_dir: str | None = None,
) -> dict:
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, torch_dtype=dtype, safety_checker=None).to(device)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    original_image = Image.open(original_image_path).convert("RGB")

    samples_dir = Path(out_dir) if out_dir else Path(original_image_path).parent / "model_leak_samples"

    per_prompt = []
    for idx, prompt in enumerate(prompts):
        prompt_dir = samples_dir / f"prompt_{idx:02d}"

        base_paths = _generate_samples(pipe, prompt, num_samples, gen_seed, resolution, prompt_dir, "base")
        base_scores = [clip_similarity(clip_model, clip_processor, original_image, Image.open(p).convert("RGB")) for p in base_paths]
        avg_base = statistics.mean(base_scores)

        pipe.load_lora_weights(suspect_lora_path)
        suspect_paths = _generate_samples(pipe, prompt, num_samples, gen_seed, resolution, prompt_dir, "suspect")
        pipe.unload_lora_weights()
        suspect_scores = [clip_similarity(clip_model, clip_processor, original_image, Image.open(p).convert("RGB")) for p in suspect_paths]
        avg_suspect = statistics.mean(suspect_scores)

        per_prompt.append(
            {
                "prompt": prompt,
                "avgBaseSimilarity": avg_base,
                "avgSuspectSimilarity": avg_suspect,
                "delta": avg_suspect - avg_base,
                "baseSamplePaths": [str(p) for p in base_paths],
                "suspectSamplePaths": [str(p) for p in suspect_paths],
            }
        )

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    deltas = [p["delta"] for p in per_prompt]
    mean_delta = statistics.mean(deltas)
    stdev_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    if mean_delta > THRESHOLD:
        verdict = "SUSPECTED_LEAK"
    elif mean_delta > THRESHOLD / 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO_EVIDENCE"

    return {
        "perPrompt": per_prompt,
        "meanDelta": mean_delta,
        "stdevDelta": stdev_delta,
        "verdict": verdict,
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suspect-lora", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--prompts", required=True, help="'|'-separated list of prompts")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
    result = detect_model_leak(
        checkpoint_path=args.checkpoint,
        suspect_lora_path=args.suspect_lora,
        original_image_path=args.original,
        prompts=prompts,
        num_samples=args.num_samples,
        resolution=args.resolution,
        gen_seed=args.gen_seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result))
