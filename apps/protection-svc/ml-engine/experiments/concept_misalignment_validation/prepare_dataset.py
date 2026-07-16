"""Builds kohya_ss-style dataset folders + dataset_config TOML files for
the Concept Misalignment Layer validation experiment -- PHASE4_SCOPING.md
§1's own "recommended validation methodology," not yet run anywhere (see
that section and concept_misalign.py's module doc for the full context).

Mirrors apps/protection-svc/ml-engine/experiments/lora_validation/ (the
ai-engine branch's real LoRA-validation experiment for style_cloak.py)
structurally, but the question is different: not "does the cloak degrade
LoRA style fidelity" but "does training on a concept-misaligned image with
its real caption make generation from that caption drift toward the decoy
concept instead of the true one." Two conditions per image, everything
else held identical so misaligned-vs-not is the only variable:

    baseline:   dataset/{name}/{repeats}_{trigger}/           <- the real image
    misaligned: dataset_misaligned/{name}/{repeats}_{trigger}/ <- concept_misalign.misalign() output

Both conditions use the *same* trigger word and prompt -- unlike the style
cloak experiment's cross-cloaking, the caption here is meant to correctly
describe the true image throughout; what changes is only the *pixels*
that word gets trained against.

Runs on the ml-engine venv (needs torch+Pillow for misalign()'s
CLIP-embedding optimization loop -- CPU works but is slow; GPU is not
required for this step, only for the training step this script's output
feeds into).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from concept_misalign import misalign  # noqa: E402

TOML_TEMPLATE = """[general]
enable_bucket = false

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{image_dir}'
  class_tokens = '{trigger}'
  num_repeats = {num_repeats}
"""

# n=5 images to start, matching PHASE4_SCOPING.md §1's "start at n=4-6
# images, expand only if the aggregate signal looks real" -- same
# incremental-sample-size practice this project already followed for the
# style-cloak LoRA validation (4 -> 6 -> 8 images) and the SDXL/
# L2_PORTFOLIO re-validations. concept_target is a decoy CONCEPT, not (as
# in the cross-cloaking style experiment) just "some other painting" --
# picked here for a clear, checkable category shift (portrait -> landscape
# or vice versa) so a human glancing at generated samples can sanity-check
# the CLIP delta means what it claims to mean.
IMAGE_CONFIGS = [
    {
        "name": "mona_lisa",
        "image": ML_ENGINE_DIR / "out" / "real" / "mona_lisa.jpg",
        "concept_target": ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg",
        "trigger": "monalisacmtest",
        "prompt_suffix": "oil painting, portrait of a woman",
    },
    {
        "name": "starry_night",
        "image": ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg",
        "concept_target": ML_ENGINE_DIR / "out" / "real" / "the_scream.jpg",
        "trigger": "starrynightcmtest",
        "prompt_suffix": "oil painting, landscape, night sky",
    },
    {
        "name": "the_scream",
        "image": ML_ENGINE_DIR / "out" / "real" / "the_scream.jpg",
        "concept_target": ML_ENGINE_DIR / "out" / "real" / "great_wave.jpg",
        "trigger": "screamcmtest",
        "prompt_suffix": "expressionist painting, portrait, screaming figure",
    },
    {
        "name": "great_wave",
        "image": ML_ENGINE_DIR / "out" / "real" / "great_wave.jpg",
        "concept_target": ML_ENGINE_DIR / "out" / "real" / "girl_pearl_earring.jpg",
        "trigger": "greatwavecmtest",
        "prompt_suffix": "woodblock print, ocean wave, landscape",
    },
    {
        "name": "girl_pearl_earring",
        "image": ML_ENGINE_DIR / "out" / "real" / "girl_pearl_earring.jpg",
        "concept_target": ML_ENGINE_DIR / "out" / "real" / "mona_lisa.jpg",
        "trigger": "pearlearringcmtest",
        "prompt_suffix": "baroque painting, portrait of a girl, dark background",
    },
]


def build_condition(out_root: Path, image_path: Path, trigger: str, num_repeats: int, resolution: int) -> Path:
    concept_dir = out_root / f"{num_repeats}_{trigger}"
    concept_dir.mkdir(parents=True, exist_ok=True)
    dest = concept_dir / image_path.name
    shutil.copy(image_path, dest)

    toml_path = out_root.parent / f"dataset_config_{out_root.name}.toml"
    toml_path.write_text(
        TOML_TEMPLATE.format(
            resolution=resolution,
            image_dir=str(concept_dir),
            trigger=trigger,
            num_repeats=num_repeats,
        )
    )
    return toml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="training resolution -- also the size misalign() runs at, so the optimization sees exactly what training sees",
    )
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"])
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for cfg in IMAGE_CONFIGS:
        name = cfg["name"]
        print(f"=== [{name}] preparing baseline (unmisaligned) condition ===")
        baseline_toml = build_condition(
            out_dir / "dataset" / name, cfg["image"], cfg["trigger"], args.num_repeats, args.resolution
        )
        print(f"  wrote {baseline_toml}")

        print(f"=== [{name}] preparing misaligned condition ({args.preset}, size={args.resolution}) ===")
        # Misaligns at the actual training resolution, same reasoning as
        # lora_validation/prepare_dataset.py's equivalent comment: the
        # validated-at-size-256 preset numbers matter less than matching
        # what the LoRA trainer actually sees.
        misaligned_image_path = out_dir / f"misaligned_{name}.png"
        misalign(
            original_path=str(cfg["image"]),
            concept_target_path=str(cfg["concept_target"]),
            output_path=str(misaligned_image_path),
            preset_name=args.preset,
            size=args.resolution,
            eot=False,
        )
        misaligned_toml = build_condition(
            out_dir / "dataset_misaligned" / name,
            misaligned_image_path,
            cfg["trigger"],
            args.num_repeats,
            args.resolution,
        )
        print(f"  wrote {misaligned_toml}")

        manifest.append(
            {
                "name": name,
                "trigger": cfg["trigger"],
                "prompt_suffix": cfg["prompt_suffix"],
                "true_image": str(cfg["image"]),
                "decoy_concept_image": str(cfg["concept_target"]),
                "baseline_dataset_config": str(baseline_toml),
                "misaligned_dataset_config": str(misaligned_toml),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("=== done ===")
    print(f"manifest written to {manifest_path}")
    for entry in manifest:
        print(f"  [{entry['name']}] trigger={entry['trigger']}")


if __name__ == "__main__":
    main()
