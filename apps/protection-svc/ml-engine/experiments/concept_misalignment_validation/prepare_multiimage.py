"""Follow-up to prepare_dataset.py's single-image result (WEAK/FAIL, see
PHASE4_SCOPING.md §1's "Update" note) -- tests whether the null result was
about the *mechanism* or about the *single-image LoRA setup* diluting it.

The first experiment trained one LoRA per (image, condition) pair, each
seeing exactly one (image, trigger) example repeated `num_repeats` times --
closer to memorizing that one pair than learning a generalizable
caption-to-visual-feature association a small pixel perturbation could
bend. This experiment instead trains ONE LoRA per condition per seed that
sees all 5 images/triggers *together*, kohya's standard multi-subset
config (matches how a real scraper's training set would actually look --
many different images/captions in one training run, not one).

Two conditions, same trigger/caption per image in both so the caption is
never the variable:
    baseline_multi:   all 5 images, unmisaligned, one shared LoRA
    misaligned_multi: all 5 images, misaligned, one shared LoRA

Reuses IMAGE_CONFIGS from prepare_dataset.py so both experiments test the
exact same images/targets/prompts -- only the training configuration
(isolated vs joint) differs, keeping this a controlled follow-up rather
than a new experiment with new confounds.

Runs on the ml-engine venv (needs torch+Pillow+open_clip for misalign()'s
CLIP-embedding optimization loop -- CPU works but is slow).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_misalign import misalign  # noqa: E402
from prepare_dataset import IMAGE_CONFIGS  # noqa: E402

SUBSET_TEMPLATE = """  [[datasets.subsets]]
  image_dir = '{image_dir}'
  class_tokens = '{trigger}'
  num_repeats = {num_repeats}
"""

TOML_HEADER = """[general]
enable_bucket = false

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

"""


def build_multi_condition(out_root: Path, images: list[tuple[Path, str]], num_repeats: int, resolution: int, config_name: str) -> Path:
    """images: list of (image_path, trigger) pairs, one per concept -- all
    copied into their own per-trigger subfolder under out_root, then
    referenced as separate kohya dataset subsets in one shared TOML so a
    single LoRA training run sees all of them."""
    subsets_toml = ""
    for image_path, trigger in images:
        concept_dir = out_root / f"{num_repeats}_{trigger}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, concept_dir / image_path.name)
        subsets_toml += SUBSET_TEMPLATE.format(image_dir=str(concept_dir), trigger=trigger, num_repeats=num_repeats)

    toml_path = out_root.parent / f"dataset_config_{config_name}.toml"
    toml_path.write_text(TOML_HEADER.format(resolution=resolution) + subsets_toml)
    return toml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"])
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out_multiimage"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_images: list[tuple[Path, str]] = []
    misaligned_images: list[tuple[Path, str]] = []
    manifest_entries = []

    for cfg in IMAGE_CONFIGS:
        name = cfg["name"]
        baseline_images.append((cfg["image"], cfg["trigger"]))

        print(f"=== [{name}] misaligning ({args.preset}, size={args.resolution}) ===")
        misaligned_image_path = out_dir / f"misaligned_{name}.png"
        misalign(
            original_path=str(cfg["image"]),
            concept_target_path=str(cfg["concept_target"]),
            output_path=str(misaligned_image_path),
            preset_name=args.preset,
            size=args.resolution,
            eot=False,
        )
        misaligned_images.append((misaligned_image_path, cfg["trigger"]))

        manifest_entries.append(
            {
                "name": name,
                "trigger": cfg["trigger"],
                "prompt_suffix": cfg["prompt_suffix"],
                "true_image": str(cfg["image"]),
                "decoy_concept_image": str(cfg["concept_target"]),
            }
        )

    print("=== building joint baseline dataset (all 5 images, one LoRA) ===")
    baseline_toml = build_multi_condition(
        out_dir / "dataset_baseline_multi", baseline_images, args.num_repeats, args.resolution, "baseline_multi"
    )
    print(f"  wrote {baseline_toml}")

    print("=== building joint misaligned dataset (all 5 images, one LoRA) ===")
    misaligned_toml = build_multi_condition(
        out_dir / "dataset_misaligned_multi", misaligned_images, args.num_repeats, args.resolution, "misaligned_multi"
    )
    print(f"  wrote {misaligned_toml}")

    manifest = {
        "images": manifest_entries,
        "baseline_dataset_config": str(baseline_toml),
        "misaligned_dataset_config": str(misaligned_toml),
    }
    manifest_path = out_dir / "manifest_multiimage.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("=== done ===")
    print(f"manifest written to {manifest_path}")
    print(f"  {len(manifest_entries)} images, 1 shared LoRA per condition per seed (not {len(manifest_entries)} separate ones)")


if __name__ == "__main__":
    main()
