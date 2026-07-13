"""Builds the two kohya_ss-style dataset folders + dataset_config TOML files
needed for the LoRA-training validation experiment: does style_cloak.py's
cloaking mechanism actually degrade LoRA training, measured against a real
LoRA run instead of the VGG19 proxy metric evaluate.py/robustness_test.py
use.

Two conditions, everything else held identical so cloak-vs-not is the only
variable:
    baseline: dataset/{repeats}_{trigger}/  <- the real starry_night.jpg
    cloaked:  dataset_cloaked/{repeats}_{trigger}/  <- style_cloak.cloak()
              output (L3_ANTI_TRAIN, the project's strongest preset)

Mirrors the folder-naming convention and dataset_config.toml shape already
proven in this project's own prior real LoRA run
(C:\\Users\\mello\\Desktop\\Develop\\LoRA\\train_config.toml on the GPU PC)
-- num_repeats/class_tokens auto-caption every image in a subset with just
the trigger word, no manual .txt captioning needed for a single-concept
style-overfit test like this one.

Runs on the *ml-engine* venv (needs torch+Pillow for cloak(), nothing
GPU-training-specific) -- CPU is fine here, this doesn't train anything.
"""

import argparse
import shutil
import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from style_cloak import cloak  # noqa: E402

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
    parser.add_argument("--style-image", default=str(ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg"))
    parser.add_argument(
        "--cloak-style-target",
        default=str(ML_ENGINE_DIR / "out" / "real" / "great_wave.jpg"),
        help="the *unrelated* style style_cloak.py pushes toward when producing the cloaked condition's image",
    )
    parser.add_argument("--trigger", default="starrynighttest")
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512, help="training resolution -- also the size cloak() runs at, so the cloak sees exactly what training sees")
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"])
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    style_image = Path(args.style_image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== preparing baseline (uncloaked) condition ===")
    baseline_toml = build_condition(
        out_dir / "dataset", style_image, args.trigger, args.num_repeats, args.resolution
    )
    print(f"  wrote {baseline_toml}")

    print(f"=== preparing cloaked condition ({args.preset}, size={args.resolution}) ===")
    # NOTE: ml-engine/README.md's presets/epsilon numbers were only validated
    # at size=256; this cloaks at the actual training resolution (512)
    # instead, which is outside that previously-validated envelope on
    # purpose -- a real LoRA trainer sees 512px input, so that's what this
    # experiment needs to test against, not the validated-but-irrelevant size.
    cloaked_image_path = out_dir / "cloaked_starry_night.png"
    cloak(
        original_path=str(style_image),
        style_target_path=args.cloak_style_target,
        output_path=str(cloaked_image_path),
        preset_name=args.preset,
        size=args.resolution,
        eot=False,
    )
    cloaked_toml = build_condition(
        out_dir / "dataset_cloaked", cloaked_image_path, args.trigger, args.num_repeats, args.resolution
    )
    print(f"  wrote {cloaked_toml}")

    print()
    print("=== done ===")
    print(f"baseline dataset_config: {baseline_toml}")
    print(f"cloaked  dataset_config: {cloaked_toml}")
    print(f"trigger word (use in generation prompts too): {args.trigger}")


if __name__ == "__main__":
    main()
