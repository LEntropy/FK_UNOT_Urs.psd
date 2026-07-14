"""Preset-scaling confirmation: does the real LoRA-degradation effect
scale with epsilon the way the VGG19-space metrics do (ml-engine/README.md's
"Measured results" section shows L1 < L2 < L3 style-drift, monotonically)?
Reuses 4 images spanning the effect range already found in the main
n=30 experiment -- great_wave (strong, +0.0289), starry_night (moderate,
+0.0198), night_watch (strong, +0.0248), mona_lisa (weak/negative,
-0.0024) -- and their SAME cloak_target pairing from prepare_dataset.py's
IMAGE_CONFIGS (target choice isn't the variable here, preset strength is),
just cloaked at L2_PORTFOLIO instead of L3_ANTI_TRAIN.

Baseline (uncloaked) LoRA weights for all 4 images already exist from the
main experiment and are reused directly (baseline doesn't depend on
preset at all -- no cloaking happens in that condition).
"""

import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from style_cloak import cloak  # noqa: E402
from prepare_dataset import IMAGE_CONFIGS  # noqa: E402

SUBSET = ["great_wave", "starry_night", "night_watch", "mona_lisa"]
RESOLUTION = 512
NUM_REPEATS = 20
PRESET = "L2_PORTFOLIO"

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


def main() -> None:
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs_by_name = {c["name"]: c for c in IMAGE_CONFIGS}
    manifest = []

    for name in SUBSET:
        cfg = configs_by_name[name]
        print(f"=== [{name}] cloaking at {PRESET} (was L3_ANTI_TRAIN in the main experiment) ===")

        cloaked_path = out_dir / f"cloaked_l2_{name}.png"
        cloak(
            original_path=str(cfg["image"]),
            style_target_path=str(cfg["cloak_target"]),
            output_path=str(cloaked_path),
            preset_name=PRESET,
            size=RESOLUTION,
            eot=False,
        )

        concept_dir = out_dir / f"dataset_l2_{name}" / f"{NUM_REPEATS}_{cfg['trigger']}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(cloaked_path, concept_dir / cloaked_path.name)

        toml_path = out_dir / f"dataset_config_l2_{name}.toml"
        toml_path.write_text(
            TOML_TEMPLATE.format(
                resolution=RESOLUTION, image_dir=str(concept_dir), trigger=cfg["trigger"], num_repeats=NUM_REPEATS
            )
        )

        manifest.append(
            {
                "name": name,
                "trigger": cfg["trigger"],
                "prompt_suffix": cfg["prompt_suffix"],
                "true_image": str(cfg["image"]),
                "dataset_config": str(toml_path),
            }
        )
        print(f"  wrote {toml_path}")

    import json

    manifest_path = out_dir / "l2_preset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print()
    print(f"=== done, manifest: {manifest_path} ===")


if __name__ == "__main__":
    main()
