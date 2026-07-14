"""SDXL confirmation: does the LoRA-degradation effect (established on
SD1.5 at n=30) reproduce on an SDXL-family checkpoint? Real-world
style-LoRA theft increasingly happens on SDXL/Illustrious-family
checkpoints, not SD1.5, so this matters more for the project's actual
threat model than the SD1.5 numbers alone.

Checkpoint: Illustrious-XL-v0.1.safetensors (confirmed genuine SDXL
architecture via safetensors key inspection -- has
`conditioner.embedders.*` keys). The other candidate checkpoint on this
GPU PC, hosekiLustrousmixAnima_animaV10.safetensors, turned out to be a
DiT/transformer-based architecture (`model.diffusion_model.blocks.N.
adaln_modulation_*` keys, not a UNet) -- incompatible with kohya_ss's
sd-scripts (train_network.py/sdxl_train_network.py are UNet-specific) and
out of scope here; a real test of that checkpoint would need a different
training framework entirely.

Small subset (2 images, not all 10) to keep SDXL's heavier per-step cost
tractable: great_wave and starry_night, same cross-cloak pairing as the
main SD1.5 experiment, cloaked at 1024px (SDXL's native training
resolution, vs SD1.5's 512px) -- this project's presets were also
separately re-validated at 1024x1024 (see README's "Re-validated at
1024x1024" section), so this isn't running outside any previously-checked
envelope the way the LoRA experiment's 512px choice was for SD1.5.
"""

import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from style_cloak import cloak  # noqa: E402
from prepare_dataset import IMAGE_CONFIGS  # noqa: E402

SUBSET = ["great_wave", "starry_night"]
RESOLUTION = 1024
NUM_REPEATS = 20
PRESET = "L3_ANTI_TRAIN"

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
    import shutil

    shutil.copy(image_path, concept_dir / image_path.name)

    toml_path = out_root.parent / f"dataset_config_{out_root.name}.toml"
    toml_path.write_text(
        TOML_TEMPLATE.format(resolution=resolution, image_dir=str(concept_dir), trigger=trigger, num_repeats=num_repeats)
    )
    return toml_path


def main() -> None:
    out_dir = Path(__file__).parent / "out" / "sdxl"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs_by_name = {c["name"]: c for c in IMAGE_CONFIGS}
    manifest = []

    for name in SUBSET:
        cfg = configs_by_name[name]
        print(f"=== [{name}] preparing baseline (uncloaked, 1024px) ===")
        baseline_toml = build_condition(out_dir / "dataset" / name, cfg["image"], cfg["trigger"], NUM_REPEATS, RESOLUTION)

        print(f"=== [{name}] cloaking at {PRESET}, size={RESOLUTION} ===")
        cloaked_path = out_dir / f"cloaked_sdxl_{name}.png"
        cloak(
            original_path=str(cfg["image"]),
            style_target_path=str(cfg["cloak_target"]),
            output_path=str(cloaked_path),
            preset_name=PRESET,
            size=RESOLUTION,
            eot=False,
        )
        cloaked_toml = build_condition(out_dir / "dataset_cloaked" / name, cloaked_path, cfg["trigger"], NUM_REPEATS, RESOLUTION)

        manifest.append(
            {
                "name": name,
                "trigger": cfg["trigger"],
                "prompt_suffix": cfg["prompt_suffix"],
                "true_image": str(cfg["image"]),
                "baseline_dataset_config": str(baseline_toml),
                "cloaked_dataset_config": str(cloaked_toml),
            }
        )
        print(f"  wrote {baseline_toml} and {cloaked_toml}")

    import json

    manifest_path = out_dir / "sdxl_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print()
    print(f"=== done, manifest: {manifest_path} ===")


if __name__ == "__main__":
    main()
