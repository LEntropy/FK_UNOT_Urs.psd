"""Dedicated confirmation experiment for the target-dissimilarity finding
(ml-engine/README.md's "Follow-up" section, Pearson r=-0.929 from reading
5 incidental cross-cloaked pairs). Isolates cloak-target choice as the
only variable: one fixed base image (starry_night.jpg, already
well-characterized from the main n=30 experiment), cloaked toward 4
targets spanning a controlled Gram-matrix similarity range (measured via
pairwise_similarity.py against all other experiment images):

    0.6350  the_scream            (most dissimilar)
    0.6855  girl_pearl_earring
    0.7594  composition_vii
    0.8198  water_lilies          (most similar)

(great_wave at 0.7445 is the 5th point on this spectrum, already measured
in the main experiment -- starry_night's mean CLIP delta there was
+0.0198 at similarity 0.7445 -- included in the final correlation, not
retrained here.)

Baseline (uncloaked starry_night) LoRA weights already exist on the GPU
PC from the main experiment (lora_starry_night_{1,2,3}_baseline/) and
don't depend on the cloak target at all -- reused directly, not
retrained, by run_target_dissimilarity.ps1 and score_target_dissimilarity.py.
"""

import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from style_cloak import cloak  # noqa: E402

BASE_IMAGE = ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg"
TRIGGER = "starrynighttest"  # same trigger as the main experiment's starry_night entry
RESOLUTION = 512
NUM_REPEATS = 20

TARGETS = [
    {"name": "the_scream", "path": ML_ENGINE_DIR / "out" / "real" / "the_scream.jpg", "similarity": 0.6350},
    {"name": "girl_pearl_earring", "path": ML_ENGINE_DIR / "out" / "real" / "girl_pearl_earring.jpg", "similarity": 0.6855},
    {"name": "composition_vii", "path": ML_ENGINE_DIR / "out" / "real" / "composition_vii.jpg", "similarity": 0.7594},
    {"name": "water_lilies", "path": ML_ENGINE_DIR / "out" / "real" / "water_lilies.jpg", "similarity": 0.8198},
]

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

    manifest = []
    for target in TARGETS:
        name = target["name"]
        print(f"=== cloaking starry_night toward {name} (similarity={target['similarity']}) ===")

        cloaked_path = out_dir / f"cloaked_starry_night_vs_{name}.png"
        cloak(
            original_path=str(BASE_IMAGE),
            style_target_path=str(target["path"]),
            output_path=str(cloaked_path),
            preset_name="L3_ANTI_TRAIN",
            size=RESOLUTION,
            eot=False,
        )

        concept_dir = out_dir / f"dataset_vs_{name}" / f"{NUM_REPEATS}_{TRIGGER}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(cloaked_path, concept_dir / cloaked_path.name)

        toml_path = out_dir / f"dataset_config_vs_{name}.toml"
        toml_path.write_text(
            TOML_TEMPLATE.format(
                resolution=RESOLUTION, image_dir=str(concept_dir), trigger=TRIGGER, num_repeats=NUM_REPEATS
            )
        )

        manifest.append(
            {
                "target_name": name,
                "similarity": target["similarity"],
                "dataset_config": str(toml_path),
            }
        )
        print(f"  wrote {toml_path}")

    import json

    manifest_path = out_dir / "target_dissimilarity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print()
    print(f"=== done, manifest: {manifest_path} ===")


if __name__ == "__main__":
    main()
