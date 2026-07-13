"""Builds kohya_ss-style dataset folders + dataset_config TOML files for
the LoRA-training validation experiment (see ml-engine/README.md's "LoRA
validation experiment" section for the full writeup) -- does
style_cloak.py's cloaking mechanism actually degrade LoRA training,
measured against a real LoRA run instead of the VGG19 proxy metric
evaluate.py/robustness_test.py use.

Now covers multiple images (cross-cloaking each real painting toward the
*other* one) so the follow-up multi-seed run isn't resting on a single
image. Two conditions per image, everything else held identical so
cloak-vs-not is the only variable:
    baseline: dataset/{name}/{repeats}_{trigger}/  <- the real image
    cloaked:  dataset_cloaked/{name}/{repeats}_{trigger}/  <- style_cloak.cloak() output

Cloaking itself doesn't depend on the training seed (style_cloak.cloak()
takes no seed parameter -- it's a deterministic PGD optimization given the
same inputs), so this only needs to run once per image regardless of how
many training seeds the run script loops over afterward.

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

# Cross-cloaking: each image's cloak pushes toward a *different* image's
# style, matching the README's existing real-artwork cloak example
# (starry_night cloaked toward great_wave). Paired up as two stylistically
# distant pairs (post-impressionist <-> ukiyo-e, renaissance portrait <->
# expressionist) rather than one-fixed-target-for-all, so the sample isn't
# secretly testing only one cloak direction four times over.
# prompt_suffix matters: it must describe each painting's actual subject,
# not a one-size-fits-all guess. A first run used "oil painting, landscape"
# for every image, which actively fought portrait/figure subjects (Mona
# Lisa, The Scream) since the LoRA's trigger word alone (text encoder
# untrained -- network_train_unet_only) is weak signal compared to a
# strong subject word already in the prompt. Generation produced unrelated
# landscapes for both baseline AND cloaked conditions on those two images
# -- not a cloak effect, a broken prompt, and it made both conditions fail
# similarly (near-zero, uninformative delta). Fixed by matching the prompt
# to the actual subject per image.
IMAGE_CONFIGS = [
    {
        "name": "starry_night",
        "image": ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "great_wave.jpg",
        "trigger": "starrynighttest",
        "prompt_suffix": "oil painting, landscape, night sky",
    },
    {
        "name": "great_wave",
        "image": ML_ENGINE_DIR / "out" / "real" / "great_wave.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg",
        "trigger": "greatwavetest",
        "prompt_suffix": "woodblock print, ocean wave, landscape",
    },
    {
        "name": "mona_lisa",
        "image": ML_ENGINE_DIR / "out" / "real" / "mona_lisa.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "the_scream.jpg",
        "trigger": "monalisatest",
        "prompt_suffix": "oil painting, portrait of a woman",
    },
    {
        "name": "the_scream",
        "image": ML_ENGINE_DIR / "out" / "real" / "the_scream.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "mona_lisa.jpg",
        "trigger": "screamtest",
        "prompt_suffix": "expressionist painting, portrait, screaming figure",
    },
    {
        "name": "composition_vii",
        "image": ML_ENGINE_DIR / "out" / "real" / "composition_vii.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "water_lilies.jpg",
        "trigger": "compositionviitest",
        "prompt_suffix": "abstract painting, geometric shapes, bold colors",
    },
    {
        "name": "water_lilies",
        "image": ML_ENGINE_DIR / "out" / "real" / "water_lilies.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "composition_vii.jpg",
        "trigger": "waterliliestest",
        "prompt_suffix": "impressionist painting, water lily pond, landscape",
    },
    {
        "name": "girl_pearl_earring",
        "image": ML_ENGINE_DIR / "out" / "real" / "girl_pearl_earring.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "birth_of_venus.jpg",
        "trigger": "pearlearringtest",
        "prompt_suffix": "baroque painting, portrait of a girl, dark background",
    },
    {
        "name": "birth_of_venus",
        "image": ML_ENGINE_DIR / "out" / "real" / "birth_of_venus.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "girl_pearl_earring.jpg",
        "trigger": "venustest",
        "prompt_suffix": "renaissance painting, woman on a shell, mythological scene",
    },
    {
        "name": "night_watch",
        "image": ML_ENGINE_DIR / "out" / "real" / "night_watch.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "the_kiss.jpg",
        "trigger": "nightwatchtest",
        "prompt_suffix": "baroque painting, group portrait, dark dramatic lighting",
    },
    {
        "name": "the_kiss",
        "image": ML_ENGINE_DIR / "out" / "real" / "the_kiss.jpg",
        "cloak_target": ML_ENGINE_DIR / "out" / "real" / "night_watch.jpg",
        "trigger": "thekisstest",
        "prompt_suffix": "art nouveau painting, gold leaf, couple embracing",
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
    parser.add_argument("--resolution", type=int, default=512, help="training resolution -- also the size cloak() runs at, so the cloak sees exactly what training sees")
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"])
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for cfg in IMAGE_CONFIGS:
        name = cfg["name"]
        print(f"=== [{name}] preparing baseline (uncloaked) condition ===")
        baseline_toml = build_condition(
            out_dir / "dataset" / name, cfg["image"], cfg["trigger"], args.num_repeats, args.resolution
        )
        print(f"  wrote {baseline_toml}")

        print(f"=== [{name}] preparing cloaked condition ({args.preset}, size={args.resolution}) ===")
        # NOTE: ml-engine/README.md's presets/epsilon numbers were only
        # validated at size=256; this cloaks at the actual training
        # resolution (512) instead, on purpose -- a real LoRA trainer sees
        # 512px input, so that's what this experiment needs to test
        # against, not the validated-but-irrelevant size.
        cloaked_image_path = out_dir / f"cloaked_{name}.png"
        cloak(
            original_path=str(cfg["image"]),
            style_target_path=str(cfg["cloak_target"]),
            output_path=str(cloaked_image_path),
            preset_name=args.preset,
            size=args.resolution,
            eot=False,
        )
        cloaked_toml = build_condition(
            out_dir / "dataset_cloaked" / name, cloaked_image_path, cfg["trigger"], args.num_repeats, args.resolution
        )
        print(f"  wrote {cloaked_toml}")

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

    import json

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("=== done ===")
    print(f"manifest written to {manifest_path}")
    for entry in manifest:
        print(f"  [{entry['name']}] trigger={entry['trigger']}")


if __name__ == "__main__":
    main()
