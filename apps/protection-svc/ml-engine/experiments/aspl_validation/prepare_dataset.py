"""Builds kohya_ss-style dataset folders + dataset_config TOML files for the
full Anti-DreamBooth (ASPL) attack's real LoRA-validation experiment --
same 5 real paintings/triggers/prompts as experiments/diffusion_attack_validation/'s
own prepare_dataset.py (which itself reused experiments/lora_validation/'s set),
so this is a direct fourth data point against:
  style_cloak (VGG19 Gram-matrix):        +0.0113  (n=10, CI includes zero)
  concept_misalign (CLIP embedding):  true -0.0058 / decoy -0.0044 (n=15, CI incl. zero)
  diffusion_attack (frozen-checkpoint denoising loss): +0.0071  (n=10, CI includes zero)

Must run under kohya_ss's venv (diffusers + peft + CUDA) -- aspl_attack.py
trains a real surrogate LoRA every outer iteration.
"""

import argparse
import json
import sys
from pathlib import Path

ML_ENGINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_ENGINE_DIR / "src"))

from aspl_attack import aspl_attack  # noqa: E402

CHECKPOINT = r"C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors"

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

IMAGE_CONFIGS = [
    {
        "name": "starry_night",
        "image": ML_ENGINE_DIR / "out" / "real" / "starry_night.jpg",
        "trigger": "starrynightaspltest",
        "prompt_suffix": "oil painting, landscape, night sky",
    },
    {
        "name": "great_wave",
        "image": ML_ENGINE_DIR / "out" / "real" / "great_wave.jpg",
        "trigger": "greatwaveaspltest",
        "prompt_suffix": "woodblock print, ocean wave, landscape",
    },
    {
        "name": "mona_lisa",
        "image": ML_ENGINE_DIR / "out" / "real" / "mona_lisa.jpg",
        "trigger": "monalisaaspltest",
        "prompt_suffix": "oil painting, portrait of a woman",
    },
    {
        "name": "the_scream",
        "image": ML_ENGINE_DIR / "out" / "real" / "the_scream.jpg",
        "trigger": "screamaspltest",
        "prompt_suffix": "expressionist painting, portrait, screaming figure",
    },
    {
        "name": "composition_vii",
        "image": ML_ENGINE_DIR / "out" / "real" / "composition_vii.jpg",
        "trigger": "compositionviiaspltest",
        "prompt_suffix": "abstract painting, geometric shapes, bold colors",
    },
]


def build_condition(out_root: Path, image_path: Path, trigger: str, num_repeats: int, resolution: int) -> Path:
    import shutil

    concept_dir = out_root / f"{num_repeats}_{trigger}"
    concept_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, concept_dir / image_path.name)

    toml_path = out_root.parent / f"dataset_config_{out_root.name}.toml"
    toml_path.write_text(
        TOML_TEMPLATE.format(resolution=resolution, image_dir=str(concept_dir), trigger=trigger, num_repeats=num_repeats)
    )
    return toml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--preset", default="L3_ANTI_TRAIN", choices=["L1_PREVIEW", "L2_PORTFOLIO", "L3_ANTI_TRAIN"])
    parser.add_argument("--seed", type=int, default=0, help="seed for the attack's own noise/timestep sampling, independent of the LoRA training seed")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for cfg in IMAGE_CONFIGS:
        name = cfg["name"]
        print(f"=== [{name}] preparing baseline (unattacked) condition ===")
        baseline_toml = build_condition(
            out_dir / "dataset" / name, cfg["image"], cfg["trigger"], args.num_repeats, args.resolution
        )
        print(f"  wrote {baseline_toml}")

        prompt = f"{cfg['trigger']}, {cfg['prompt_suffix']}"
        print(f"=== [{name}] ASPL-attacking ({args.preset}, size={args.resolution}) prompt={prompt!r} ===")
        attacked_image_path = out_dir / f"attacked_{name}.png"
        aspl_attack(
            original_path=str(cfg["image"]),
            checkpoint_path=CHECKPOINT,
            prompt=prompt,
            output_path=str(attacked_image_path),
            preset_name=args.preset,
            size=args.resolution,
            seed=args.seed,
        )
        attacked_toml = build_condition(
            out_dir / "dataset_attacked" / name, attacked_image_path, cfg["trigger"], args.num_repeats, args.resolution
        )
        print(f"  wrote {attacked_toml}")

        manifest.append(
            {
                "name": name,
                "trigger": cfg["trigger"],
                "prompt_suffix": cfg["prompt_suffix"],
                "true_image": str(cfg["image"]),
                "baseline_dataset_config": str(baseline_toml),
                "attacked_dataset_config": str(attacked_toml),
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
