"""n=30 multi-architecture (SD1.5+SDXL) ensemble ASPL validation, run
per-pod against a subset of the 10-image manifest (image assignment passed
via --images). Orchestrates: attack (MULTIARCH_FULL preset, once per image)
-> train 4 LoRAs per (image, seed) in SEEDS -> writes a manifest for the
scoring pass (run_multiarch_n30_score.py) to consume.

Must run under kohya_ss's venv (diffusers + peft + accelerate + bitsandbytes).
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/dontai-protection-svc/ml-engine/src")
from ensemble_attack_multiarch import multiarch_ensemble_attack  # noqa: E402

SEEDS = [1, 2, 3]
SD15_CKPT = "/workspace/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
SDXL_CKPT = "/workspace/checkpoints/Illustrious-XL-v0.1.safetensors"
TRUE_DIR = "/workspace/dontai-protection-svc/ml-engine/out/real"
OUT_DIR = Path("/workspace/n30_out")

TOML_TEMPLATE = """[general]
enable_bucket = false

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{image_dir}'
  class_tokens = '{trigger}'
  num_repeats = 20
"""


def build_dataset(image_path: str, trigger: str, resolution: int, cond_dir: Path) -> Path:
    concept_dir = cond_dir / f"20_{trigger}"
    concept_dir.mkdir(parents=True, exist_ok=True)
    dst = concept_dir / Path(image_path).name
    shutil.copy(image_path, dst)
    toml_path = cond_dir.parent / f"dataset_config_{cond_dir.name}_{resolution}.toml"
    toml_path.write_text(TOML_TEMPLATE.format(resolution=resolution, image_dir=str(concept_dir), trigger=trigger))
    return toml_path


def train(script: str, ckpt: str, dataset_config: Path, output_dir: Path, output_name: str, seed: int, log_path: Path) -> None:
    cmd = [
        "/workspace/kohya_ss/venv/bin/accelerate", "launch", "--num_cpu_threads_per_process", "1",
        f"/workspace/kohya_ss/sd-scripts/{script}",
        "--pretrained_model_name_or_path", ckpt,
        "--dataset_config", str(dataset_config),
        "--output_dir", str(output_dir),
        "--output_name", output_name,
        "--logging_dir", str(OUT_DIR / "logs"),
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora",
        "--network_dim", "32", "--network_alpha", "16",
        "--optimizer_type", "AdamW8bit",
        "--learning_rate", "5e-5", "--unet_lr", "5e-5",
        "--lr_scheduler", "cosine_with_restarts", "--lr_warmup_steps", "20",
        "--max_train_epochs", "10", "--save_every_n_epochs", "10",
        "--mixed_precision", "bf16", "--sdpa", "--gradient_checkpointing",
        "--cache_latents", "--network_train_unet_only",
        "--seed", str(seed), "--max_data_loader_n_workers", "2",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"training failed (exit {result.returncode}) -- see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="comma-separated image names (no extension) assigned to this pod")
    parser.add_argument("--pod-tag", required=True, help="unique tag for this pod's manifest/log output files")
    args = parser.parse_args()

    image_names = [n.strip() for n in args.images.split(",") if n.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for idx, name in enumerate(image_names, start=1):
        image_path = f"{TRUE_DIR}/{name}.png"
        trigger = f"{name.replace('_', '')}n30test"
        prompt = f"{trigger}, {name.replace('_', ' ')}, anime illustration"

        t0 = time.time()
        attacked_path = OUT_DIR / f"attacked_{name}.png"
        if attacked_path.exists():
            print(f"=== [{idx}/{len(image_names)}] [{name}] attack already done, skipping (resume) ===")
        else:
            print(f"=== [{idx}/{len(image_names)}] [{name}] attacking (MULTIARCH_FULL) ===")
            multiarch_ensemble_attack(
                original_path=image_path,
                sd15_checkpoint=SD15_CKPT,
                sdxl_checkpoint=SDXL_CKPT,
                prompt=prompt,
                output_path=str(attacked_path),
                preset_name="MULTIARCH_FULL",
                seed=0,
            )
            print(f"  attack done in {time.time() - t0:.1f}s")

        baseline_sd15_toml = build_dataset(image_path, trigger, 512, OUT_DIR / "dataset" / f"{name}_sd15")
        baseline_sdxl_toml = build_dataset(image_path, trigger, 1024, OUT_DIR / "dataset" / f"{name}_sdxl")
        attacked_sd15_toml = build_dataset(str(attacked_path), trigger, 512, OUT_DIR / "dataset_attacked" / f"{name}_sd15")
        attacked_sdxl_toml = build_dataset(str(attacked_path), trigger, 1024, OUT_DIR / "dataset_attacked" / f"{name}_sdxl")

        for seed in SEEDS:
            t1 = time.time()
            print(f"=== [{name}/{seed}] training 4 LoRAs ===")
            runs = [
                ("train_network.py", SD15_CKPT, baseline_sd15_toml, "sd15", "baseline"),
                ("train_network.py", SD15_CKPT, attacked_sd15_toml, "sd15", "attacked"),
                ("sdxl_train_network.py", SDXL_CKPT, baseline_sdxl_toml, "sdxl", "baseline"),
                ("sdxl_train_network.py", SDXL_CKPT, attacked_sdxl_toml, "sdxl", "attacked"),
            ]
            for script, ckpt, dcfg, arch, cond in runs:
                out_dir = OUT_DIR / f"lora_{name}_{seed}_{arch}_{cond}"
                log_path = OUT_DIR / "logs" / f"train_{name}_{seed}_{arch}_{cond}.log"
                out_file = out_dir / f"{cond}_v1.safetensors"
                if out_file.exists():
                    print(f"  [{name}/{seed}/{arch}/{cond}] already trained, skipping (resume)")
                    continue
                train(script, ckpt, dcfg, out_dir, f"{cond}_v1", seed, log_path)
            print(f"  seed {seed} training done in {time.time() - t1:.1f}s")

        manifest.append({"name": name, "trigger": trigger, "prompt": prompt, "true_image": image_path})

    manifest_path = OUT_DIR / f"manifest_{args.pod_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"=== DONE, manifest written to {manifest_path} ===")


if __name__ == "__main__":
    main()
