# SDXL confirmation: does the LoRA-degradation effect reproduce on
# Illustrious-XL (genuine SDXL architecture)? See prepare_sdxl.py's
# docstring for the full design and why hosekiLustrousmixAnima was ruled
# out (DiT architecture, incompatible with sd-scripts).
#
# Mirrors this GPU PC's own proven SDXL LoRA config
# (C:\Users\mello\Desktop\Develop\LoRA\train_accel.ps1) rather than
# reinventing settings for an architecture this project hasn't trained
# LoRAs on before -- same network_dim/alpha/optimizer/scheduler, same
# cache_text_encoder_outputs flag (valid for SDXL, unlike SD1.x where it
# broke train_network.py in the main experiment).

$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "0"
$env:PYTHONIOENCODING = "utf-8:replace"
$env:PYTHONUNBUFFERED = "1"

$ML_ENGINE   = "C:\dontai-ml-engine"
$ML_VENV_PY  = "$ML_ENGINE\.venv\Scripts\python.exe"

$KOHYA       = "C:\Users\mello\Desktop\Develop\kohya_ss"
$KOHYA_PY    = "$KOHYA\venv\Scripts\python.exe"
$ACCEL       = "$KOHYA\venv\Scripts\accelerate.exe"
$TRAIN_SCRIPT = "$KOHYA\sd-scripts\sdxl_train_network.py"

$CHECKPOINT  = "C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\Illustrious-XL-v0.1.safetensors"

$EXP_DIR     = "$ML_ENGINE\experiments\lora_validation"
$OUT_DIR     = "$EXP_DIR\out\sdxl"
$LOGS_DIR    = "$OUT_DIR\logs"

$SEEDS = @(1, 2, 3)
$RUN_NAME = "v1"

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

Write-Host "=== 1/3: preparing SDXL datasets (1024px cloak + baseline) ==="
& $ML_VENV_PY "$EXP_DIR\prepare_sdxl.py"
if ($LASTEXITCODE -ne 0) { throw "prepare_sdxl.py failed" }

$manifest = Get-Content "$OUT_DIR\sdxl_manifest.json" | ConvertFrom-Json

function Train-Condition {
    param([string]$Name, [string]$Condition, [string]$DatasetConfig, [int]$Seed)
    $outputName = "${Condition}_$RUN_NAME"

    # Reuse LoRAs already trained in an earlier pass (great_wave/starry_night
    # at n=6) instead of retraining from scratch -- same compute-reuse
    # principle as run_target_dissimilarity.ps1/run_l2_preset.ps1.
    $expectedFile = "$OUT_DIR\lora_sdxl_${Name}_${Seed}_${Condition}\${outputName}.safetensors"
    if (Test-Path $expectedFile) {
        Write-Host "=== skipping sdxl $Name / $Condition (seed $Seed): already trained at $expectedFile ==="
        return
    }

    Write-Host "=== training: sdxl $Name / $Condition (seed $Seed) ==="
    $argList = @(
        "launch", "--num_cpu_threads_per_process", "1", $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", "$OUT_DIR\lora_sdxl_${Name}_${Seed}_${Condition}",
        "--output_name", $outputName,
        "--logging_dir", $LOGS_DIR,
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora",
        "--network_dim", "32", "--network_alpha", "16",
        "--optimizer_type", "AdamW8bit",
        "--learning_rate", "5e-5", "--unet_lr", "5e-5",
        "--lr_scheduler", "cosine_with_restarts", "--lr_warmup_steps", "20",
        "--max_train_epochs", "10", "--save_every_n_epochs", "10",
        "--mixed_precision", "bf16", "--sdpa", "--gradient_checkpointing",
        "--cache_latents", "--cache_latents_to_disk",
        "--network_train_unet_only",
        "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
        "--seed", "$Seed", "--max_data_loader_n_workers", "2"
    )
    New-Item -ItemType Directory -Force -Path "$OUT_DIR\lora_sdxl_${Name}_${Seed}_${Condition}" | Out-Null

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ACCEL @argList *>&1 | Tee-Object -FilePath "$LOGS_DIR\train_${Name}_${Seed}_${Condition}.log"
    $ErrorActionPreference = $prevEAP

    if ($LASTEXITCODE -ne 0) { throw "training sdxl $Name/$Condition seed $Seed failed (exit $LASTEXITCODE)" }
}

$total = $manifest.Count * $SEEDS.Count * 2
$i = 0
foreach ($entry in $manifest) {
    $name = $entry.name
    foreach ($seed in $SEEDS) {
        $i += 1
        Write-Host "=== 2/3: [$i/$total] sdxl $name baseline, seed $seed ==="
        Train-Condition -Name $name -Condition "baseline" -DatasetConfig $entry.baseline_dataset_config -Seed $seed
        $i += 1
        Write-Host "=== 2/3: [$i/$total] sdxl $name cloaked, seed $seed ==="
        Train-Condition -Name $name -Condition "cloaked" -DatasetConfig $entry.cloaked_dataset_config -Seed $seed
    }
}

Write-Host "=== 3/3: generating samples + scoring (SDXL pipeline, kohya venv) ==="
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$seedsArg = ($SEEDS -join ",")
& $KOHYA_PY "$EXP_DIR\score_sdxl.py" `
    --checkpoint $CHECKPOINT `
    --manifest "$OUT_DIR\sdxl_manifest.json" `
    --lora-root $OUT_DIR `
    --seeds $seedsArg `
    --run-name $RUN_NAME `
    --out-dir "$OUT_DIR\generated" `
    *>&1 | Tee-Object -FilePath "$OUT_DIR\report_sdxl.txt"
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) { throw "score_sdxl.py failed -- see $OUT_DIR\report_sdxl.txt" }

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report_sdxl.txt"
