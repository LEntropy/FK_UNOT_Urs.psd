# Expands the L2_PORTFOLIO preset-scaling validation from n=4 to n=10
# images (all 10 from the main L3_ANTI_TRAIN n=30 experiment), for direct
# n=10-vs-n=10 parity instead of the original under-powered n=4 subset --
# see ml-engine/README.md's "Preset scaling" section for why n=4 was flagged
# as noisy.
#
# Mirrors run_lora_validation.ps1's structure but:
#   - only trains the "cloaked" (L2_PORTFOLIO) condition -- baseline LoRAs
#     for all 10 images already exist from the main experiment and are
#     reused directly (baseline doesn't depend on preset).
#   - skips training for the 4 images already done in the original n=4 run
#     (great_wave, starry_night, night_watch, mona_lisa) -- only the 6 new
#     images (the_scream, composition_vii, water_lilies, girl_pearl_earring,
#     birth_of_venus, the_kiss) x 3 seeds = 18 new training runs happen here.
#
# Run this ON the GPU PC (same convention as run_lora_validation.ps1).

$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "0"
$env:PYTHONIOENCODING = "utf-8:replace"
$env:PYTHONUNBUFFERED = "1"

$ML_ENGINE   = "C:\dontai-ml-engine"
$ML_VENV_PY  = "$ML_ENGINE\.venv\Scripts\python.exe"

$KOHYA       = "C:\Users\mello\Desktop\Develop\kohya_ss"
$KOHYA_PY    = "$KOHYA\venv\Scripts\python.exe"
$ACCEL       = "$KOHYA\venv\Scripts\accelerate.exe"
$TRAIN_SCRIPT = "$KOHYA\sd-scripts\train_network.py"

$CHECKPOINT  = "C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors"

$EXP_DIR     = "$ML_ENGINE\experiments\lora_validation"
$OUT_DIR     = "$EXP_DIR\out"
$LOGS_DIR    = "$OUT_DIR\logs"

$SEEDS       = @(1, 2, 3)
$RUN_NAME    = "v1"

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

Write-Host "=== 1/3: preparing L2_PORTFOLIO datasets for all 10 images (skips re-cloaking the original 4) ==="
& $ML_VENV_PY "$EXP_DIR\prepare_l2_preset.py"
if ($LASTEXITCODE -ne 0) { throw "prepare_l2_preset.py failed" }

$manifest = Get-Content "$OUT_DIR\l2_preset_manifest.json" | ConvertFrom-Json

function Train-Cloaked {
    param(
        [string]$Name,
        [string]$DatasetConfig,
        [int]$Seed
    )
    $outputDir = "$OUT_DIR\lora_l2_${Name}_${Seed}_cloaked"
    $weightsPath = "$outputDir\cloaked_v1.safetensors"
    if (Test-Path $weightsPath) {
        Write-Host "=== [$Name / seed $Seed] already trained, skipping ==="
        return
    }

    Write-Host "=== training: $Name / L2_PORTFOLIO cloaked (seed $Seed) ==="
    $argList = @(
        "launch",
        "--num_cpu_threads_per_process", "1",
        $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", $outputDir,
        "--output_name", "cloaked_$RUN_NAME",
        "--logging_dir", $LOGS_DIR,
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora",
        "--network_dim", "32",
        "--network_alpha", "16",
        "--optimizer_type", "AdamW8bit",
        "--learning_rate", "5e-5",
        "--unet_lr", "5e-5",
        "--lr_scheduler", "cosine_with_restarts",
        "--lr_warmup_steps", "20",
        "--max_train_epochs", "10",
        "--save_every_n_epochs", "10",
        "--mixed_precision", "bf16",
        "--sdpa",
        "--gradient_checkpointing",
        "--cache_latents",
        "--network_train_unet_only",
        "--seed", "$Seed",
        "--max_data_loader_n_workers", "2"
    )
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ACCEL @argList *>&1 | Tee-Object -FilePath "$LOGS_DIR\train_l2_${Name}_${Seed}_cloaked.log"
    $ErrorActionPreference = $prevEAP

    if ($LASTEXITCODE -ne 0) { throw "training 'l2/$Name/$Seed' failed (exit $LASTEXITCODE) -- see $LOGS_DIR\train_l2_${Name}_${Seed}_cloaked.log" }
}

$totalRuns = $manifest.Count * $SEEDS.Count
$runNum = 0
foreach ($entry in $manifest) {
    $name = $entry.name
    foreach ($seed in $SEEDS) {
        $runNum += 1
        Write-Host "=== 2/3: [$runNum/$totalRuns] $name / seed $seed ==="
        Train-Cloaked -Name $name -DatasetConfig $entry.dataset_config -Seed $seed
    }
}

Write-Host "=== 3/3: generating samples + scoring (baseline recomputed fresh + L2 cloaked, all 10 images) ==="
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$seedsArg = ($SEEDS -join ",")
& $KOHYA_PY "$EXP_DIR\score_l2_preset.py" `
    --checkpoint $CHECKPOINT `
    --manifest "$OUT_DIR\l2_preset_manifest.json" `
    --lora-root $OUT_DIR `
    --seeds $seedsArg `
    --out-dir "$OUT_DIR\generated_l2" `
    *>&1 | Tee-Object -FilePath "$OUT_DIR\report_l2_preset.txt"
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) { throw "score_l2_preset.py failed (exit $LASTEXITCODE) -- see $OUT_DIR\report_l2_preset.txt" }

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report_l2_preset.txt"
