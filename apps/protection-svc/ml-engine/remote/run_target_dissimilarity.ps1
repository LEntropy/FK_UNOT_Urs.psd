# Dedicated confirmation experiment: does cloak-target dissimilarity
# (Gram-matrix similarity to the original, measured before cloaking)
# really predict real LoRA-training effect, isolated to ONE base image
# (starry_night) across 4 controlled target-similarity levels? See
# prepare_target_dissimilarity.py's docstring for the full design.
#
# Baseline (uncloaked starry_night) LoRA weights already exist from the
# main experiment (lora_starry_night_{1,2,3}_baseline/) -- reused
# directly by score_target_dissimilarity.py, not retrained here. Only the
# 4 new cloaked-toward-a-controlled-target conditions get trained.

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

$SEEDS = @(1, 2, 3)
$RUN_NAME = "v1"

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

Write-Host "=== 1/3: cloaking starry_night toward 4 controlled-similarity targets ==="
& $ML_VENV_PY "$EXP_DIR\prepare_target_dissimilarity.py"
if ($LASTEXITCODE -ne 0) { throw "prepare_target_dissimilarity.py failed" }

$manifest = Get-Content "$OUT_DIR\target_dissimilarity_manifest.json" | ConvertFrom-Json

function Train-Cloaked {
    param([string]$RunTag, [string]$DatasetConfig, [int]$Seed)
    Write-Host "=== training: $RunTag (seed $Seed) ==="
    $argList = @(
        "launch", "--num_cpu_threads_per_process", "1", $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", "$OUT_DIR\lora_${RunTag}_cloaked",
        "--output_name", "cloaked_$RUN_NAME",
        "--logging_dir", $LOGS_DIR,
        "--save_model_as", "safetensors",
        "--network_module", "networks.lora",
        "--network_dim", "32", "--network_alpha", "16",
        "--optimizer_type", "AdamW8bit",
        "--learning_rate", "5e-5", "--unet_lr", "5e-5",
        "--lr_scheduler", "cosine_with_restarts", "--lr_warmup_steps", "20",
        "--max_train_epochs", "10", "--save_every_n_epochs", "10",
        "--mixed_precision", "bf16", "--sdpa", "--gradient_checkpointing",
        "--cache_latents", "--network_train_unet_only",
        "--seed", "$Seed", "--max_data_loader_n_workers", "2"
    )
    New-Item -ItemType Directory -Force -Path "$OUT_DIR\lora_${RunTag}_cloaked" | Out-Null

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ACCEL @argList *>&1 | Tee-Object -FilePath "$LOGS_DIR\train_${RunTag}_cloaked.log"
    $ErrorActionPreference = $prevEAP

    if ($LASTEXITCODE -ne 0) { throw "training '$RunTag' failed (exit $LASTEXITCODE) -- see $LOGS_DIR\train_${RunTag}_cloaked.log" }
}

$total = $manifest.Count * $SEEDS.Count
$i = 0
foreach ($entry in $manifest) {
    $name = $entry.target_name
    foreach ($seed in $SEEDS) {
        $i += 1
        Write-Host "=== 2/3: [$i/$total] starry_night vs $name, seed $seed ==="
        Train-Cloaked -RunTag "starry_night_vs_${name}" -DatasetConfig $entry.dataset_config -Seed $seed
    }
}

Write-Host "=== 3/3: scoring against reused baseline + generating report ==="
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$seedsArg = ($SEEDS -join ",")
& $KOHYA_PY "$EXP_DIR\score_target_dissimilarity.py" `
    --checkpoint $CHECKPOINT `
    --manifest "$OUT_DIR\target_dissimilarity_manifest.json" `
    --lora-root $OUT_DIR `
    --seeds $seedsArg `
    --true-image "$ML_ENGINE\out\real\starry_night.jpg" `
    --out-dir "$OUT_DIR\generated_dissimilarity" `
    *>&1 | Tee-Object -FilePath "$OUT_DIR\report_dissimilarity.txt"
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) { throw "score_target_dissimilarity.py failed -- see $OUT_DIR\report_dissimilarity.txt" }

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report_dissimilarity.txt"
