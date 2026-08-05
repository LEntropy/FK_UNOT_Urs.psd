# Multi-surrogate ensemble attack's real LoRA-validation experiment --
# does attacking N=3 diverse LoRA surrogates jointly, at a relaxed
# (visibly non-invisible) epsilon=0.25 budget, transfer into a measurably
# worse LoRA when every single-surrogate/invisible-epsilon attempt so far
# didn't? See ensemble_attack.py's module doc for the mechanism and why
# these two specific variables (surrogate diversity, epsilon budget) were
# picked as the next thing to test.
#
# Same 5 images x 2 seeds x 2 conditions = 20 trainings as the prior
# validation runs, for a direct, comparable sixth data point against
# style_cloak (+0.0113), concept_misalign (~0), diffusion_attack (+0.0071),
# aspl_attack (+0.0050), and hybrid_attack (-0.0054) -- all five already
# validated and negative.
#
# Like diffusion_attack_validation/aspl_validation/hybrid_validation's step
# 1/3 (not lora_validation's), this experiment's prepare_dataset.py needs
# kohya_ss's venv (diffusers + peft + CUDA) for ensemble_attack.py's real
# multi-adapter surrogate training + VAE/UNet forward/backward.

$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "0"
$env:PYTHONIOENCODING = "utf-8:replace"
$env:PYTHONUNBUFFERED = "1"

$PROTECTION_SVC = "C:\dontai-protection-svc"
$ML_ENGINE   = "$PROTECTION_SVC\ml-engine"

$KOHYA        = "C:\Users\mello\Desktop\Develop\kohya_ss"
$KOHYA_PY     = "$KOHYA\venv\Scripts\python.exe"
$ACCEL        = "$KOHYA\venv\Scripts\accelerate.exe"
$TRAIN_SCRIPT = "$KOHYA\sd-scripts\train_network.py"

$CHECKPOINT  = "C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors"

$EXP_DIR     = "$ML_ENGINE\experiments\ensemble_validation"
$OUT_DIR     = "$EXP_DIR\out"
$LOGS_DIR    = "$OUT_DIR\logs"

$SEEDS       = @(1, 2)
$RUN_NAME    = "v1"

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

function Run-Step {
    param([string]$Label, [string]$LogName, [ScriptBlock]$Body)
    Write-Host "=== $Label ==="
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Body *>&1 | Tee-Object -FilePath "$LOGS_DIR\$LogName.log"
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE) -- see $LOGS_DIR\$LogName.log" }
}

& $KOHYA_PY -c "import peft" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== installing peft (needed for diffusers LoRA loading) ==="
    & $KOHYA_PY -m pip install peft -q
}

Run-Step "1/3: preparing datasets (ensemble-attacking all 5 images, kohya venv -- needs GPU+diffusers+peft)" "prepare" {
    & $KOHYA_PY "$EXP_DIR\prepare_dataset.py"
}

$manifest = Get-Content "$OUT_DIR\manifest.json" | ConvertFrom-Json

function Train-Condition {
    param([string]$RunTag, [string]$Condition, [string]$DatasetConfig, [int]$Seed)
    $outputName = "${Condition}_$RUN_NAME"
    $argList = @(
        "launch", "--num_cpu_threads_per_process", "1", $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", "$OUT_DIR\lora_${RunTag}_${Condition}",
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
        "--cache_latents", "--network_train_unet_only",
        "--seed", "$Seed", "--max_data_loader_n_workers", "2"
    )
    New-Item -ItemType Directory -Force -Path "$OUT_DIR\lora_${RunTag}_${Condition}" | Out-Null
    Run-Step "training: $RunTag / $Condition (seed $Seed)" "train_${RunTag}_${Condition}" { & $ACCEL @argList }
}

$totalRuns = $manifest.Count * $SEEDS.Count * 2
$runNum = 0
foreach ($entry in $manifest) {
    $name = $entry.name
    foreach ($seed in $SEEDS) {
        $runTag = "${name}_${seed}"
        $runNum += 1
        Write-Host "=== 2/3: [$runNum/$totalRuns] $runTag baseline ==="
        Train-Condition -RunTag $runTag -Condition "baseline" -DatasetConfig $entry.baseline_dataset_config -Seed $seed
        $runNum += 1
        Write-Host "=== 2/3: [$runNum/$totalRuns] $runTag attacked ==="
        Train-Condition -RunTag $runTag -Condition "attacked" -DatasetConfig $entry.attacked_dataset_config -Seed $seed
    }
}

$seedsArg = ($SEEDS -join ",")
Run-Step "3/3: generating samples + scoring (CLIP similarity, kohya venv)" "score" {
    & $KOHYA_PY "$EXP_DIR\generate_and_score.py" `
        --checkpoint $CHECKPOINT `
        --manifest "$OUT_DIR\manifest.json" `
        --lora-root $OUT_DIR `
        --seeds $seedsArg `
        --run-name $RUN_NAME `
        --out-dir "$OUT_DIR\generated"
}
Copy-Item "$LOGS_DIR\score.log" "$OUT_DIR\report.txt" -Force

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report.txt"
