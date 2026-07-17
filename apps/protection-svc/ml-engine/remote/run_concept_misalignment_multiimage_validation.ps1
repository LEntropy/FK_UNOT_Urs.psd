# Multi-image follow-up to run_concept_misalignment_validation.ps1's
# single-image result (WEAK/FAIL -- see PHASE4_SCOPING.md §1's "Update"
# note and experiments/concept_misalignment_validation/README.md).
#
# Tests whether that null result was about the misalignment *mechanism*
# or about the single-image LoRA setup diluting it: instead of one LoRA
# per (image, condition) that only ever sees that one image, this trains
# ONE LoRA per condition per seed that sees all 5 images/triggers jointly
# (kohya's standard multi-subset config) -- closer to a real scraper's
# actual training set (many images, many captions) than an isolated
# single-image LoRA that's mostly just memorizing one pair.
#
# Only 2 LoRAs per seed (baseline_multi, misaligned_multi) instead of 10
# (2 per image x 5 images) -- 3 seeds = 6 total trainings, not 30.
#
# Run this ON the GPU PC. Assumes apps/protection-svc/ml-engine has
# already been synced here (same convention as run_concept_misalignment_
# validation.ps1 and ml-engine/remote/run_remote.sh).

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

$EXP_DIR     = "$ML_ENGINE\experiments\concept_misalignment_validation"
$OUT_DIR     = "$EXP_DIR\out_multiimage"
$LOGS_DIR    = "$OUT_DIR\logs"

$SEEDS       = @(1, 2, 3)
$RUN_NAME    = "v1"

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

& $KOHYA_PY -c "import peft" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== installing peft (needed for diffusers LoRA loading) ==="
    & $KOHYA_PY -m pip install peft -q
}

Push-Location "$ML_ENGINE\src"
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $ML_VENV_PY -c "from model import ConceptFeatureExtractor; import torch; ConceptFeatureExtractor(torch.device('cpu'))" 2>&1 | Out-Null
$clipCheckExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Pop-Location
if ($clipCheckExit -ne 0) {
    throw "ConceptFeatureExtractor failed to load its CLIP checkpoint -- resolve before continuing."
}

Write-Host "=== 1/3: preparing joint multi-image datasets (misalign() runs 5x here, CPU-fine) ==="
& $ML_VENV_PY "$EXP_DIR\prepare_multiimage.py" --out-dir $OUT_DIR
if ($LASTEXITCODE -ne 0) { throw "prepare_multiimage.py failed" }

$manifest = Get-Content "$OUT_DIR\manifest_multiimage.json" | ConvertFrom-Json

function Train-Condition {
    param(
        [string]$Condition,
        [string]$DatasetConfig,
        [int]$Seed
    )
    $outputName = "${Condition}_$RUN_NAME"
    Write-Host "=== training: multi / $Condition (seed $Seed) ==="
    $argList = @(
        "launch",
        "--num_cpu_threads_per_process", "1",
        $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", "$OUT_DIR\lora_multi_${Seed}_${Condition}",
        "--output_name", $outputName,
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
    New-Item -ItemType Directory -Force -Path "$OUT_DIR\lora_multi_${Seed}_${Condition}" | Out-Null

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ACCEL @argList *>&1 | Tee-Object -FilePath "$LOGS_DIR\train_multi_${Seed}_${Condition}.log"
    $ErrorActionPreference = $prevEAP

    if ($LASTEXITCODE -ne 0) { throw "training 'multi/$Condition' (seed $Seed) failed (exit $LASTEXITCODE) -- see $LOGS_DIR\train_multi_${Seed}_${Condition}.log" }
}

$totalRuns = $SEEDS.Count * 2
$runNum = 0
foreach ($seed in $SEEDS) {
    $runNum += 1
    Write-Host "=== 2/3: [$runNum/$totalRuns] seed $seed baseline (joint, 5 images) ==="
    Train-Condition -Condition "baseline" -DatasetConfig $manifest.baseline_dataset_config -Seed $seed
    $runNum += 1
    Write-Host "=== 2/3: [$runNum/$totalRuns] seed $seed misaligned (joint, 5 images) ==="
    Train-Condition -Condition "misaligned" -DatasetConfig $manifest.misaligned_dataset_config -Seed $seed
}

Write-Host "=== 3/3: generating samples + scoring (CLIP similarity to true AND decoy concept, kohya venv) ==="
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$seedsArg = ($SEEDS -join ",")
& $KOHYA_PY "$EXP_DIR\generate_and_score_multiimage.py" `
    --checkpoint $CHECKPOINT `
    --manifest "$OUT_DIR\manifest_multiimage.json" `
    --lora-root $OUT_DIR `
    --seeds $seedsArg `
    --run-name $RUN_NAME `
    --out-dir "$OUT_DIR\generated" `
    *>&1 | Tee-Object -FilePath "$OUT_DIR\report_multiimage.txt"
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) { throw "generate_and_score_multiimage.py failed (exit $LASTEXITCODE) -- see $OUT_DIR\report_multiimage.txt" }

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report_multiimage.txt"
