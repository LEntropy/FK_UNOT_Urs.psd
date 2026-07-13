# Real LoRA-training validation experiment, orchestrated entirely on the
# GPU PC (this machine): does style_cloak.py's cloak actually degrade LoRA
# training, measured against a real train_network.py run instead of the
# VGG19 proxy metric evaluate.py/robustness_test.py use.
#
# Reuses this machine's existing, already-proven kohya_ss setup (same one
# used for the real renboerocat SDXL LoRA run in
# C:\Users\mello\Desktop\Develop\LoRA\train_accel.ps1) rather than
# installing a fresh training stack -- see apps/protection-svc/ml-engine's
# README "LoRA validation experiment" section for the full design writeup.
#
# Run this ON the GPU PC (not remotely delegated further -- this already
# *is* the delegation target). Assumes apps/protection-svc/ml-engine has
# already been synced here (same convention as ml-engine/remote/run_remote.sh).

$ErrorActionPreference = "Stop"

# sd-scripts logs some messages in Japanese; Windows' cp949 (Korean)
# console codepage can't encode them and accelerate.print() crashes with
# UnicodeEncodeError mid-training. Same fix as this machine's own proven
# train_accel.ps1 (C:\Users\mello\Desktop\Develop\LoRA\train_accel.ps1).
$env:PYTHONUTF8 = "0"
$env:PYTHONIOENCODING = "utf-8:replace"
$env:PYTHONUNBUFFERED = "1"

$ML_ENGINE   = "C:\dontai-ml-engine"
$ML_VENV_PY  = "$ML_ENGINE\.venv\Scripts\python.exe"

$KOHYA       = "C:\Users\mello\Desktop\Develop\kohya_ss"
$KOHYA_PY    = "$KOHYA\venv\Scripts\python.exe"
$ACCEL       = "$KOHYA\venv\Scripts\accelerate.exe"
$TRAIN_SCRIPT = "$KOHYA\sd-scripts\train_network.py"
# Generation uses diffusers directly (see generate_and_score.py's docstring
# for why, not sd-scripts/gen_img.py) -- no script path needed here.

$CHECKPOINT  = "C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors"

$EXP_DIR     = "$ML_ENGINE\experiments\lora_validation"
$OUT_DIR     = "$EXP_DIR\out"
$LOGS_DIR    = "$OUT_DIR\logs"

$TRIGGER     = "starrynighttest"
$SEED        = 42

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

# diffusers' pipe.load_lora_weights() (used by generate_and_score.py)
# requires peft, which this venv's existing diffusers/transformers/
# accelerate install didn't already include -- install if missing so a
# fresh run doesn't fail partway through step 4.
$peftCheck = & $KOHYA_PY -c "import peft" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== installing peft (needed for diffusers LoRA loading, not previously in this venv) ==="
    & $KOHYA_PY -m pip install peft -q
}

Write-Host "=== 1/4: preparing datasets (cloak() runs here, CPU-fine, no GPU needed yet) ==="
& $ML_VENV_PY "$EXP_DIR\prepare_dataset.py" --trigger $TRIGGER
if ($LASTEXITCODE -ne 0) { throw "prepare_dataset.py failed" }

function Train-Condition {
    param(
        [string]$Name,
        [string]$DatasetConfig,
        [string]$OutputName
    )
    Write-Host "=== training condition: $Name ==="
    $argList = @(
        "launch",
        "--num_cpu_threads_per_process", "1",
        $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", "$OUT_DIR\lora_$Name",
        "--output_name", $OutputName,
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
        "--seed", "$SEED",
        "--max_data_loader_n_workers", "2"
    )
    New-Item -ItemType Directory -Force -Path "$OUT_DIR\lora_$Name" | Out-Null

    # accelerate/torch write benign warnings to stderr (e.g. "triton not
    # found"). With $ErrorActionPreference = "Stop" (set at script scope),
    # `2>&1 | Tee-Object` wraps every stderr line from a native exe in a
    # terminating ErrorRecord regardless of actual exit code -- a known
    # PowerShell gotcha, not a real failure. Switch to "Continue" for just
    # this call and rely on $LASTEXITCODE (the real signal) instead.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ACCEL @argList *>&1 | Tee-Object -FilePath "$LOGS_DIR\train_$Name.log"
    $ErrorActionPreference = $prevEAP

    if ($LASTEXITCODE -ne 0) { throw "training condition '$Name' failed (exit $LASTEXITCODE) -- see $LOGS_DIR\train_$Name.log" }
}

Write-Host "=== 2/4: training baseline (uncloaked) LoRA -- smoke test this one first ==="
Train-Condition -Name "baseline" -DatasetConfig "$OUT_DIR\dataset_config_dataset.toml" -OutputName "baseline_v1"

Write-Host "=== 3/4: training cloaked LoRA ==="
Train-Condition -Name "cloaked" -DatasetConfig "$OUT_DIR\dataset_config_dataset_cloaked.toml" -OutputName "cloaked_v1"

Write-Host "=== 4/4: generating samples + scoring (CLIP similarity, kohya venv) ==="
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $KOHYA_PY "$EXP_DIR\generate_and_score.py" `
    --checkpoint $CHECKPOINT `
    --lora-baseline "$OUT_DIR\lora_baseline\baseline_v1.safetensors" `
    --lora-cloaked "$OUT_DIR\lora_cloaked\cloaked_v1.safetensors" `
    --true-style-image "$ML_ENGINE\out\real\starry_night.jpg" `
    --trigger $TRIGGER `
    --seed $SEED `
    --out-dir "$OUT_DIR\generated" `
    *>&1 | Tee-Object -FilePath "$OUT_DIR\report.txt"
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) { throw "generate_and_score.py failed (exit $LASTEXITCODE) -- see $OUT_DIR\report.txt" }

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report.txt"
