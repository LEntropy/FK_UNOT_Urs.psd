# Real LoRA-training validation experiment for concept_misalign.py --
# PHASE4_SCOPING.md §1's own "recommended validation methodology," not run
# anywhere until this script is actually executed on the GPU PC. Answers
# the question concept_misalign.py's module doc states plainly has NOT
# been answered: does training a LoRA on a concept-misaligned image with
# its real caption make generation from that caption drift toward the
# decoy concept instead of the true one?
#
# Structurally mirrors experiments/lora_validation/'s
# run_lora_validation.ps1 (ai-engine branch) -- same GPU PC, same kohya_ss
# setup, same PowerShell gotchas already fixed there (see comments below
# and that script's header). Reuses this machine's existing, already-
# proven kohya_ss setup rather than installing a fresh training stack.
#
# Run this ON the GPU PC (not remotely delegated further -- this already
# *is* the delegation target). Assumes apps/protection-svc/ml-engine has
# already been synced here (same convention as ml-engine/remote/run_remote.sh).
#
# NOT YET RUN. Prepared per PROJECT_DESIGN.md §8 Phase 4 / user request to
# have the validation groundwork ready, not executed as part of writing
# it -- see this experiment folder's README.md before running.

$ErrorActionPreference = "Stop"

# sd-scripts logs some messages in Japanese; Windows' cp949 (Korean)
# console codepage can't encode them and accelerate.print() crashes with
# UnicodeEncodeError mid-training. Same fix as lora_validation's proven
# run_lora_validation.ps1.
$env:PYTHONUTF8 = "0"
$env:PYTHONIOENCODING = "utf-8:replace"
$env:PYTHONUNBUFFERED = "1"

$ML_ENGINE   = "C:\dontai-ml-engine"
$ML_VENV_PY  = "$ML_ENGINE\.venv\Scripts\python.exe"

$KOHYA       = "C:\Users\mello\Desktop\Develop\kohya_ss"
$KOHYA_PY    = "$KOHYA\venv\Scripts\python.exe"
$ACCEL       = "$KOHYA\venv\Scripts\accelerate.exe"
$TRAIN_SCRIPT = "$KOHYA\sd-scripts\train_network.py"
# Generation uses diffusers directly (see generate_and_score.py's
# docstring for why, not sd-scripts/gen_img.py) -- no script path needed here.

$CHECKPOINT  = "C:\Users\mello\ComfyUI-Installs\Philosophyz\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors"

$EXP_DIR     = "$ML_ENGINE\experiments\concept_misalignment_validation"
$OUT_DIR     = "$EXP_DIR\out"
$LOGS_DIR    = "$OUT_DIR\logs"

# Image set lives in prepare_dataset.py's IMAGE_CONFIGS (currently 5:
# mona_lisa, starry_night, the_scream, great_wave, girl_pearl_earring) --
# this script reads the resulting manifest.json rather than hardcoding
# names here too.
$SEEDS       = @(1, 2, 3)
$RUN_NAME    = "v1"

New-Item -ItemType Directory -Force -Path $LOGS_DIR | Out-Null

# diffusers' pipe.load_lora_weights() (used by generate_and_score.py)
# requires peft -- install if missing so a fresh run doesn't fail partway
# through the scoring step (same check as lora_validation's script).
& $KOHYA_PY -c "import peft" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== installing peft (needed for diffusers LoRA loading, not previously in this venv) ==="
    & $KOHYA_PY -m pip install peft -q
}

# concept_misalign.py's own module doc: loading open_clip's pretrained
# checkpoint was blocked by a *different* environment's external-code
# safety gate (not this one -- this machine already has the checkpoint
# cache or network access a sandboxed dev environment doesn't). If this
# fails here too, that gate is this run's actual first blocker, before
# training even starts -- resolve it (approve the download / pre-seed the
# checkpoint cache) before continuing.
& $ML_VENV_PY -c "from model import ConceptFeatureExtractor; import torch; ConceptFeatureExtractor(torch.device('cpu'))" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "ConceptFeatureExtractor failed to load its CLIP checkpoint -- see concept_misalign.py's module doc for the known blocker this hit in a different environment. Resolve before continuing."
}

Write-Host "=== 1/3: preparing datasets for all images (misalign() runs here, CPU-fine, no GPU needed yet) ==="
& $ML_VENV_PY "$EXP_DIR\prepare_dataset.py"
if ($LASTEXITCODE -ne 0) { throw "prepare_dataset.py failed" }

$manifest = Get-Content "$OUT_DIR\manifest.json" | ConvertFrom-Json

function Train-Condition {
    param(
        [string]$RunTag,
        [string]$Condition,
        [string]$DatasetConfig,
        [int]$Seed
    )
    $outputName = "${Condition}_$RUN_NAME"
    Write-Host "=== training: $RunTag / $Condition (seed $Seed) ==="
    $argList = @(
        "launch",
        "--num_cpu_threads_per_process", "1",
        $TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", $CHECKPOINT,
        "--dataset_config", $DatasetConfig,
        "--output_dir", "$OUT_DIR\lora_${RunTag}_${Condition}",
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
    New-Item -ItemType Directory -Force -Path "$OUT_DIR\lora_${RunTag}_${Condition}" | Out-Null

    # accelerate/torch write benign warnings to stderr (e.g. "triton not
    # found"). With $ErrorActionPreference = "Stop" (set at script scope),
    # redirecting stderr wraps every line from a native exe in a
    # terminating ErrorRecord regardless of actual exit code -- a known
    # PowerShell gotcha, not a real failure. Switch to "Continue" for just
    # this call and rely on $LASTEXITCODE (the real signal) instead.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $ACCEL @argList *>&1 | Tee-Object -FilePath "$LOGS_DIR\train_${RunTag}_${Condition}.log"
    $ErrorActionPreference = $prevEAP

    if ($LASTEXITCODE -ne 0) { throw "training '$RunTag/$Condition' failed (exit $LASTEXITCODE) -- see $LOGS_DIR\train_${RunTag}_${Condition}.log" }
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
        Write-Host "=== 2/3: [$runNum/$totalRuns] $runTag misaligned ==="
        Train-Condition -RunTag $runTag -Condition "misaligned" -DatasetConfig $entry.misaligned_dataset_config -Seed $seed
    }
}

Write-Host "=== 3/3: generating samples + scoring (CLIP similarity to true AND decoy concept, kohya venv) ==="
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$seedsArg = ($SEEDS -join ",")
& $KOHYA_PY "$EXP_DIR\generate_and_score.py" `
    --checkpoint $CHECKPOINT `
    --manifest "$OUT_DIR\manifest.json" `
    --lora-root $OUT_DIR `
    --seeds $seedsArg `
    --run-name $RUN_NAME `
    --out-dir "$OUT_DIR\generated" `
    *>&1 | Tee-Object -FilePath "$OUT_DIR\report.txt"
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) { throw "generate_and_score.py failed (exit $LASTEXITCODE) -- see $OUT_DIR\report.txt" }

Write-Host ""
Write-Host "Full report written to $OUT_DIR\report.txt"
