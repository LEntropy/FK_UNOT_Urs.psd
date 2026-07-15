# Launches run_l2_preset_validation.ps1 fully detached (survives the SSH
# session/channel that launched it closing) with output reliably captured.
# A prior attempt used Start-Process -RedirectStandardOutput directly from
# an inline ssh command; it produced 0-byte log files, most likely because
# heavy quoting through bash -> ssh -> cmd.exe -> powershell mangled the
# argument list. Running as a real wrapper script (single simple ssh
# argument, no nested quoting) avoids that class of problem entirely.
$scriptPath = "C:\dontai-ml-engine\experiments\lora_validation\run_l2_preset_validation.ps1"
$stdout = "C:\dontai-ml-engine\experiments\lora_validation\out\l2_run_stdout.log"
$stderr = "C:\dontai-ml-engine\experiments\lora_validation\out\l2_run_stderr.log"

$proc = Start-Process powershell `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptPath) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Write-Output "launched PID $($proc.Id)"
