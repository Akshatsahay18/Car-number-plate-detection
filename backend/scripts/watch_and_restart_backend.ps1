param(
    [string]$BackendDir = "."
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] $Message"
}

$backendPath = Resolve-Path $BackendDir
$logPath = Join-Path $backendPath "auto_restart.log"
$outLog = Join-Path $backendPath "backend_api.out.log"
$errLog = Join-Path $backendPath "backend_api.err.log"
$pythonExe = Join-Path $backendPath ".venv\Scripts\python.exe"
$bestModel = Join-Path $backendPath "models\best.pt"

Write-Log "Watcher started in $backendPath"

while ($true) {
    $trainProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*scripts\train_yolo.py*" }

    if (-not $trainProcs) {
        break
    }

    Write-Log "Training still running (count=$($trainProcs.Count)); waiting..."
    Start-Sleep -Seconds 15
}

if (Test-Path $bestModel) {
    Write-Log "Training finished and best model found at $bestModel"
} else {
    Write-Log "Training finished but best model not found; restarting backend anyway"
}

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($listener) {
    Stop-Process -Id $listener.OwningProcess -Force
    Write-Log "Stopped existing backend process PID $($listener.OwningProcess)"
}

$newProc = Start-Process -FilePath $pythonExe `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $backendPath `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru

Write-Log "Started backend process PID $($newProc.Id)"
