# Full reproducible pipeline (Windows PowerShell). Linux/macOS: same commands with python from .venv/bin.
#   powershell -ExecutionPolicy Bypass -File training\run_all.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_XET = "1"
Set-Location $PSScriptRoot

# Use the project's own venv if it exists, else the shared projects/.venv, else whatever "python" is.
$py = @("..\.venv\Scripts\python.exe", "..\..\.venv\Scripts\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) { $py = "python" }
Write-Host "using $py"

& $py prepare_data.py                      ; if ($LASTEXITCODE) { exit 1 }   # data + EDA
& $py evaluate.py coco-baseline            ; if ($LASTEXITCODE) { exit 1 }   # pretrained models on COCO
& $py evaluate.py domain-gap               ; if ($LASTEXITCODE) { exit 1 }   # pretrained models on VisDrone
& $py train.py --exp yolo11n-visdrone      ; if ($LASTEXITCODE) { exit 1 }
& $py train.py --exp yolo11s-visdrone      ; if ($LASTEXITCODE) { exit 1 }
& $py evaluate.py finetuned                ; if ($LASTEXITCODE) { exit 1 }   # mAP, per-class, sizes
& $py export_onnx.py                       ; if ($LASTEXITCODE) { exit 1 }   # ONNX + parity + latency
