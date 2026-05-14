# One-shot demo runner. Boots the API in DEMO_MODE so reviewers can click
# through the UI with no API key and no Tesseract.
#
#   .\scripts\demo.ps1            # default port 8000
#   .\scripts\demo.ps1 -Port 9000

param(
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
  Write-Host "Creating virtualenv..." -ForegroundColor Cyan
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

$env:DEMO_MODE = "true"
$env:ANTHROPIC_API_KEY = ""

Write-Host ""
Write-Host "Open http://localhost:$Port in your browser." -ForegroundColor Green
Write-Host "Two sample documents are pre-seeded. Click both -> Generate Draft." -ForegroundColor Green
Write-Host ""

uvicorn api.main:app --host 0.0.0.0 --port $Port
