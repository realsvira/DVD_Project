# One-time environment setup for Windows PowerShell.
# Run from the project root:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Optional: -Ngspice "C:\path\to\ngspice_con.exe" if ngspice lives elsewhere.
param(
    [string]$Ngspice = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== Python =="
python --version
if ($LASTEXITCODE -ne 0) { throw "python not found on PATH" }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "== Creating virtual environment .venv =="
    python -m venv .venv
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "== Installing dependencies =="
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if ($Ngspice -ne "") { $env:NGSPICE = $Ngspice }

Write-Host "== Inventory of supplied inputs =="
& $Py scripts\inventory_inputs.py

Write-Host "== Environment check =="
& $Py scripts\check_environment.py
$check = $LASTEXITCODE

Write-Host "== Unit tests =="
& $Py -m pytest -q tests
if ($check -ne 0 -or $LASTEXITCODE -ne 0) { exit 1 }
Write-Host "Setup complete."
