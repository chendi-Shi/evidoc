param([int]$Port = 8767)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.tools\python\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Run scripts/setup.ps1 first, or use Python 3.12: python -m uvicorn app.api:app --host 127.0.0.1 --port 8767'
}
& $projectPython -m uvicorn app.api:app --host 127.0.0.1 --port $Port
