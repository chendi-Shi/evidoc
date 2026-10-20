param([int]$Port = 8083, [switch]$Inspect, [string]$Bundle)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.tools\python\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Install the training environment first; see training/requirements.txt.'
}
$env:PYTHONUTF8 = '1'
$adapterArguments = @('-m', 'scripts.serve_adapter', '--port', "$Port")
if ($Inspect) { $adapterArguments += '--inspect' }
if ($Bundle) { $adapterArguments += @('--bundle', $Bundle) }
& $projectPython @adapterArguments
exit $LASTEXITCODE
