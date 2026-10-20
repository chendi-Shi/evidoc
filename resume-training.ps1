param([string]$Output = 'runs/experiment-96h', [switch]$WhatIf)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.tools\python\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Project Python is missing. Run scripts/setup.ps1 -Training first.' }
$controlArguments = @('-u', '-m', 'scripts.manage_training', '--action', 'resume', '--output', $Output)
if ($WhatIf) { $controlArguments += '--dry-run' }
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PYTHONUTF8 = '1'
    & $projectPython @controlArguments
    $controlExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $controlExitCode
