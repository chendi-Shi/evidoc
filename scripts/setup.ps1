param([switch]$Training, [switch]$UseChinaMirror)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runtimeDir = Join-Path $projectRoot '.tools\python'
$projectPython = Join-Path $runtimeDir 'python.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot '.tools') | Out-Null
if (-not (Test-Path -LiteralPath $projectPython)) {
    $archivePath = Join-Path $projectRoot '.tools\python-3.12.10.zip'
    & curl.exe -L --fail --retry 4 -o $archivePath 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip'
    if ($LASTEXITCODE -ne 0) { throw 'Python download failed; retry setup.' }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $runtimeDir -Force
    $signature = Get-AuthenticodeSignature -LiteralPath $projectPython
    if ($signature.Status -ne 'Valid') { throw 'Python executable signature verification failed.' }
    @('python312.zip','.','Lib\site-packages','..\..','import site') | Set-Content (Join-Path $runtimeDir 'python312._pth') -Encoding ascii
}
& $projectPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    $getPip = Join-Path $projectRoot '.tools\get-pip.py'
    & curl.exe -L --fail --retry 4 -o $getPip 'https://bootstrap.pypa.io/get-pip.py'
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap download failed.' }
    & $projectPython $getPip
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
}
$indexArgs = @()
if ($UseChinaMirror) { $indexArgs = @('--index-url','https://mirrors.aliyun.com/pypi/simple/') }
& $projectPython -m pip install @indexArgs -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
if ($Training) {
    & $projectPython -m pip install 'torch==2.7.1+cu118' --index-url 'https://download.pytorch.org/whl/cu118'
    if ($LASTEXITCODE -ne 0) { throw 'GPU PyTorch installation failed.' }
    & $projectPython -m pip install @indexArgs -r training/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
    & $projectPython scripts/probe_training_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'GPU probe failed; do not launch training.' }
}
Write-Host 'Ready. Start the app with ./start.ps1. Training commands are in README.md.'
