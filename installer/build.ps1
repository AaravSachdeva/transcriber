# Builds dist\TranscriberSetup.exe. Needs uv and Inno Setup 6:
#   winget install JRSoftware.InnoSetup
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

uv lock --check
if ($LASTEXITCODE) { throw "uv.lock is out of date; run 'uv lock' first." }
Copy-Item (Get-Command uv).Source "$PSScriptRoot\uv.exe"

$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
          "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found. winget install JRSoftware.InnoSetup" }
& $iscc "$PSScriptRoot\transcriber.iss"
if ($LASTEXITCODE) { throw "ISCC failed." }
