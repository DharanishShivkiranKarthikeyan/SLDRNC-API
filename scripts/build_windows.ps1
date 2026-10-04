# build_windows.ps1 -- build SLD-RNC on Windows with MSVC + Ninja (Visual Studio 2022) and copy the DLL into dist/.
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1 [-Test]
param([switch]$Test)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with C++ tools not found" }
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue }
}
Set-Location $root
cmake -S . -B build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release | Out-Host
cmake --build build/windows | Out-Host
if ($LASTEXITCODE -ne 0) { throw "build failed" }
$dist = Join-Path $root "dist/windows-x64"
New-Item -ItemType Directory -Force $dist | Out-Null
Copy-Item build/windows/sldrnc.dll, build/windows/sldrnc.lib $dist -Force
New-Item -ItemType Directory -Force (Join-Path $root "python/sldrnc/_native/windows-x64") | Out-Null
Copy-Item build/windows/sldrnc.dll (Join-Path $root "python/sldrnc/_native/windows-x64") -Force
if ($Test) { & build/windows/sldrnc_test.exe; if ($LASTEXITCODE -ne 0) { throw "tests failed" } }
