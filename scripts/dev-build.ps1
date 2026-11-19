# Incremental rebuild of the game after a change to the port sources.
#
# Needs one complete run of Setup.ps1 first (it configures build\game).
# Skips vcpkg, the dependencies and the RPX recompilation: Ninja rebuilds only
# what changed, then the new Splatoon.exe is copied into the game folder.
#
# Builds are queued: several callers (people, agents) can start this script at
# the same time; each takes a ticket in build\build-queue and they run one at
# a time, first come first served. A caller that dies leaves a stale ticket,
# which the next one removes. No lock file to create or delete by hand.
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev-build.ps1 -Output G:\jeux\splatoon\game
#   ... -Label aim     name shown in the queue and in logs\build-queue.log
#   ... -CopyTo <dir>  also copy Splatoon.exe there while still holding the turn
#   ... -Status        show the queue and the last builds, then exit
#   ... -Fast:$false   switch the build folder back to the release flags (/GL + /LTCG)

param(
    [string]$Output,          # game folder to update (skipped when empty)
    [int]$Jobs = 0,           # 0 = automatic, from cores and RAM
    [Nullable[bool]]$Fast = $null,  # $true/$false reconfigures RBREW_FAST_LINK; omitted keeps it
    [string]$Label = 'build', # who is building (queue display)
    [string]$CopyTo,          # extra folder that receives Splatoon.exe before the turn ends
    [switch]$Status           # print the queue and exit
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$gameBuild = Join-Path $Root 'build\game'
$queueDir = Join-Path $Root 'build\build-queue'
$queueLog = Join-Path (Split-Path -Parent $Root) 'logs\build-queue.log'
[void][IO.Directory]::CreateDirectory($queueDir)
[void][IO.Directory]::CreateDirectory((Split-Path -Parent $queueLog))

# Live tickets, oldest first. A ticket whose process is gone is removed.
function Get-Queue {
    $live = @()
    foreach ($file in @(Get-ChildItem -LiteralPath $queueDir -Filter *.ticket -ErrorAction SilentlyContinue | Sort-Object Name)) {
        $parts = $file.BaseName -split '-', 3
        $owner = Get-Process -Id ([int]$parts[1]) -ErrorAction SilentlyContinue
        if ($owner -and $owner.ProcessName -match '^(powershell|pwsh)') {
            $live += [pscustomobject]@{ File = $file.FullName; Name = $file.Name; Label = $parts[2]; Since = $file.CreationTime }
        } else {
            Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue
        }
    }
    return , $live
}
function Write-QueueLog([string]$Text) {
    try { Add-Content -LiteralPath $queueLog -Value ("{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Text) -Encoding utf8 } catch {}
}

if ($Status) {
    $queue = Get-Queue
    if ($queue.Count -eq 0) { Write-Host 'Build queue: empty.' }
    for ($i = 0; $i -lt $queue.Count; $i++) {
        $state = if ($i -eq 0) { 'building' } else { "waiting (#$i)" }
        Write-Host ("  {0,-14} {1,-12} since {2:HH:mm:ss}" -f $state, $queue[$i].Label, $queue[$i].Since)
    }
    if (Test-Path -LiteralPath $queueLog) { Write-Host 'Last builds:'; Get-Content -LiteralPath $queueLog -Tail 8 }
    exit 0
}

if (-not (Test-Path -LiteralPath (Join-Path $gameBuild 'build.ninja'))) {
    Write-Host "build\game is not configured: run Setup.ps1 once first." -ForegroundColor Red
    exit 1
}

# Take a ticket and wait for the turn.
$safeLabel = ($Label -replace '[^A-Za-z0-9_]', '_')
$ticket = Join-Path $queueDir ("{0}-{1}-{2}.ticket" -f (Get-Date -Format 'yyyyMMddHHmmssfff'), $PID, $safeLabel)
Set-Content -LiteralPath $ticket -Value $Label -Encoding ascii
$waitWatch = [Diagnostics.Stopwatch]::StartNew()
$lastShown = -1
while ($true) {
    $queue = Get-Queue
    $position = 0
    for ($i = 0; $i -lt $queue.Count; $i++) { if ($queue[$i].File -eq $ticket) { $position = $i; break } }
    if ($position -eq 0) { break }
    if ($position -ne $lastShown) {
        Write-Host ("Build queue: {0} ahead of '{1}' (now building: {2})." -f $position, $Label, $queue[0].Label) -ForegroundColor Yellow
        $lastShown = $position
    }
    Start-Sleep -Seconds 2
}
$waited = [math]::Round($waitWatch.Elapsed.TotalSeconds)

$code = 0
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    # Visual Studio's compiler environment, cmake and ninja.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw 'Visual Studio C++ tools not found.' }
    $envLines = cmd.exe /c "call `"$(Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat')`" >nul & set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
    $cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (-not (Test-Path -LiteralPath $cmake)) { $cmake = (Get-Command cmake).Source }

    if ($null -ne $Fast) {
        $value = if ($Fast) { 'ON' } else { 'OFF' }
        & $cmake -S (Join-Path $Root 'recompiler\port') -B $gameBuild "-DRBREW_FAST_LINK=$value"
        $code = $LASTEXITCODE
    }
    if ($code -eq 0) {
        if ($Jobs -le 0) {
            $ramGb = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
            $Jobs = [math]::Max(1, [math]::Min([Environment]::ProcessorCount, [math]::Floor($ramGb / 4)))
        }
        & $cmake --build $gameBuild --target Gambit -- -j "$Jobs"
        $code = $LASTEXITCODE
    }
    if ($code -eq 0) {
        Write-Host ("Build done in {0:N0} s ({1} jobs, waited {2} s in the queue)." -f $sw.Elapsed.TotalSeconds, $Jobs, $waited) -ForegroundColor Green
        $exe = Get-ChildItem -LiteralPath $gameBuild -Recurse -Filter Splatoon.exe | Select-Object -First 1
        foreach ($target in @($Output, $CopyTo)) {
            if (-not $target) { continue }
            Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $target 'Splatoon.exe') -Force
            Write-Host "Updated $(Join-Path $target 'Splatoon.exe')" -ForegroundColor Green
        }
    } else {
        Write-Host 'Build failed.' -ForegroundColor Red
    }
} catch {
    Write-Host "Build failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($code -eq 0) { $code = 1 }
} finally {
    Write-QueueLog ("{0,-12} waited {1,4} s  built {2,4:N0} s  exit {3}" -f $Label, $waited, $sw.Elapsed.TotalSeconds, $code)
    Remove-Item -LiteralPath $ticket -Force -ErrorAction SilentlyContinue
}
exit $code
