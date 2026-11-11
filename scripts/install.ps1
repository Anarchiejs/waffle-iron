#Requires -Version 5.1
<#
.SYNOPSIS
  Install Waffle Iron on Windows and open the interface.
.DESCRIPTION
  Entry point for install.cmd and launch.cmd. Windows PowerShell 5.1.
  -AutoTest checks the CPU/CUDA selection logic without compiling anything.
#>
param(
    [switch]$Help,
    [switch]$Launch,
    [switch]$NoLaunch,
    [switch]$Cpu,
    [switch]$Cuda,
    [switch]$Tests,
    [string]$Architecture,
    [switch]$AutoTest
)

$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$script:TranscriptOn = $false
$script:TestFailures = 0

function Write-Step([string]$Message) {
    Write-Host ("==> " + $Message) -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host ("    " + $Message) -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host ("    " + $Message) -ForegroundColor Yellow
}

function ConvertTo-JsonString([string]$Text) {
    if ($null -eq $Text) { return '' }
    $builder = New-Object System.Text.StringBuilder
    foreach ($ch in $Text.ToCharArray()) {
        $code = [int]$ch
        if ($ch -eq '"') { [void]$builder.Append('\"') }
        elseif ($ch -eq '\') { [void]$builder.Append('\\') }
        elseif ($ch -eq "`b") { [void]$builder.Append('\b') }
        elseif ($ch -eq "`f") { [void]$builder.Append('\f') }
        elseif ($ch -eq "`n") { [void]$builder.Append('\n') }
        elseif ($ch -eq "`r") { [void]$builder.Append('\r') }
        elseif ($ch -eq "`t") { [void]$builder.Append('\t') }
        elseif ($code -lt 32) { [void]$builder.AppendFormat('\u{0:x4}', $code) }
        else { [void]$builder.Append($ch) }
    }
    return $builder.ToString()
}

function Convert-ComputeCapability([string]$Capability) {
    if ($null -eq $Capability) { return $null }
    $text = $Capability.Trim()
    if ($text -notmatch '^(\d+)\.(\d)$') { return $null }
    return ($Matches[1] + $Matches[2])
}

function Format-Architectures([string[]]$Archs) {
    $list = New-Object System.Collections.Generic.List[string]
    foreach ($arch in @($Archs)) {
        if ($arch -and -not $list.Contains([string]$arch)) { [void]$list.Add([string]$arch) }
    }
    $sorted = New-Object System.Collections.Generic.List[string]
    foreach ($arch in @($list | Sort-Object { [int]$_ })) {
        if ($arch) { [void]$sorted.Add([string]$arch) }
    }
    return ($sorted -join ';')
}

function Parse-ArchitectureRequest([string]$Text) {
    $list = New-Object System.Collections.Generic.List[string]
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return [pscustomobject]@{ Ok = $true; ArchitectureText = ''; Error = '' }
    }
    foreach ($part in ($Text -split '[;,\s]+')) {
        if (-not $part) { continue }
        $arch = $null
        if ($part -match '^\d+\.\d$') { $arch = Convert-ComputeCapability $part }
        elseif ($part -match '^\d{2,3}$') { $arch = $part }
        if (-not $arch) {
            return [pscustomobject]@{
                Ok = $false
                ArchitectureText = ''
                Error = ("Invalid CUDA architecture: " + $part + ". Examples: 89 or 8.9.")
            }
        }
        if (-not $list.Contains($arch)) { [void]$list.Add($arch) }
    }
    if ($list.Count -eq 0) {
        return [pscustomobject]@{ Ok = $false; ArchitectureText = ''; Error = "Empty CUDA architecture." }
    }
    return [pscustomobject]@{ Ok = $true; ArchitectureText = (Format-Architectures @($list.ToArray())); Error = '' }
}

function Parse-CMakeVersion([string]$Text) {
    if ($Text -match 'cmake version (\d+)\.(\d+)\.(\d+)') {
        return [version]("$($Matches[1]).$($Matches[2]).$($Matches[3])")
    }
    return $null
}

function Parse-NvccVersion([string]$Text) {
    if ($Text -match 'release (\d+)\.(\d+)') {
        return [version]("$($Matches[1]).$($Matches[2])")
    }
    return $null
}

function Get-CudaToolkitWarning {
    param([version]$Toolkit, [string]$ArchitectureText)
    if (-not $Toolkit -or -not $ArchitectureText) { return $null }
    $max = 0
    foreach ($part in ($ArchitectureText -split ';')) {
        $value = 0
        if ([int]::TryParse($part, [ref]$value) -and $value -gt $max) { $max = $value }
    }
    if ($max -ge 100 -and $Toolkit -lt [version]'12.8') {
        return ("sm_" + $max + " requires CUDA Toolkit 12.8 or newer (detected: " + $Toolkit + ").")
    }
    if ($max -ge 89 -and $Toolkit -lt [version]'11.8') {
        return ("sm_" + $max + " requires CUDA Toolkit 11.8 or newer (detected: " + $Toolkit + ").")
    }
    if ($max -ge 80 -and $Toolkit -lt [version]'11.1') {
        return ("sm_" + $max + " requires CUDA Toolkit 11.1 or newer (detected: " + $Toolkit + ").")
    }
    return $null
}

function New-Plan {
    param(
        [bool]$Ok,
        [string]$Message,
        [string]$Kind,
        [string]$Preset,
        [string]$ArchitectureText,
        [string]$Note,
        [bool]$WriteUserPreset
    )
    $exe = 'build\cpu-release\WaffleIron.exe'
    if ($Preset) { $exe = 'build\' + $Preset + '\WaffleIron.exe' }
    return [pscustomobject]@{
        Ok = $Ok
        Error = $Message
        Kind = $Kind
        Preset = $Preset
        ArchitectureText = $ArchitectureText
        ExeRelative = $exe
        Note = $Note
        WriteUserPreset = $WriteUserPreset
    }
}

function Select-InstallPlan {
    param(
        [bool]$ForceCpu,
        [bool]$ForceCuda,
        [string]$RequestedArchitecture,
        [bool]$HasCudaToolkit,
        [string[]]$DetectedArchitectures,
        [string[]]$GpuLabels
    )
    if ($ForceCpu -and $ForceCuda) {
        return (New-Plan $false "Choose only one mode: -Cpu or -Cuda." '' '' '' '' $false)
    }
    $requested = Parse-ArchitectureRequest $RequestedArchitecture
    if (-not $requested.Ok) {
        return (New-Plan $false $requested.Error '' '' '' '' $false)
    }
    if ($ForceCpu -and $requested.ArchitectureText) {
        return (New-Plan $false "-Architecture cannot be used with -Cpu." '' '' '' '' $false)
    }
    if (($ForceCuda -or $requested.ArchitectureText) -and -not $HasCudaToolkit) {
        $why = "Install the CUDA Toolkit, then run install.cmd again:`r`nhttps://developer.nvidia.com/cuda-downloads"
        if ($ForceCuda) {
            $why = "The -Cuda option requires the CUDA Toolkit (nvcc), which was not found.`r`n" + $why
        } else {
            $why = "A CUDA architecture was requested, but nvcc was not found.`r`n" + $why
        }
        return (New-Plan $false $why '' '' '' '' $false)
    }
    $detectedText = Format-Architectures $DetectedArchitectures
    $archText = $requested.ArchitectureText
    if (-not $archText) { $archText = $detectedText }
    $labels = @($GpuLabels | Where-Object { $_ })
    $labelText = $labels -join ', '

    if ($ForceCpu) {
        return (New-Plan $true '' 'cpu' 'cpu-release' '' "CPU build (-Cpu)." $false)
    }
    if ($HasCudaToolkit -and ($ForceCuda -or $archText)) {
        if ($archText) {
            $note = "CUDA build for sm_" + (($archText -split ';') -join ', sm_') + "."
            if ($labelText) { $note += " GPU: " + $labelText + "." }
            if ($requested.ArchitectureText -and $detectedText -and $requested.ArchitectureText -ne $detectedText) {
                $note += " The -Architecture option overrides detection (" + $detectedText + ")."
            }
            return (New-Plan $true '' 'cuda' 'cuda-release-local' $archText $note $true)
        }
        $note = "CUDA build using the cuda-release preset architecture (sm_89): the GPU was not identified. Use -Architecture 86 to set another one."
        return (New-Plan $true '' 'cuda' 'cuda-release' '' $note $false)
    }
    if ($labelText) {
        $note = "NVIDIA GPU detected (" + $labelText + "), but the CUDA Toolkit is missing: CPU build. Install the CUDA Toolkit, then run install.cmd again for a GPU build.`r`nhttps://developer.nvidia.com/cuda-downloads"
        return (New-Plan $true '' 'cpu' 'cpu-release' '' $note $false)
    }
    if ($HasCudaToolkit) {
        $note = "CUDA Toolkit present, no NVIDIA GPU detected: CPU build. To force the cuda-release preset (sm_89), run again with -Cuda."
        return (New-Plan $true '' 'cpu' 'cpu-release' '' $note $false)
    }
    return (New-Plan $true '' 'cpu' 'cpu-release' '' "No NVIDIA GPU and no CUDA Toolkit: CPU build." $false)
}

function New-CudaUserPresetJson {
    param([string]$ArchitectureText, [string]$Summary)
    $arch = ConvertTo-JsonString $ArchitectureText
    $display = ConvertTo-JsonString ("CUDA, Release (install.cmd: " + $Summary + ")")
    $lines = @(
        '{',
        '  "version": 6,',
        '  "vendor": {',
        '    "waffle.iron/install/1.0": { "generated": true }',
        '  },',
        '  "configurePresets": [',
        '    {',
        '      "name": "cuda-release-local",',
        ('      "displayName": "' + $display + '",'),
        '      "inherits": "cuda-release",',
        '      "cacheVariables": {',
        ('        "CMAKE_CUDA_ARCHITECTURES": "' + $arch + '"'),
        '      }',
        '    }',
        '  ],',
        '  "buildPresets": [',
        '    { "name": "cuda-release-local", "configurePreset": "cuda-release-local" }',
        '  ]',
        '}'
    )
    return (($lines -join "`n") + "`n")
}

function Assert-Equal {
    param([string]$Name, $Actual, $Expected)
    $left = if ($null -eq $Actual) { '<null>' } else { [string]$Actual }
    $right = if ($null -eq $Expected) { '<null>' } else { [string]$Expected }
    if ($left -ne $right) {
        Write-Host ("FAIL " + $Name + ": got [" + $left + "] expected [" + $right + "]") -ForegroundColor Red
        $script:TestFailures++
    }
}

function Assert-True {
    param([string]$Name, $Condition)
    if (-not $Condition) {
        Write-Host ("FAIL " + $Name) -ForegroundColor Red
        $script:TestFailures++
    }
}

function Assert-Plan {
    param($Name, $Plan, [bool]$Ok, [string]$Kind, [string]$Preset, [string]$Arch, [string]$NoteLike)
    Assert-Equal ($Name + " ok") ([string]$Plan.Ok) ([string]$Ok)
    if ($Ok) {
        Assert-Equal ($Name + " kind") $Plan.Kind $Kind
        Assert-Equal ($Name + " preset") $Plan.Preset $Preset
        Assert-Equal ($Name + " arch") $Plan.ArchitectureText $Arch
        if ($NoteLike) { Assert-True ($Name + " note " + $Plan.Note) ($Plan.Note -like $NoteLike) }
    }
}

function Invoke-InstallSelfTest {
    $script:TestFailures = 0
    Assert-Equal 'cap 8.9' (Convert-ComputeCapability '8.9') '89'
    Assert-Equal 'cap 12.0' (Convert-ComputeCapability '12.0') '120'
    Assert-Equal 'cap 10.0' (Convert-ComputeCapability '10.0') '100'
    Assert-Equal 'cap 7.5' (Convert-ComputeCapability '7.5') '75'
    Assert-Equal 'cap 9.0' (Convert-ComputeCapability '9.0') '90'
    Assert-Equal 'cap 8.6' (Convert-ComputeCapability ' 8.6 ') '86'
    Assert-Equal 'cap 8.90' (Convert-ComputeCapability '8.90') $null
    Assert-Equal 'cap empty' (Convert-ComputeCapability '') $null
    Assert-Equal 'cap sm' (Convert-ComputeCapability 'sm_89') $null

    $none = Select-InstallPlan $false $false '' $false @() @()
    Assert-Plan 'nothing' $none $true 'cpu' 'cpu-release' '' '*No NVIDIA GPU*'

    $gpuOnly = Select-InstallPlan $false $false '' $false @('89') @('RTX 4060 Ti (sm_89)')
    Assert-Plan 'gpu without toolkit' $gpuOnly $true 'cpu' 'cpu-release' '' '*CUDA Toolkit*'

    $cuda = Select-InstallPlan $false $false '' $true @('89') @('RTX 4060 Ti (sm_89)')
    Assert-Plan 'cuda 89' $cuda $true 'cuda' 'cuda-release-local' '89' '*sm_89*'
    Assert-Equal 'cuda write' ([string]$cuda.WriteUserPreset) ([string]$true)
    Assert-Equal 'cuda exe' $cuda.ExeRelative 'build\cuda-release-local\WaffleIron.exe'

    $multi = Select-InstallPlan $false $false '' $true @('89', '86') @('A', 'B')
    Assert-Plan 'multi' $multi $true 'cuda' 'cuda-release-local' '86;89' '*sm_86, sm_89*'

    $forceCpu = Select-InstallPlan $true $false '' $true @('89') @('RTX')
    Assert-Plan 'force cpu' $forceCpu $true 'cpu' 'cpu-release' '' '*-Cpu*'

    $forceCudaMissing = Select-InstallPlan $false $true '' $false @() @()
    Assert-Equal 'force cuda without toolkit' ([string]$forceCudaMissing.Ok) ([string]$false)
    Assert-True 'force cuda message' ($forceCudaMissing.Error -like '*nvcc*')

    $forceCudaBlind = Select-InstallPlan $false $true '' $true @() @()
    Assert-Plan 'force cuda blind' $forceCudaBlind $true 'cuda' 'cuda-release' '' '*sm_89*'
    Assert-Equal 'force cuda write' ([string]$forceCudaBlind.WriteUserPreset) ([string]$false)

    $requested = Select-InstallPlan $false $false '8.9' $true @() @()
    Assert-Plan 'arch 8.9' $requested $true 'cuda' 'cuda-release-local' '89' '*sm_89*'

    $requestedMany = Select-InstallPlan $false $false '120,89' $true @('86') @('older')
    Assert-Plan 'arch list' $requestedMany $true 'cuda' 'cuda-release-local' '89;120' '*overrides*'

    $dup = Select-InstallPlan $false $false '89,89' $true @() @()
    Assert-Plan 'duplicate arch' $dup $true 'cuda' 'cuda-release-local' '89' '*sm_89*'

    $bad = Select-InstallPlan $false $false 'sm_89' $true @() @()
    Assert-Equal 'invalid arch' ([string]$bad.Ok) ([string]$false)

    $both = Select-InstallPlan $true $true '' $true @() @()
    Assert-Equal 'cpu and cuda' ([string]$both.Ok) ([string]$false)

    $archCpu = Select-InstallPlan $true $false '86' $true @() @()
    Assert-Equal 'arch and cpu' ([string]$archCpu.Ok) ([string]$false)

    $toolkitOnly = Select-InstallPlan $false $false '' $true @() @()
    Assert-Plan 'toolkit only' $toolkitOnly $true 'cpu' 'cpu-release' '' '*sm_89*'

    $json = New-CudaUserPresetJson -ArchitectureText '86;89' -Summary 'RTX "4060"'
    $parsed = $json | ConvertFrom-Json
    Assert-Equal 'json name' $parsed.configurePresets[0].name 'cuda-release-local'
    Assert-Equal 'json arch' $parsed.configurePresets[0].cacheVariables.CMAKE_CUDA_ARCHITECTURES '86;89'
    Assert-Equal 'json bin missing' $parsed.configurePresets[0].PSObject.Properties['binaryDir'] $null
    Assert-Equal 'json inherits' $parsed.configurePresets[0].inherits 'cuda-release'
    Assert-True 'json quote' ($parsed.configurePresets[0].displayName -like '*RTX "4060"*')
    $vendor = $parsed.vendor.PSObject.Properties['waffle.iron/install/1.0'].Value
    Assert-True 'json vendor' ($vendor.generated -eq $true)
    Assert-True 'json marker' ($json.Contains('waffle.iron/install/1.0'))

    Assert-Equal 'cmake 3.25' (Parse-CMakeVersion "cmake version 3.25.1`n") ([version]'3.25.1')
    Assert-True 'cmake old' ((Parse-CMakeVersion 'cmake version 3.24.0') -lt [version]'3.25.0')
    Assert-Equal 'nvcc' (Parse-NvccVersion "Cuda compilation tools, release 12.6, V12.6.77") ([version]'12.6')
    Assert-True 'warn 120' ((Get-CudaToolkitWarning ([version]'12.6') '120') -like '*12.8*')
    Assert-Equal 'warn 89 recent' (Get-CudaToolkitWarning ([version]'12.6') '89') $null
    Assert-True 'warn 89 old' ((Get-CudaToolkitWarning ([version]'11.7') '89') -like '*11.8*')
    Assert-True 'warn 86' ((Get-CudaToolkitWarning ([version]'11.0') '86') -like '*11.1*')
    Assert-Equal 'warn 75' (Get-CudaToolkitWarning ([version]'11.0') '75') $null

    if ($script:TestFailures -gt 0) {
        throw ("Self-test: " + $script:TestFailures + " failure(s).")
    }
    Write-Host "Self-test passed."
}

function Show-InstallHelp {
    Write-Host @"
Waffle Iron - install and launch

  install.cmd
      Checks prerequisites, fetches llama.cpp if needed, chooses CPU or CUDA,
      builds, then opens the interface.

  install.cmd -NoLaunch
      Builds without opening the interface.

  install.cmd -Cpu
      Forces a CPU build (preset cpu-release).

  install.cmd -Cuda
      Forces a CUDA build. With no GPU identified, the architecture stays sm_89.

  install.cmd -Architecture 86
      Sets the CUDA architecture (86, 89, 120...). 8.9 is accepted and becomes 89.
      Several values: -Architecture 86,89

  install.cmd -Tests
      Runs the tests after the build.

  install.cmd -Help
      Shows this help.

  launch.cmd
      Opens an already built WaffleIron.exe, without rebuilding.

The first CUDA build is long. Running install.cmd again resumes where it stopped.
Log: build\install.log
"@
}

function Add-ToolDirectory([string]$FilePath) {
    if (-not $FilePath) { return }
    $dir = Split-Path -Parent $FilePath
    if (-not $dir) { return }
    $parts = @($env:PATH -split ';')
    if ($parts -notcontains $dir) { $env:PATH = $dir + ';' + $env:PATH }
}

function Find-FirstFile([string[]]$Candidates) {
    foreach ($candidate in @($Candidates)) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    return $null
}

function Import-VcEnvironment([string]$VcVarsPath) {
    $env:WAFFLE_VCVARS = $VcVarsPath
    $batch = Join-Path $env:TEMP ("waffle-vcvars-" + $PID + ".cmd")
    $content = "@echo off`r`ncall `"%WAFFLE_VCVARS%`" >nul`r`nif errorlevel 1 exit /b 1`r`nchcp 65001 >nul`r`nset`r`n"
    [System.IO.File]::WriteAllText($batch, $content, [System.Text.Encoding]::ASCII)
    $previous = [Console]::OutputEncoding
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    try {
        $lines = & cmd.exe /d /c "`"$batch`""
        $code = $LASTEXITCODE
    } finally {
        [Console]::OutputEncoding = $previous
        Remove-Item -LiteralPath $batch -Force -ErrorAction SilentlyContinue
        Remove-Item Env:WAFFLE_VCVARS -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) {
        throw ("Cannot initialize Visual Studio (vcvars64.bat, code " + $code + ").")
    }
    foreach ($line in @($lines)) {
        if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            if ($Matches[1] -eq 'WAFFLE_VCVARS') { continue }
            Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2]
        }
    }
}

function Get-VisualStudioRoot {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $raw = @(& $vswhere -latest -version '[17.0,)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
        foreach ($line in $raw) {
            $trim = ([string]$line).Trim()
            if ($trim -match '^[A-Za-z]:\\') {
                $vcvars = Join-Path $trim 'VC\Auxiliary\Build\vcvars64.bat'
                if (Test-Path -LiteralPath $vcvars) { return $trim }
            }
        }
    }
    $fallback = 'C:\Program Files\Microsoft Visual Studio\18\Community'
    if (Test-Path -LiteralPath (Join-Path $fallback 'VC\Auxiliary\Build\vcvars64.bat')) { return $fallback }
    return $null
}

function Find-CudaToolkitRoot {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:CUDA_PATH) { [void]$candidates.Add($env:CUDA_PATH) }
    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $root) {
        $dirs = @(Get-ChildItem -LiteralPath $root -Directory | Sort-Object Name -Descending)
        foreach ($dir in $dirs) { [void]$candidates.Add($dir.FullName) }
    }
    foreach ($dir in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $dir 'bin\nvcc.exe')) { return $dir }
    }
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return (Split-Path -Parent (Split-Path -Parent $cmd.Source)) }
    return $null
}

function Get-NvidiaGpuList {
    $result = New-Object System.Collections.Generic.List[object]
    $smi = Find-FirstFile @(
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    )
    if (-not $smi) {
        $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($cmd) { $smi = $cmd.Source }
    }
    if (-not $smi) { return ,$result }
    $raw = @(& $smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader)
    if ($LASTEXITCODE -ne 0) { return ,$result }
    foreach ($line in $raw) {
        if ($line -match '^\s*(.*?)\s*,\s*([0-9]+\.[0-9])\s*,\s*(.+?)\s*$') {
            $arch = Convert-ComputeCapability $Matches[2]
            if ($arch) {
                [void]$result.Add([pscustomobject]@{
                    Name = $Matches[1]
                    Compute = $Matches[2]
                    Memory = $Matches[3]
                    Arch = $arch
                })
            }
        }
    }
    return ,$result
}

function Invoke-Checked {
    param([string]$File, [string[]]$ArgumentList)
    Write-Host ("    " + $File + " " + ($ArgumentList -join ' '))
    & $File @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed (code " + $LASTEXITCODE + "): " + $File)
    }
}

function Write-Utf8File([string]$Path, [string]$Content) {
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

function Get-LaunchExecutable([string]$Root) {
    $marker = Join-Path $Root 'build\waffle-launch.txt'
    if (Test-Path -LiteralPath $marker) {
        $line = Get-Content -LiteralPath $marker -TotalCount 1
        if ($line) {
            $full = Join-Path $Root ([string]$line).Trim()
            if (Test-Path -LiteralPath $full) { return $full }
        }
    }
    $best = $null
    $bestTime = $null
    foreach ($relative in @(
        'build\cuda-release\WaffleIron.exe',
        'build\cpu-release\WaffleIron.exe',
        'build\cuda-release-local\WaffleIron.exe'
    )) {
        $full = Join-Path $Root $relative
        if (Test-Path -LiteralPath $full) {
            $time = (Get-Item -LiteralPath $full).LastWriteTimeUtc
            if (-not $best -or $time -gt $bestTime) {
                $best = $full
                $bestTime = $time
            }
        }
    }
    return $best
}

function Start-WaffleGui([string]$Exe) {
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) {
        throw "WaffleIron.exe was not found. Run install.cmd from the repository root first."
    }
    Start-Process -FilePath $Exe -WorkingDirectory (Split-Path -Parent $Exe) | Out-Null
    Write-Ok ("Interface started: " + $Exe)
}

function Start-InstallLog([string]$Root) {
    $dir = Join-Path $Root 'build'
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    try {
        Start-Transcript -Path (Join-Path $dir 'install.log') -Force | Out-Null
        $script:TranscriptOn = $true
    } catch {
        Write-Warn ("Log unavailable: " + $_.Exception.Message)
    }
}

function Stop-InstallLog {
    if ($script:TranscriptOn) {
        try { Stop-Transcript | Out-Null } catch {}
        $script:TranscriptOn = $false
    }
}

function Show-Problems($Problems) {
    Write-Host ""
    Write-Host "Installation cannot continue: prerequisites are missing." -ForegroundColor Red
    Write-Host ""
    foreach ($problem in $Problems) {
        Write-Host $problem
        Write-Host ""
    }
    Write-Host "Fix the items above, then run install.cmd again."
    throw "Prerequisites are incomplete."
}

function Start-WaffleInstall {
    $root = Split-Path -Parent $PSScriptRoot
    try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch {}
    $OutputEncoding = [Console]::OutputEncoding
    if (-not $Launch) { Start-InstallLog $root }
    Write-Step "Waffle Iron"
    Write-Host ("    Folder: " + $root)

    if ($Launch) {
        Start-WaffleGui (Get-LaunchExecutable $root)
        return
    }

    $problems = New-Object System.Collections.Generic.List[string]
    $git = Find-FirstFile @(
        (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Git\cmd\git.exe'),
        (Join-Path $env:LocalAppData 'Programs\Git\cmd\git.exe')
    )
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git -and $gitCmd) { $git = $gitCmd.Source }
    if (-not $git) {
        [void]$problems.Add(@"
Git was not found.
It is used to fetch llama.cpp (the submodule) and by vcpkg.
Install Git for Windows: https://git-scm.com/download/win
Then open a new terminal and run install.cmd again.
"@.Trim())
    } else {
        Add-ToolDirectory $git
    }

    $vsRoot = Get-VisualStudioRoot
    if (-not $vsRoot) {
        [void]$problems.Add(@"
Visual Studio with the C++ tools (MSVC x64) was not found.
Install Visual Studio 2022 or newer (2026 is fine):
  https://visualstudio.microsoft.com/downloads/
In the installer, select the "Desktop development with C++" workload.
It must include the MSVC tools, "C++ CMake tools for Windows" (CMake and Ninja),
and "vcpkg package manager". The Build Tools alone are enough too.
"@.Trim())
    } else {
        Write-Ok ("Visual Studio : " + $vsRoot)
        Import-VcEnvironment (Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat')
    }

    if ($vsRoot) {
        $cl = Get-Command cl -ErrorAction SilentlyContinue
        if (-not $cl) {
            [void]$problems.Add(@"
The MSVC compiler (cl.exe) was not found after initializing Visual Studio.
Modify the installation and select the MSVC C++ tools (x64/x86).
"@.Trim())
        }
        $cmake = $null
        $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
        if ($cmakeCmd) { $cmake = $cmakeCmd.Source }
        if (-not $cmake) {
            $cmake = Find-FirstFile @(
                (Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'),
                (Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe')
            )
            if ($cmake) { Add-ToolDirectory $cmake }
        }
        if (-not $cmake) {
            [void]$problems.Add(@"
CMake 3.25 or newer was not found.
Select "C++ CMake tools for Windows" in the Visual Studio Installer,
or install CMake: https://cmake.org/download/
Then run install.cmd again.
"@.Trim())
        } else {
            $cmakeVersion = Parse-CMakeVersion ((@(& $cmake --version)) -join "`n")
            if (-not $cmakeVersion -or $cmakeVersion -lt [version]'3.25.0') {
                $shown = if ($cmakeVersion) { [string]$cmakeVersion } else { "unknown" }
                [void]$problems.Add(("CMake " + $shown + " is too old. Version 3.25 or newer is required (presets).`r`nhttps://cmake.org/download/"))
            } else {
                Write-Ok ("CMake " + $cmakeVersion)
            }
        }
        $ninja = $null
        $ninjaCmd = Get-Command ninja -ErrorAction SilentlyContinue
        if ($ninjaCmd) { $ninja = $ninjaCmd.Source }
        if (-not $ninja) {
            $ninja = Find-FirstFile @(
                (Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe')
            )
            if ($ninja) { Add-ToolDirectory $ninja }
        }
        if (-not $ninja) {
            [void]$problems.Add(@"
Ninja was not found.
It ships with "C++ CMake tools for Windows" in Visual Studio,
or separately: https://ninja-build.org/
"@.Trim())
        } else {
            Write-Ok ("Ninja : " + $ninja)
        }
        $vcpkgRoot = $null
        if ($env:VCPKG_ROOT -and (Test-Path -LiteralPath (Join-Path $env:VCPKG_ROOT 'vcpkg.exe'))) {
            $vcpkgRoot = $env:VCPKG_ROOT
        }
        if (-not $vcpkgRoot) {
            $bundled = Join-Path $vsRoot 'VC\vcpkg'
            if (Test-Path -LiteralPath (Join-Path $bundled 'vcpkg.exe')) { $vcpkgRoot = $bundled }
        }
        if (-not $vcpkgRoot) {
            $vcpkgCmd = Get-Command vcpkg -ErrorAction SilentlyContinue
            if ($vcpkgCmd) { $vcpkgRoot = Split-Path -Parent $vcpkgCmd.Source }
        }
        $toolchain = $null
        if ($vcpkgRoot) { $toolchain = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake' }
        if (-not $toolchain -or -not (Test-Path -LiteralPath $toolchain)) {
            [void]$problems.Add(@"
vcpkg was not found (VCPKG_ROOT, or VC\vcpkg next to Visual Studio).
Option 1: Visual Studio Installer, Modify, "Desktop development with C++" workload,
          select "vcpkg package manager".
Option 2: standalone install
          git clone https://github.com/microsoft/vcpkg C:\vcpkg
          C:\vcpkg\bootstrap-vcpkg.bat
          set VCPKG_ROOT=C:\vcpkg
Then run install.cmd again.
"@.Trim())
        } else {
            $env:VCPKG_ROOT = $vcpkgRoot
            Write-Ok ("vcpkg : " + $vcpkgRoot)
        }
    }

    if ($problems.Count -gt 0) { Show-Problems $problems }

    $llama = Join-Path $root 'external\llama.cpp\CMakeLists.txt'
    if (-not (Test-Path -LiteralPath $llama)) {
        if (-not (Test-Path -LiteralPath (Join-Path $root '.git'))) {
            throw "This folder is not a Git clone. Use git clone --recurse-submodules, not a zip archive: the external/llama.cpp submodule would be missing."
        }
        Write-Step "Fetching the llama.cpp submodule (internet access)"
        $env:GIT_TERMINAL_PROMPT = '0'
        Invoke-Checked $git @('-C', $root, 'submodule', 'update', '--init', '--recursive')
        if (-not (Test-Path -LiteralPath $llama)) {
            throw "The external/llama.cpp submodule is still missing after git submodule update."
        }
    } else {
        Write-Ok "llama.cpp submodule is present."
    }

    $cudaRoot = Find-CudaToolkitRoot
    if ($cudaRoot) {
        $env:CUDA_PATH = $cudaRoot
        Add-ToolDirectory (Join-Path $cudaRoot 'bin')
        Write-Ok ("CUDA Toolkit : " + $cudaRoot)
    }
    $gpus = Get-NvidiaGpuList
    $detected = New-Object System.Collections.Generic.List[string]
    $labels = New-Object System.Collections.Generic.List[string]
    foreach ($gpu in $gpus) {
        [void]$detected.Add([string]$gpu.Arch)
        [void]$labels.Add([string]$gpu.Name + " (sm_" + $gpu.Arch + ", " + $gpu.Memory + ")")
        Write-Ok ("GPU : " + $gpu.Name + "  compute " + $gpu.Compute + "  " + $gpu.Memory)
    }
    if ($gpus.Count -eq 0) { Write-Warn "No NVIDIA GPU detected (nvidia-smi)." }

    $plan = Select-InstallPlan -ForceCpu:([bool]$Cpu) -ForceCuda:([bool]$Cuda) -RequestedArchitecture $Architecture -HasCudaToolkit:([bool]$cudaRoot) -DetectedArchitectures @($detected.ToArray()) -GpuLabels @($labels.ToArray())
    if (-not $plan.Ok) { throw $plan.Error }
    Write-Step "Selected configuration"
    Write-Host ("    " + $plan.Note)
    if ($plan.Kind -eq 'cuda') {
        $nvccOut = ((@(& nvcc --version)) -join "`n")
        $nvccVersion = Parse-NvccVersion $nvccOut
        $archForWarning = $plan.ArchitectureText
        if (-not $archForWarning) { $archForWarning = '89' }
        $warning = Get-CudaToolkitWarning $nvccVersion $archForWarning
        if ($warning) { Write-Warn $warning }
        if (-not $cudaRoot) { Write-Warn "A CUDA build was requested, but nvcc was not found before configuration." }
    }

    $configurePreset = $plan.Preset
    $cmakeArgs = @('--preset', $configurePreset)
    if ($plan.WriteUserPreset) {
        $userPresetPath = Join-Path $root 'CMakeUserPresets.json'
        $foreign = $false
        if (Test-Path -LiteralPath $userPresetPath) {
            $existing = Get-Content -LiteralPath $userPresetPath -Raw
            if ($existing -notlike '*waffle.iron/install/1.0*') { $foreign = $true }
        }
        if ($foreign) {
            Write-Warn "CMakeUserPresets.json exists and was not created by install.cmd: it is left unchanged."
            $archFile = Join-Path $root 'build\waffle-cuda-arch.cmake'
            Write-Utf8File $archFile ("set(CMAKE_CUDA_ARCHITECTURES `"" + $plan.ArchitectureText + "`" CACHE STRING `"Chosen by install.cmd`" FORCE)`r`n")
            $include = (Resolve-Path -LiteralPath $archFile).Path.Replace('\', '/')
            if ($include.Contains(';')) {
                throw ("The repository path contains a semicolon, which cannot be used in a CMake list: " + $include)
            }
            $configurePreset = 'cuda-release'
            $plan.ExeRelative = 'build\cuda-release\WaffleIron.exe'
            $cmakeArgs = @('--preset', 'cuda-release', ('-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=' + $include))
        } else {
            $summary = ([regex]::Replace($plan.Note, '\s+', ' ')).Trim()
            if ($summary.Length -gt 140) { $summary = $summary.Substring(0, 140) }
            Write-Utf8File $userPresetPath (New-CudaUserPresetJson -ArchitectureText $plan.ArchitectureText -Summary $summary)
            Write-Ok ("Local preset: CMakeUserPresets.json (sm_" + (($plan.ArchitectureText -split ';') -join ', sm_') + ").")
        }
    }

    Write-Host ("    Preset : " + $configurePreset)
    Write-Host ("    Executable: " + $plan.ExeRelative)
    Push-Location $root
    try {
        Write-Step "CMake configure (the first run downloads vcpkg dependencies; internet required)"
        Invoke-Checked $cmake $cmakeArgs
        if ($plan.Kind -eq 'cuda') {
            Write-Step "CUDA build. It can take a long time. If it stops, run install.cmd again: it resumes."
        } else {
            Write-Step "Build"
        }
        Invoke-Checked $cmake @('--build', '--preset', $configurePreset)
    } finally {
        Pop-Location
    }

    $exe = Join-Path $root $plan.ExeRelative
    if (-not (Test-Path -LiteralPath $exe)) {
        throw ("The build finished without producing " + $exe)
    }
    Write-Utf8File (Join-Path $root 'build\waffle-launch.txt') ($plan.ExeRelative + "`r`n")

    if ($Tests) {
        Write-Step "Tests"
        $ctest = $null
        $ctestCmd = Get-Command ctest -ErrorAction SilentlyContinue
        if ($ctestCmd) { $ctest = $ctestCmd.Source }
        if (-not $ctest) { $ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe' }
        Invoke-Checked $ctest @('--test-dir', (Split-Path -Parent $exe), '--output-on-failure')
    }

    if ($NoLaunch) {
        Write-Ok "Build finished. Open the interface with launch.cmd."
    } else {
        Write-Step "Opening the interface"
        Start-WaffleGui $exe
        Write-Host "    On first display, the library scans your model folders."
        Write-Host "    To open it again later: launch.cmd"
    }
    Write-Ok "Log: build\install.log"
}

try {
    if ($AutoTest) {
        Invoke-InstallSelfTest
        exit 0
    }
    if ($Help) {
        Show-InstallHelp
        exit 0
    }
    if ($env:OS -ne 'Windows_NT') {
        throw "Waffle Iron builds on Windows only."
    }
    Start-WaffleInstall
} catch {
    Write-Host ""
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    Stop-InstallLog
}
