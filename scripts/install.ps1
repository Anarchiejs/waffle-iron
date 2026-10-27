#Requires -Version 5.1
<#
.SYNOPSIS
  Installe Banc Local sous Windows et ouvre l'interface.
.DESCRIPTION
  Point d'entrée de install.cmd et lancer.cmd. Windows PowerShell 5.1.
  -AutoTest vérifie la logique de choix CPU/CUDA sans rien compiler.
#>
param(
    [switch]$Aide,
    [switch]$Lancer,
    [switch]$SansLancement,
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
                Error = ("Architecture CUDA invalide : " + $part + ". Exemples : 89 ou 8.9.")
            }
        }
        if (-not $list.Contains($arch)) { [void]$list.Add($arch) }
    }
    if ($list.Count -eq 0) {
        return [pscustomobject]@{ Ok = $false; ArchitectureText = ''; Error = "Architecture CUDA vide." }
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
        return ("sm_" + $max + " demande un CUDA Toolkit 12.8 ou plus récent (détecté : " + $Toolkit + ").")
    }
    if ($max -ge 89 -and $Toolkit -lt [version]'11.8') {
        return ("sm_" + $max + " demande un CUDA Toolkit 11.8 ou plus récent (détecté : " + $Toolkit + ").")
    }
    if ($max -ge 80 -and $Toolkit -lt [version]'11.1') {
        return ("sm_" + $max + " demande un CUDA Toolkit 11.1 ou plus récent (détecté : " + $Toolkit + ").")
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
    $exe = 'build\cpu-release\BancLocal.exe'
    if ($Preset) { $exe = 'build\' + $Preset + '\BancLocal.exe' }
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
        return (New-Plan $false "Choisissez un seul mode : -Cpu ou -Cuda." '' '' '' '' $false)
    }
    $requested = Parse-ArchitectureRequest $RequestedArchitecture
    if (-not $requested.Ok) {
        return (New-Plan $false $requested.Error '' '' '' '' $false)
    }
    if ($ForceCpu -and $requested.ArchitectureText) {
        return (New-Plan $false "-Architecture ne s'utilise pas avec -Cpu." '' '' '' '' $false)
    }
    if (($ForceCuda -or $requested.ArchitectureText) -and -not $HasCudaToolkit) {
        $why = "Installez le CUDA Toolkit, puis relancez install.cmd :`r`nhttps://developer.nvidia.com/cuda-downloads"
        if ($ForceCuda) {
            $why = "L'option -Cuda demande le CUDA Toolkit (nvcc), qui est introuvable.`r`n" + $why
        } else {
            $why = "Une architecture CUDA est demandée, mais nvcc est introuvable.`r`n" + $why
        }
        return (New-Plan $false $why '' '' '' '' $false)
    }
    $detectedText = Format-Architectures $DetectedArchitectures
    $archText = $requested.ArchitectureText
    if (-not $archText) { $archText = $detectedText }
    $labels = @($GpuLabels | Where-Object { $_ })
    $labelText = $labels -join ', '

    if ($ForceCpu) {
        return (New-Plan $true '' 'cpu' 'cpu-release' '' "Compilation CPU (option -Cpu)." $false)
    }
    if ($HasCudaToolkit -and ($ForceCuda -or $archText)) {
        if ($archText) {
            $note = "Compilation CUDA pour sm_" + (($archText -split ';') -join ', sm_') + "."
            if ($labelText) { $note += " Carte : " + $labelText + "." }
            if ($requested.ArchitectureText -and $detectedText -and $requested.ArchitectureText -ne $detectedText) {
                $note += " L'option -Architecture remplace la détection (" + $detectedText + ")."
            }
            return (New-Plan $true '' 'cuda' 'cuda-release-local' $archText $note $true)
        }
        $note = "Compilation CUDA avec l'architecture du preset cuda-release (sm_89) : la carte n'a pas été identifiée. Utilisez -Architecture 86 pour en imposer une autre."
        return (New-Plan $true '' 'cuda' 'cuda-release' '' $note $false)
    }
    if ($labelText) {
        $note = "Carte NVIDIA détectée (" + $labelText + "), mais le CUDA Toolkit est absent : compilation CPU. Installez le CUDA Toolkit puis relancez install.cmd pour un build GPU.`r`nhttps://developer.nvidia.com/cuda-downloads"
        return (New-Plan $true '' 'cpu' 'cpu-release' '' $note $false)
    }
    if ($HasCudaToolkit) {
        $note = "CUDA Toolkit présent, aucune carte NVIDIA détectée : compilation CPU. Pour forcer le preset cuda-release (sm_89), relancez avec -Cuda."
        return (New-Plan $true '' 'cpu' 'cpu-release' '' $note $false)
    }
    return (New-Plan $true '' 'cpu' 'cpu-release' '' "Aucune carte NVIDIA ni CUDA Toolkit : compilation CPU." $false)
}

function New-CudaUserPresetJson {
    param([string]$ArchitectureText, [string]$Summary)
    $arch = ConvertTo-JsonString $ArchitectureText
    $display = ConvertTo-JsonString ("CUDA, Release (install.cmd : " + $Summary + ")")
    $lines = @(
        '{',
        '  "version": 6,',
        '  "vendor": {',
        '    "banc.local/install/1.0": { "generated": true }',
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
        Write-Host ("ECHEC " + $Name + " : obtenu [" + $left + "] attendu [" + $right + "]") -ForegroundColor Red
        $script:TestFailures++
    }
}

function Assert-True {
    param([string]$Name, $Condition)
    if (-not $Condition) {
        Write-Host ("ECHEC " + $Name) -ForegroundColor Red
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
    Assert-Equal 'cap vide' (Convert-ComputeCapability '') $null
    Assert-Equal 'cap sm' (Convert-ComputeCapability 'sm_89') $null

    $none = Select-InstallPlan $false $false '' $false @() @()
    Assert-Plan 'rien' $none $true 'cpu' 'cpu-release' '' '*Aucune carte NVIDIA*'

    $gpuOnly = Select-InstallPlan $false $false '' $false @('89') @('RTX 4060 Ti (sm_89)')
    Assert-Plan 'gpu sans toolkit' $gpuOnly $true 'cpu' 'cpu-release' '' '*CUDA Toolkit*'

    $cuda = Select-InstallPlan $false $false '' $true @('89') @('RTX 4060 Ti (sm_89)')
    Assert-Plan 'cuda 89' $cuda $true 'cuda' 'cuda-release-local' '89' '*sm_89*'
    Assert-Equal 'cuda write' ([string]$cuda.WriteUserPreset) ([string]$true)
    Assert-Equal 'cuda exe' $cuda.ExeRelative 'build\cuda-release-local\BancLocal.exe'

    $multi = Select-InstallPlan $false $false '' $true @('89', '86') @('A', 'B')
    Assert-Plan 'multi' $multi $true 'cuda' 'cuda-release-local' '86;89' '*sm_86, sm_89*'

    $forceCpu = Select-InstallPlan $true $false '' $true @('89') @('RTX')
    Assert-Plan 'force cpu' $forceCpu $true 'cpu' 'cpu-release' '' '*-Cpu*'

    $forceCudaMissing = Select-InstallPlan $false $true '' $false @() @()
    Assert-Equal 'force cuda sans toolkit' ([string]$forceCudaMissing.Ok) ([string]$false)
    Assert-True 'force cuda message' ($forceCudaMissing.Error -like '*nvcc*')

    $forceCudaBlind = Select-InstallPlan $false $true '' $true @() @()
    Assert-Plan 'force cuda aveugle' $forceCudaBlind $true 'cuda' 'cuda-release' '' '*sm_89*'
    Assert-Equal 'force cuda write' ([string]$forceCudaBlind.WriteUserPreset) ([string]$false)

    $requested = Select-InstallPlan $false $false '8.9' $true @() @()
    Assert-Plan 'arch 8.9' $requested $true 'cuda' 'cuda-release-local' '89' '*sm_89*'

    $requestedMany = Select-InstallPlan $false $false '120,89' $true @('86') @('ancienne')
    Assert-Plan 'arch liste' $requestedMany $true 'cuda' 'cuda-release-local' '89;120' '*remplace*'

    $dup = Select-InstallPlan $false $false '89,89' $true @() @()
    Assert-Plan 'arch doublon' $dup $true 'cuda' 'cuda-release-local' '89' '*sm_89*'

    $bad = Select-InstallPlan $false $false 'sm_89' $true @() @()
    Assert-Equal 'arch invalide' ([string]$bad.Ok) ([string]$false)

    $both = Select-InstallPlan $true $true '' $true @() @()
    Assert-Equal 'cpu et cuda' ([string]$both.Ok) ([string]$false)

    $archCpu = Select-InstallPlan $true $false '86' $true @() @()
    Assert-Equal 'arch et cpu' ([string]$archCpu.Ok) ([string]$false)

    $toolkitOnly = Select-InstallPlan $false $false '' $true @() @()
    Assert-Plan 'toolkit seul' $toolkitOnly $true 'cpu' 'cpu-release' '' '*sm_89*'

    $json = New-CudaUserPresetJson -ArchitectureText '86;89' -Summary 'RTX "4060"'
    $parsed = $json | ConvertFrom-Json
    Assert-Equal 'json nom' $parsed.configurePresets[0].name 'cuda-release-local'
    Assert-Equal 'json arch' $parsed.configurePresets[0].cacheVariables.CMAKE_CUDA_ARCHITECTURES '86;89'
    Assert-Equal 'json bin absent' $parsed.configurePresets[0].PSObject.Properties['binaryDir'] $null
    Assert-Equal 'json herite' $parsed.configurePresets[0].inherits 'cuda-release'
    Assert-True 'json guillemet' ($parsed.configurePresets[0].displayName -like '*RTX "4060"*')
    $vendor = $parsed.vendor.PSObject.Properties['banc.local/install/1.0'].Value
    Assert-True 'json vendor' ($vendor.generated -eq $true)
    Assert-True 'json marqueur' ($json.Contains('banc.local/install/1.0'))

    Assert-Equal 'cmake 3.25' (Parse-CMakeVersion "cmake version 3.25.1`n") ([version]'3.25.1')
    Assert-True 'cmake ancien' ((Parse-CMakeVersion 'cmake version 3.24.0') -lt [version]'3.25.0')
    Assert-Equal 'nvcc' (Parse-NvccVersion "Cuda compilation tools, release 12.6, V12.6.77") ([version]'12.6')
    Assert-True 'warn 120' ((Get-CudaToolkitWarning ([version]'12.6') '120') -like '*12.8*')
    Assert-Equal 'warn 89 recent' (Get-CudaToolkitWarning ([version]'12.6') '89') $null
    Assert-True 'warn 89 vieux' ((Get-CudaToolkitWarning ([version]'11.7') '89') -like '*11.8*')
    Assert-True 'warn 86' ((Get-CudaToolkitWarning ([version]'11.0') '86') -like '*11.1*')
    Assert-Equal 'warn 75' (Get-CudaToolkitWarning ([version]'11.0') '75') $null

    if ($script:TestFailures -gt 0) {
        throw ("Autotest : " + $script:TestFailures + " échec(s).")
    }
    Write-Host "Autotest réussi."
}

function Show-InstallHelp {
    Write-Host @"
Banc Local - installation et lancement

  install.cmd
      Vérifie les prérequis, récupère llama.cpp si besoin, choisit CPU ou CUDA,
      compile, puis ouvre l'interface.

  install.cmd -SansLancement
      Compile sans ouvrir l'interface.

  install.cmd -Cpu
      Force la compilation CPU (preset cpu-release).

  install.cmd -Cuda
      Force la compilation CUDA. Sans carte identifiée, l'architecture reste sm_89.

  install.cmd -Architecture 86
      Impose l'architecture CUDA (86, 89, 120...). 8.9 est accepté et devient 89.
      Plusieurs valeurs : -Architecture 86,89

  install.cmd -Tests
      Lance les tests après la compilation.

  install.cmd -Aide
      Affiche cette aide.

  lancer.cmd
      Rouvre BancLocal.exe déjà compilé, sans recompiler.

La première compilation CUDA est longue. Relancer install.cmd reprend où elle s'est arrêtée.
Journal : build\install.log
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
    $env:BANC_VCVARS = $VcVarsPath
    $batch = Join-Path $env:TEMP ("banc-vcvars-" + $PID + ".cmd")
    $content = "@echo off`r`ncall `"%BANC_VCVARS%`" >nul`r`nif errorlevel 1 exit /b 1`r`nchcp 65001 >nul`r`nset`r`n"
    [System.IO.File]::WriteAllText($batch, $content, [System.Text.Encoding]::ASCII)
    $previous = [Console]::OutputEncoding
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    try {
        $lines = & cmd.exe /d /c "`"$batch`""
        $code = $LASTEXITCODE
    } finally {
        [Console]::OutputEncoding = $previous
        Remove-Item -LiteralPath $batch -Force -ErrorAction SilentlyContinue
        Remove-Item Env:BANC_VCVARS -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) {
        throw ("Impossible d'initialiser Visual Studio (vcvars64.bat, code " + $code + ").")
    }
    foreach ($line in @($lines)) {
        if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            if ($Matches[1] -eq 'BANC_VCVARS') { continue }
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
        throw ("Commande échouée (code " + $LASTEXITCODE + ") : " + $File)
    }
}

function Write-Utf8File([string]$Path, [string]$Content) {
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

function Get-LaunchExecutable([string]$Root) {
    $marker = Join-Path $Root 'build\banc-launch.txt'
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
        'build\cuda-release\BancLocal.exe',
        'build\cpu-release\BancLocal.exe',
        'build\cuda-release-local\BancLocal.exe'
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

function Start-BancGui([string]$Exe) {
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) {
        throw "BancLocal.exe est introuvable. Lancez d'abord install.cmd à la racine du dépôt."
    }
    Start-Process -FilePath $Exe -WorkingDirectory (Split-Path -Parent $Exe) | Out-Null
    Write-Ok ("Interface lancée : " + $Exe)
}

function Start-InstallLog([string]$Root) {
    $dir = Join-Path $Root 'build'
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    try {
        Start-Transcript -Path (Join-Path $dir 'install.log') -Force | Out-Null
        $script:TranscriptOn = $true
    } catch {
        Write-Warn ("Journal indisponible : " + $_.Exception.Message)
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
    Write-Host "Installation impossible : des prérequis manquent." -ForegroundColor Red
    Write-Host ""
    foreach ($problem in $Problems) {
        Write-Host $problem
        Write-Host ""
    }
    Write-Host "Corrigez les points ci-dessus, puis relancez install.cmd."
    throw "Prérequis incomplets."
}

function Start-BancInstall {
    $root = Split-Path -Parent $PSScriptRoot
    try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch {}
    $OutputEncoding = [Console]::OutputEncoding
    if (-not $Lancer) { Start-InstallLog $root }
    Write-Step "Banc Local"
    Write-Host ("    Dossier : " + $root)

    if ($Lancer) {
        Start-BancGui (Get-LaunchExecutable $root)
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
Git est introuvable.
Il sert à récupérer llama.cpp (sous-module) et à vcpkg.
Installez Git pour Windows : https://git-scm.com/download/win
Puis ouvrez un nouveau terminal et relancez install.cmd.
"@.Trim())
    } else {
        Add-ToolDirectory $git
    }

    $vsRoot = Get-VisualStudioRoot
    if (-not $vsRoot) {
        [void]$problems.Add(@"
Visual Studio avec les outils C++ (MSVC x64) est introuvable.
Installez Visual Studio 2022 ou plus récent (2026 convient) :
  https://visualstudio.microsoft.com/fr/downloads/
Dans l'installateur, cochez la charge de travail « Développement Desktop en C++ ».
Elle doit inclure les outils MSVC, « Outils CMake C++ pour Windows » (CMake et Ninja)
et « Gestionnaire de package vcpkg ». Les Build Tools seuls conviennent aussi.
"@.Trim())
    } else {
        Write-Ok ("Visual Studio : " + $vsRoot)
        Import-VcEnvironment (Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat')
    }

    if ($vsRoot) {
        $cl = Get-Command cl -ErrorAction SilentlyContinue
        if (-not $cl) {
            [void]$problems.Add(@"
Le compilateur MSVC (cl.exe) est introuvable après l'initialisation de Visual Studio.
Modifiez l'installation et cochez les outils MSVC C++ (x64/x86).
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
CMake 3.25 ou plus récent est introuvable.
Cochez « Outils CMake C++ pour Windows » dans Visual Studio Installer,
ou installez CMake : https://cmake.org/download/
Puis relancez install.cmd.
"@.Trim())
        } else {
            $cmakeVersion = Parse-CMakeVersion ((@(& $cmake --version)) -join "`n")
            if (-not $cmakeVersion -or $cmakeVersion -lt [version]'3.25.0') {
                $shown = if ($cmakeVersion) { [string]$cmakeVersion } else { "inconnue" }
                [void]$problems.Add(("CMake " + $shown + " est trop ancien. Il faut la version 3.25 ou plus (presets).`r`nhttps://cmake.org/download/"))
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
Ninja est introuvable.
Il est fourni avec « Outils CMake C++ pour Windows » dans Visual Studio,
ou séparément : https://ninja-build.org/
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
vcpkg est introuvable (VCPKG_ROOT, ou VC\vcpkg à côté de Visual Studio).
Option 1 : Visual Studio Installer, Modifier, charge « Développement Desktop en C++ »,
           cocher « Gestionnaire de package vcpkg ».
Option 2 : installation autonome
           git clone https://github.com/microsoft/vcpkg C:\vcpkg
           C:\vcpkg\bootstrap-vcpkg.bat
           set VCPKG_ROOT=C:\vcpkg
Puis relancez install.cmd.
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
            throw "Ce dossier n'est pas un clone Git. Utilisez git clone --recurse-submodules, pas une archive zip : le sous-module external/llama.cpp manquerait."
        }
        Write-Step "Récupération du sous-module llama.cpp (accès Internet)"
        $env:GIT_TERMINAL_PROMPT = '0'
        Invoke-Checked $git @('-C', $root, 'submodule', 'update', '--init', '--recursive')
        if (-not (Test-Path -LiteralPath $llama)) {
            throw "Le sous-module external/llama.cpp est toujours absent après git submodule update."
        }
    } else {
        Write-Ok "Sous-module llama.cpp présent."
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
    if ($gpus.Count -eq 0) { Write-Warn "Aucune carte NVIDIA détectée (nvidia-smi)." }

    $plan = Select-InstallPlan -ForceCpu:([bool]$Cpu) -ForceCuda:([bool]$Cuda) -RequestedArchitecture $Architecture -HasCudaToolkit:([bool]$cudaRoot) -DetectedArchitectures @($detected.ToArray()) -GpuLabels @($labels.ToArray())
    if (-not $plan.Ok) { throw $plan.Error }
    Write-Step "Configuration retenue"
    Write-Host ("    " + $plan.Note)
    if ($plan.Kind -eq 'cuda') {
        $nvccOut = ((@(& nvcc --version)) -join "`n")
        $nvccVersion = Parse-NvccVersion $nvccOut
        $archForWarning = $plan.ArchitectureText
        if (-not $archForWarning) { $archForWarning = '89' }
        $warning = Get-CudaToolkitWarning $nvccVersion $archForWarning
        if ($warning) { Write-Warn $warning }
        if (-not $cudaRoot) { Write-Warn "Le build CUDA est demandé alors que nvcc n'a pas été trouvé avant la configuration." }
    }

    $configurePreset = $plan.Preset
    $cmakeArgs = @('--preset', $configurePreset)
    if ($plan.WriteUserPreset) {
        $userPresetPath = Join-Path $root 'CMakeUserPresets.json'
        $foreign = $false
        if (Test-Path -LiteralPath $userPresetPath) {
            $existing = Get-Content -LiteralPath $userPresetPath -Raw
            if ($existing -notlike '*banc.local/install/1.0*') { $foreign = $true }
        }
        if ($foreign) {
            Write-Warn "CMakeUserPresets.json existe et n'a pas été créé par install.cmd : il est laissé intact."
            $archFile = Join-Path $root 'build\banc-cuda-arch.cmake'
            Write-Utf8File $archFile ("set(CMAKE_CUDA_ARCHITECTURES `"" + $plan.ArchitectureText + "`" CACHE STRING `"Choisi par install.cmd`" FORCE)`r`n")
            $include = (Resolve-Path -LiteralPath $archFile).Path.Replace('\', '/')
            if ($include.Contains(';')) {
                throw ("Le chemin du dépôt contient un point-virgule, inutilisable dans une liste CMake : " + $include)
            }
            $configurePreset = 'cuda-release'
            $plan.ExeRelative = 'build\cuda-release\BancLocal.exe'
            $cmakeArgs = @('--preset', 'cuda-release', ('-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=' + $include))
        } else {
            $summary = ([regex]::Replace($plan.Note, '\s+', ' ')).Trim()
            if ($summary.Length -gt 140) { $summary = $summary.Substring(0, 140) }
            Write-Utf8File $userPresetPath (New-CudaUserPresetJson -ArchitectureText $plan.ArchitectureText -Summary $summary)
            Write-Ok ("Preset local : CMakeUserPresets.json (sm_" + (($plan.ArchitectureText -split ';') -join ', sm_') + ").")
        }
    }

    Write-Host ("    Preset : " + $configurePreset)
    Write-Host ("    Exécutable : " + $plan.ExeRelative)
    Push-Location $root
    try {
        Write-Step "Configuration CMake (le premier passage télécharge les dépendances vcpkg, Internet requis)"
        Invoke-Checked $cmake $cmakeArgs
        if ($plan.Kind -eq 'cuda') {
            Write-Step "Compilation CUDA. Elle peut être longue. En cas d'arrêt, relancez install.cmd : elle reprend."
        } else {
            Write-Step "Compilation"
        }
        Invoke-Checked $cmake @('--build', '--preset', $configurePreset)
    } finally {
        Pop-Location
    }

    $exe = Join-Path $root $plan.ExeRelative
    if (-not (Test-Path -LiteralPath $exe)) {
        throw ("La compilation s'est terminée sans produire " + $exe)
    }
    Write-Utf8File (Join-Path $root 'build\banc-launch.txt') ($plan.ExeRelative + "`r`n")

    if ($Tests) {
        Write-Step "Tests"
        $ctest = $null
        $ctestCmd = Get-Command ctest -ErrorAction SilentlyContinue
        if ($ctestCmd) { $ctest = $ctestCmd.Source }
        if (-not $ctest) { $ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe' }
        Invoke-Checked $ctest @('--test-dir', (Split-Path -Parent $exe), '--output-on-failure')
    }

    if ($SansLancement) {
        Write-Ok "Compilation terminée. Ouvrez l'interface avec lancer.cmd."
    } else {
        Write-Step "Ouverture de l'interface"
        Start-BancGui $exe
        Write-Host "    Au premier affichage, la bibliothèque parcourt vos dossiers de modèles."
        Write-Host "    Pour rouvrir plus tard : lancer.cmd"
    }
    Write-Ok "Journal : build\install.log"
}

try {
    if ($AutoTest) {
        Invoke-InstallSelfTest
        exit 0
    }
    if ($Aide) {
        Show-InstallHelp
        exit 0
    }
    if ($env:OS -ne 'Windows_NT') {
        throw "Banc Local se construit uniquement sous Windows."
    }
    Start-BancInstall
} catch {
    Write-Host ""
    Write-Host ("ERREUR : " + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    Stop-InstallLog
}
