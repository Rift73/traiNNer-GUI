[CmdletBinding()]
param (
    [string]$BuildDirectory = (Join-Path $env:LOCALAPPDATA 'traiNNer-GUI\build\launcher'),
    [string]$Destination = $PSScriptRoot,
    [ValidatePattern('^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-preview\.([1-9][0-9]*))?$')]
    [string]$ReleaseTag = 'v0.1.0-preview.3',
    [switch]$Package
)
$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'native\studio\launcher'
$BuildDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BuildDirectory)
$Destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)
if ($Package -and $Destination.TrimEnd('\') -eq $PSScriptRoot.TrimEnd('\')) {
    throw 'Use a separate -Destination for the Windows package.'
}

function Invoke-PackageTool {
    param([string]$Operation, [string[]]$Paths)

    if ($PSScriptRoot -match '^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)\\(.+)$') {
        $distro = $Matches[1]
        $root = '/' + $Matches[2].Replace('\', '/')
        $linuxPaths = @(foreach ($path in $Paths) {
            $converted = & wsl.exe --distribution $distro --cd / --exec wslpath -a -u $path
            if ($LASTEXITCODE -ne 0 -or !$converted) { throw "WSL cannot open $path." }
            $converted
        })
        & wsl.exe --distribution $distro --cd $root --exec python3 scripts/package-windows.py $Operation @linuxPaths
    } else {
        $python = Get-Command py.exe -ErrorAction SilentlyContinue
        if (!$python) { throw 'Python 3.12 is required to package from a Windows checkout.' }
        & $python.Source -3.12 (Join-Path $PSScriptRoot 'scripts\package-windows.py') $Operation @Paths
    }
    if ($LASTEXITCODE -ne 0) { throw "Windows packager failed during $Operation." }
}

$payloadHash = ''
if ($Package) {
    $zip = Join-Path $Destination 'traiNNer-GUI-Windows-x64.zip'
    if (Test-Path -LiteralPath $zip) { throw "A Windows bundle already exists at $zip." }
    $staging = Join-Path $BuildDirectory ('package-' + [Guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($staging)
    $archive = Join-Path $staging 'trainner-gui-source.tar.gz'
    $payloadHash = Invoke-PackageTool -Operation source -Paths @($archive)
    if ($payloadHash -cnotmatch '^[a-f0-9]{64}$') { throw 'The Windows packager did not return a SHA-256.' }
}

# Visual Studio supplies the Windows SDK and linker for clang-cl.
$vswhere = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio\Installer\vswhere.exe'
$install = & $vswhere -latest -products '*' -requires 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' -property installationPath
if ($LASTEXITCODE -ne 0 -or !$install) { throw 'Visual Studio C++ Build Tools are required.' }
Import-Module -Name (Join-Path $install 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $install -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$clang = Get-Command clang-cl -ErrorAction SilentlyContinue
if (!$clang) {
    $installed = 'C:\Executables\clang+llvm-23.1.2-x86_64-pc-windows-msvc\bin\clang-cl.exe'
    if (Test-Path -LiteralPath $installed) { $clang = Get-Item -LiteralPath $installed }
}
if (!$clang) { throw 'LLVM clang-cl is required to build the desktop launcher.' }
$compiler = if ($clang -is [System.IO.FileInfo]) { $clang.FullName } else { $clang.Source }
$configure = @('-S', $source, '-B', $BuildDirectory, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_CXX_COMPILER=$compiler", "-DTRAINNER_GUI_PAYLOAD_SHA256=$payloadHash",
    "-DTRAINNER_GUI_RELEASE_TAG=$ReleaseTag")
& cmake @configure
if ($LASTEXITCODE -ne 0) { throw 'Launcher configuration failed.' }
& cmake --build $BuildDirectory --target traiNNer-GUI
if ($LASTEXITCODE -ne 0) { throw 'Launcher build failed.' }

[void][IO.Directory]::CreateDirectory($Destination)
$built = Join-Path $BuildDirectory 'traiNNer-GUI.exe'
if ($Package) {
    Copy-Item -LiteralPath $built -Destination (Join-Path $staging 'traiNNer-GUI.exe')
    Invoke-PackageTool -Operation bundle -Paths @($staging, $zip) | Out-Null
    Write-Output $zip
} else {
    $launcher = Join-Path $Destination 'traiNNer-GUI.exe'
    if ($built -ne $launcher) {
        Copy-Item -LiteralPath $built -Destination $launcher
    }
}
