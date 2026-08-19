param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",
    [switch]$SkipWebBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$androidRoot = Join-Path $projectRoot "android"
$toolRoot = Join-Path $env:LOCALAPPDATA "llm-service-tools"
$sdkRoot = Join-Path $toolRoot "android-sdk"
$jdkRoot = Join-Path $toolRoot "jdk\jdk-17.0.20+8"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vswhere)) {
    throw "Visual Studio C++ build tools are required for the Vulkan shader generator."
}
$vsRoot = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vsDevCmd = Join-Path $vsRoot "Common7\Tools\VsDevCmd.bat"
$environmentLines = cmd.exe /d /c call "$vsDevCmd" -arch=x64 -host_arch=x64 `&`& set
foreach ($line in $environmentLines) {
    $separator = $line.IndexOf("=")
    if ($separator -gt 0) {
        [Environment]::SetEnvironmentVariable(
            $line.Substring(0, $separator),
            $line.Substring($separator + 1),
            "Process")
    }
}

$env:JAVA_HOME = $jdkRoot
$env:ANDROID_HOME = $sdkRoot
$env:PYTHONUTF8 = "1"
$env:Path = (Join-Path $sdkRoot "cmake\3.22.1\bin") + ";" + $env:Path

$task = if ($Configuration -eq "Release") { ":app:assembleRelease" } else { ":app:assembleDebug" }
$variant = $Configuration.ToLowerInvariant()
# A cancelled Android packaging task can leave a truncated ZIP which AGP then
# mistakes for an incremental input. APKs are generated outputs, so always
# begin packaging from a fresh file while preserving all native build caches.
$apkOutput = Join-Path $androidRoot "app\build\outputs\apk\$variant\app-$variant.apk"
if (Test-Path $apkOutput) {
    Remove-Item -LiteralPath $apkOutput -Force
}
$gradleArguments = @("-p", $androidRoot, $task, "--no-daemon")
if ($SkipWebBuild) {
    if (-not (Test-Path (Join-Path $projectRoot "web\.output\public\index.html"))) {
        throw "Cannot skip the web build because web/.output/public/index.html is missing"
    }
    $gradleArguments += @("-x", ":app:buildWebUi")
}
& (Join-Path $androidRoot "gradlew.bat") @gradleArguments
if ($LASTEXITCODE -ne 0) {
    throw "Android Gradle build failed with exit code $LASTEXITCODE"
}
