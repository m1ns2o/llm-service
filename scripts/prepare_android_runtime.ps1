param(
    [string]$LlamaCppRef = "25ae3a9b331fffea50ff8d07a5cad34c33f1276f",
    [string]$SpirvHeadersRef = "0d25db97cb9b8f725e4c95e4553001710e7fc39d",
    [string]$VulkanHeadersRef = "0b7f383797fa7be53ae28213e001ae60668ee511",
    [string]$OpenClHeadersRef = "15b536b7fbe1098cea462a27db496b287ac89b63",
    [string]$OpenClLoaderRef = "45cdbda4ddd31c324e32a744f112087c42da18f7"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $projectRoot "third_party\llama.cpp"
$spirvDir = Join-Path $projectRoot "third_party\SPIRV-Headers"
$vulkanHeadersDir = Join-Path $projectRoot "third_party\Vulkan-Headers"
$openClHeadersDir = Join-Path $projectRoot "third_party\OpenCL-Headers"
$openClLoaderDir = Join-Path $projectRoot "third_party\OpenCL-ICD-Loader"

if (-not (Test-Path (Join-Path $sourceDir ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $sourceDir) | Out-Null
    git init $sourceDir
    git -C $sourceDir remote add origin https://github.com/ggml-org/llama.cpp.git
}

git -C $sourceDir fetch --depth 1 origin $LlamaCppRef
git -C $sourceDir checkout --detach FETCH_HEAD
$resolved = git -C $sourceDir rev-parse HEAD
if ($resolved -ne $LlamaCppRef) {
    throw "llama.cpp revision mismatch: $resolved != $LlamaCppRef"
}

if (-not (Test-Path (Join-Path $spirvDir ".git"))) {
    git init $spirvDir
    git -C $spirvDir remote add origin https://github.com/KhronosGroup/SPIRV-Headers.git
}
git -C $spirvDir fetch --depth 1 origin $SpirvHeadersRef
git -C $spirvDir checkout --detach FETCH_HEAD
$resolvedSpirv = git -C $spirvDir rev-parse HEAD
if ($resolvedSpirv -ne $SpirvHeadersRef) {
    throw "SPIRV-Headers revision mismatch: $resolvedSpirv != $SpirvHeadersRef"
}

if (-not (Test-Path (Join-Path $vulkanHeadersDir ".git"))) {
    git init $vulkanHeadersDir
    git -C $vulkanHeadersDir remote add origin https://github.com/KhronosGroup/Vulkan-Headers.git
}
git -C $vulkanHeadersDir fetch --depth 1 origin $VulkanHeadersRef
git -C $vulkanHeadersDir checkout --detach FETCH_HEAD
$resolvedVulkanHeaders = git -C $vulkanHeadersDir rev-parse HEAD
if ($resolvedVulkanHeaders -ne $VulkanHeadersRef) {
    throw "Vulkan-Headers revision mismatch: $resolvedVulkanHeaders != $VulkanHeadersRef"
}

if (-not (Test-Path (Join-Path $openClHeadersDir ".git"))) {
    git init $openClHeadersDir
    git -C $openClHeadersDir remote add origin https://github.com/KhronosGroup/OpenCL-Headers.git
}
git -C $openClHeadersDir fetch --depth 1 origin $OpenClHeadersRef
git -C $openClHeadersDir checkout --detach FETCH_HEAD
if ((git -C $openClHeadersDir rev-parse HEAD) -ne $OpenClHeadersRef) {
    throw "OpenCL-Headers revision mismatch"
}

if (-not (Test-Path (Join-Path $openClLoaderDir ".git"))) {
    git init $openClLoaderDir
    git -C $openClLoaderDir remote add origin https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
}
git -C $openClLoaderDir fetch --depth 1 origin $OpenClLoaderRef
git -C $openClLoaderDir checkout --detach FETCH_HEAD
if ((git -C $openClLoaderDir rev-parse HEAD) -ne $OpenClLoaderRef) {
    throw "OpenCL-ICD-Loader revision mismatch"
}

$spirvBuild = Join-Path $spirvDir "build"
$spirvInstall = Join-Path $spirvDir "install"
cmake -S $spirvDir -B $spirvBuild "-DCMAKE_INSTALL_PREFIX=$spirvInstall"
cmake --install $spirvBuild --config Release

# Build a link-time OpenCL stub. It is deliberately excluded from the APK;
# Android supplies the device vendor implementation at runtime.
$sdkRoot = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else {
    Join-Path $env:LOCALAPPDATA "llm-service-tools\android-sdk"
}
$cmakeExe = Join-Path $sdkRoot "cmake\3.22.1\bin\cmake.exe"
$ninjaExe = Join-Path $sdkRoot "cmake\3.22.1\bin\ninja.exe"
$ndkRoot = Join-Path $sdkRoot "ndk\26.1.10909125"
$openClBuild = Join-Path $openClLoaderDir "build-android2"
& $cmakeExe -S $openClLoaderDir -B $openClBuild -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninjaExe" `
    "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $ndkRoot 'build\cmake\android.toolchain.cmake')" `
    -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=28 `
    "-DOPENCL_ICD_LOADER_HEADERS_DIR=$openClHeadersDir" `
    -DENABLE_OPENCL_LAYERS=OFF -DOPENCL_ICD_LOADER_BUILD_SHARED_LIBS=ON
& $cmakeExe --build $openClBuild --target OpenCL

Write-Host "Prepared pinned llama.cpp runtime: $resolved"
Write-Host "Prepared pinned SPIRV-Headers: $resolvedSpirv"
Write-Host "Prepared pinned Vulkan-Headers: $resolvedVulkanHeaders"
Write-Host "Prepared pinned OpenCL headers and Android link-time loader"
Write-Host "Build with Android Studio or: android\gradlew.bat -p android assembleDebug"
