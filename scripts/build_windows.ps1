# Build for Windows + RTX 2070 (sm_75).
# Prerequisites:
#   - Visual Studio 2022 (Desktop C++ workload)
#   - CUDA Toolkit 12.x (or 11.8+)
#   - CMake 3.20+
#   - vcpkg with grpc + protobuf
#
# Example:
#   $env:VCPKG_ROOT = "C:\src\vcpkg"
#   .\scripts\build_windows.ps1

param(
  [string]$VcpkgRoot = $env:VCPKG_ROOT,
  [string]$BuildDir = "build",
  [string]$Config = "Release",
  [string]$CudaArchs = "75",
  [switch]$NoGrpc,
  [switch]$NoCuda
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $VcpkgRoot) {
  Write-Error "Set VCPKG_ROOT or pass -VcpkgRoot. Example: C:\src\vcpkg"
}

$Toolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
if (-not (Test-Path $Toolchain)) {
  Write-Error "vcpkg toolchain not found: $Toolchain"
}

Write-Host "Installing gRPC/protobuf via vcpkg (x64-windows)..."
& (Join-Path $VcpkgRoot "vcpkg.exe") install grpc:x64-windows protobuf:x64-windows
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed" }

$CudaFlag = "ON"
if ($NoCuda) { $CudaFlag = "OFF" }
$GrpcFlag = "ON"
if ($NoGrpc) { $GrpcFlag = "OFF" }

Write-Host "Configuring CMake (CUDA=$CudaFlag arch=$CudaArchs GRPC=$GrpcFlag)..."
cmake -S . -B $BuildDir `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
  -DLLM_ENABLE_CUDA=$CudaFlag `
  -DLLM_ENABLE_GRPC=$GrpcFlag `
  -DLLM_CUDA_ARCHS=$CudaArchs

if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "Building ($Config)..."
cmake --build $BuildDir --config $Config -j
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$OutDir = Join-Path $BuildDir $Config
Write-Host ""
Write-Host "Build OK. Binaries:"
Write-Host "  $(Join-Path $OutDir 'llm_bench.exe')"
if ($GrpcFlag -eq "ON") {
  Write-Host "  $(Join-Path $OutDir 'llm_server_app.exe')"
}
Write-Host ""
Write-Host "Smoke test:"
Write-Host "  & '$OutDir\llm_bench.exe'"
Write-Host "Server:"
Write-Host "  & '$OutDir\llm_server_app.exe' --addr 0.0.0.0:50051 --blocks 1024"
