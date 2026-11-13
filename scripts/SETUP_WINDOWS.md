# Windows + RTX 2070 setup checklist

## 1. Install toolchain

| Tool | Notes |
|---|---|
| Visual Studio 2022 | Workload: **Desktop development with C++** |
| CUDA Toolkit 12.x (or 11.8+) | Must match a VS-supported version; reboot after install |
| CMake 3.20+ | `winget install Kitware.CMake` |
| Git | `winget install Git.Git` |
| vcpkg | See below |

Verify:

```powershell
nvidia-smi          # should show RTX 2070
nvcc --version
cl                  # from "x64 Native Tools Command Prompt for VS 2022"
cmake --version
```

RTX 2070 compute capability is **7.5** → we pass `-DLLM_CUDA_ARCHS=75`.

## 2. vcpkg + gRPC

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
cd C:\src\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install grpc:x64-windows protobuf:x64-windows
$env:VCPKG_ROOT = "C:\src\vcpkg"
```

Persist `VCPKG_ROOT` in System Environment Variables if you want.

## 3. Build this repo

From the repo root (PowerShell):

```powershell
$env:VCPKG_ROOT = "C:\src\vcpkg"
.\scripts\build_windows.ps1
```

Or manually:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DLLM_ENABLE_CUDA=ON -DLLM_CUDA_ARCHS=75 -DLLM_ENABLE_GRPC=ON

cmake --build build --config Release -j
```

## 4. Run

```powershell
.\build\Release\llm_bench.exe
.\build\Release\llm_server_app.exe --addr 0.0.0.0:50051 --blocks 2048
```

## Common issues

- **`nvcc` not found** — install CUDA Toolkit and reopen the VS developer shell.
- **`grpc` not found** — pass `-DCMAKE_TOOLCHAIN_FILE=...\vcpkg.cmake` and reinstall `grpc:x64-windows`.
- **Wrong architecture / no kernel image** — ensure `-DLLM_CUDA_ARCHS=75` (not 80/86/89).
- **MSB3721 CUDA compile errors** — use the “x64 Native Tools” prompt so `cl.exe` is on PATH for nvcc’s host compiler.
