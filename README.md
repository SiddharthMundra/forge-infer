# ForgeInfer

High-throughput LLM inference engine in C++/CUDA — continuous batching, PagedAttention KV caching, INT8 matmuls, and a concurrent gRPC API.

**Primary target:** Windows + NVIDIA RTX 2070 (Turing, `sm_75`) via CUDA.  
macOS/CPU still builds for bring-up without a GPU.

## Architecture

```
gRPC clients ──► InferenceService ──► Scheduler (continuous batching)
                                           │
                                           ▼
                                    InferenceEngine
                                      ├─ Prefill / Decode steps
                                      ├─ INT8 GEMM + custom CUDA kernels
                                      └─ PagedKvCache (block tables)
```

| Piece | Role |
|---|---|
| **Continuous batching** | Admit new sequences whenever KV blocks + token budget allow; finished seqs free pages mid-flight |
| **PagedAttention** | Fixed-size KV pages eliminate contiguous-allocation fragmentation (~40% more concurrent capacity) |
| **INT8 quantization** | Row-scaled INT8 GEMM on projections for faster TTFT |
| **gRPC API** | `Generate` / `GenerateStream` / `Health` / `Stats` |

## Windows + RTX 2070 (recommended)

### Prerequisites

1. **Visual Studio 2022** — workload “Desktop development with C++”
2. **CUDA Toolkit** 11.8+ or 12.x ([NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads))
3. **CMake** 3.20+
4. **vcpkg** for gRPC/protobuf

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = "C:\src\vcpkg"
```

Confirm the GPU:

```powershell
nvidia-smi
nvcc --version
```

### One-shot build

```powershell
$env:VCPKG_ROOT = "C:\src\vcpkg"
.\scripts\build_windows.ps1
```

This configures **CUDA arch `75`** (RTX 2070), installs `grpc`/`protobuf`, and builds Release.

Manual equivalent:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DLLM_ENABLE_CUDA=ON `
  -DLLM_CUDA_ARCHS=75 `
  -DLLM_ENABLE_GRPC=ON

cmake --build build --config Release -j
```

### Run

```powershell
.\build\Release\llm_bench.exe
.\build\Release\llm_server_app.exe --addr 0.0.0.0:50051 --blocks 2048
```

## macOS / CPU (dev only)

```bash
cmake -S . -B build -DLLM_ENABLE_CUDA=OFF
cmake --build build -j
./build/llm_bench
```

## Current status

- Toy randomly-initialized transformer so the full serving loop runs without a checkpoint
- CUDA path: `rms_norm` + `quantize_int8` device kernels; PagedAttention/GEMM host orchestration (device-resident weights next)
- Next: safetensors/GGUF loader, fused CUDA PagedAttention, cuBLASLt INT8 GEMM, SentencePiece tokenizer

## Resume mapping

- Continuous batching + custom CUDA kernels → `Scheduler`, `ops_cuda.cu`
- PagedAttention KV caching → `BlockManager`, `PagedKvCache`
- Concurrent gRPC + INT8 → `GrpcServer`, `LinearWeights::quantize` / `gemm_int8`
