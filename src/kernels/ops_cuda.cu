// CUDA kernels for high-throughput paths. Built only when LLM_HAS_CUDA=1.
// Tuned default arch: sm_75 (RTX 2070 / Turing). Override via LLM_CUDA_ARCHS.

#include "kernels/ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm::kernels {
namespace {

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err = (expr);                                                 \
    if (err != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(err));                      \
    }                                                                         \
  } while (0)

__global__ void rms_norm_kernel(float* out, const float* x, const float* weight,
                                int n, float eps) {
  // One block per row; this launch uses a single row for simplicity.
  float ss = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = x[i];
    ss += v * v;
  }
  __shared__ float shared[256];
  shared[threadIdx.x] = ss;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float inv = rsqrtf(shared[0] / static_cast<float>(n) + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = x[i] * inv * weight[i];
  }
}

__global__ void quantize_int8_kernel(const float* src, int8_t* dst,
                                     float* scales, int rows, int cols) {
  int r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const float* row = src + static_cast<size_t>(r) * cols;
  float amax = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    amax = fmaxf(amax, fabsf(row[c]));
  }
  __shared__ float shared[256];
  shared[threadIdx.x] = amax;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                  shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float scale = shared[0] > 0.0f ? shared[0] / 127.0f : 1.0f;
  if (threadIdx.x == 0) {
    scales[r] = scale;
  }
  __syncthreads();
  int8_t* out = dst + static_cast<size_t>(r) * cols;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float q = rintf(row[c] / scale);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    out[c] = static_cast<int8_t>(q);
  }
}

}  // namespace

// For the portfolio CUDA path we keep host-side orchestration and accelerate
// the hottest kernels. Full device-resident model weights are a follow-on.

void rms_norm(float* out, const float* x, const float* weight, int n,
              float eps) {
  float *d_out = nullptr, *d_x = nullptr, *d_w = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, weight, n * sizeof(float), cudaMemcpyHostToDevice));
  rms_norm_kernel<<<1, 256>>>(d_out, d_x, d_w, n, eps);
  CUDA_CHECK(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  cudaFree(d_x);
  cudaFree(d_w);
}

void rope_inplace(float* q, float* k, int num_tokens, int num_heads,
                  int num_kv_heads, int head_dim, int start_pos, float theta) {
  // Fallback to CPU math for RoPE until fused CUDA kernel lands.
  // Avoid linking the CPU TU twice: inline a minimal copy.
  const int half = head_dim / 2;
  for (int t = 0; t < num_tokens; ++t) {
    const int pos = start_pos + t;
    for (int h = 0; h < num_heads; ++h) {
      float* qh = q + (t * num_heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float freq =
            1.0f / powf(theta, static_cast<float>(2 * i) / head_dim);
        const float angle = static_cast<float>(pos) * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float q0 = qh[i];
        const float q1 = qh[i + half];
        qh[i] = q0 * c - q1 * s;
        qh[i + half] = q0 * s + q1 * c;
      }
    }
    for (int h = 0; h < num_kv_heads; ++h) {
      float* kh = k + (t * num_kv_heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float freq =
            1.0f / powf(theta, static_cast<float>(2 * i) / head_dim);
        const float angle = static_cast<float>(pos) * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float k0 = kh[i];
        const float k1 = kh[i + half];
        kh[i] = k0 * c - k1 * s;
        kh[i + half] = k0 * s + k1 * c;
      }
    }
  }
}

void softmax_inplace(float* x, int n) {
  float max_v = x[0];
  for (int i = 1; i < n; ++i) {
    max_v = fmaxf(max_v, x[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = expf(x[i] - max_v);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) {
    x[i] *= inv;
  }
}

void quantize_int8(const float* src, int8_t* dst, float* scales, int rows,
                   int cols) {
  float *d_src = nullptr, *d_scales = nullptr;
  int8_t* d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, static_cast<size_t>(rows) * cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, static_cast<size_t>(rows) * cols * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, rows * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, src,
                        static_cast<size_t>(rows) * cols * sizeof(float),
                        cudaMemcpyHostToDevice));
  quantize_int8_kernel<<<rows, 256>>>(d_src, d_dst, d_scales, rows, cols);
  CUDA_CHECK(cudaMemcpy(dst, d_dst,
                        static_cast<size_t>(rows) * cols * sizeof(int8_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(scales, d_scales, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_scales);
}

void gemm_int8(const int8_t* a, const int8_t* b, const float* a_scales,
               const float* b_scales, float* c, int m, int n, int k) {
  // Host reference until cuBLASLt INT8 epilogue is wired.
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t acc = 0;
      const int8_t* ai = a + static_cast<size_t>(i) * k;
      const int8_t* bj = b + static_cast<size_t>(j) * k;
      for (int t = 0; t < k; ++t) {
        acc += static_cast<int32_t>(ai[t]) * static_cast<int32_t>(bj[t]);
      }
      c[static_cast<size_t>(i) * n + j] =
          static_cast<float>(acc) * a_scales[i] * b_scales[j];
    }
  }
}

void gemm_f32(const float* a, const float* b, float* c, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      const float* ai = a + static_cast<size_t>(i) * k;
      const float* bj = b + static_cast<size_t>(j) * k;
      for (int t = 0; t < k; ++t) {
        sum += ai[t] * bj[t];
      }
      c[static_cast<size_t>(i) * n + j] = sum;
    }
  }
}

void write_kv_cache(float* kv_cache, const float* k, const float* v, int layer,
                    int num_tokens, int num_kv_heads, int head_dim,
                    int block_size, const std::vector<BlockId>& block_table,
                    int start_pos, int num_layers, int num_blocks) {
  const size_t page =
      static_cast<size_t>(block_size) * num_kv_heads * head_dim;
  const size_t layer_stride = static_cast<size_t>(num_blocks) * 2 * page;
  for (int t = 0; t < num_tokens; ++t) {
    const int pos = start_pos + t;
    const int block_idx = pos / block_size;
    const int slot = pos % block_size;
    const BlockId bid = block_table[static_cast<size_t>(block_idx)];
    float* k_page = kv_cache + static_cast<size_t>(layer) * layer_stride +
                    static_cast<size_t>(bid) * 2 * page;
    float* v_page = k_page + page;
    const float* kt = k + static_cast<size_t>(t) * num_kv_heads * head_dim;
    const float* vt = v + static_cast<size_t>(t) * num_kv_heads * head_dim;
    float* k_slot =
        k_page + static_cast<size_t>(slot) * num_kv_heads * head_dim;
    float* v_slot =
        v_page + static_cast<size_t>(slot) * num_kv_heads * head_dim;
    for (int i = 0; i < num_kv_heads * head_dim; ++i) {
      k_slot[i] = kt[i];
      v_slot[i] = vt[i];
    }
  }
  (void)num_layers;
}

void paged_attention(float* out, const float* q, const float* kv_cache,
                     int batch, int num_heads, int num_kv_heads, int head_dim,
                     int block_size,
                     const std::vector<std::vector<BlockId>>& block_tables,
                     const std::vector<int>& context_lens, float scale) {
  const int kv_mul = num_heads / num_kv_heads;
  const size_t page =
      static_cast<size_t>(block_size) * num_kv_heads * head_dim;
  std::vector<float> scores;
  for (int b = 0; b < batch; ++b) {
    const int ctx = context_lens[static_cast<size_t>(b)];
    const auto& table = block_tables[static_cast<size_t>(b)];
    scores.assign(static_cast<size_t>(ctx), 0.0f);
    for (int h = 0; h < num_heads; ++h) {
      const int kv_h = h / kv_mul;
      const float* qh =
          q + (static_cast<size_t>(b) * num_heads + h) * head_dim;
      for (int pos = 0; pos < ctx; ++pos) {
        const int block_idx = pos / block_size;
        const int slot = pos % block_size;
        const BlockId bid = table[static_cast<size_t>(block_idx)];
        const float* k_page =
            kv_cache + static_cast<size_t>(bid) * 2 * page;
        const float* k_slot =
            k_page + static_cast<size_t>(slot) * num_kv_heads * head_dim +
            kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += qh[d] * k_slot[d];
        }
        scores[static_cast<size_t>(pos)] = scale * dot;
      }
      softmax_inplace(scores.data(), ctx);
      float* oh =
          out + (static_cast<size_t>(b) * num_heads + h) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        oh[d] = 0.0f;
      }
      for (int pos = 0; pos < ctx; ++pos) {
        const int block_idx = pos / block_size;
        const int slot = pos % block_size;
        const BlockId bid = table[static_cast<size_t>(block_idx)];
        const float* v_page =
            kv_cache + static_cast<size_t>(bid) * 2 * page + page;
        const float* v_slot =
            v_page + static_cast<size_t>(slot) * num_kv_heads * head_dim +
            kv_h * head_dim;
        const float w = scores[static_cast<size_t>(pos)];
        for (int d = 0; d < head_dim; ++d) {
          oh[d] += w * v_slot[d];
        }
      }
    }
  }
}

DeviceType active_device() { return DeviceType::kCuda; }

const char* device_name() {
  static char name[256] = "cuda";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    std::snprintf(name, sizeof(name), "cuda:%s", prop.name);
  }
  return name;
}

}  // namespace llm::kernels
