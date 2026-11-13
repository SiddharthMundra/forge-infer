#pragma once

#include "common/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm::kernels {

// Device-agnostic kernel surface. CUDA builds override with .cu implementations
// when LLM_HAS_CUDA is defined; otherwise CPU reference paths are used.

void rms_norm(float* out, const float* x, const float* weight, int n, float eps);

void rope_inplace(float* q, float* k, int num_tokens, int num_heads,
                  int num_kv_heads, int head_dim, int start_pos, float theta);

void softmax_inplace(float* x, int n);

// Quantize FP32 row-wise to INT8 with per-row scale.
void quantize_int8(const float* src, int8_t* dst, float* scales, int rows,
                   int cols);

// INT8 GEMM: C[M,N] = (A_int8[M,K] @ B_int8[K,N]^T) * scales, FP32 out.
void gemm_int8(const int8_t* a, const int8_t* b, const float* a_scales,
               const float* b_scales, float* c, int m, int n, int k);

void gemm_f32(const float* a, const float* b, float* c, int m, int n, int k);

// PagedAttention decode: one query token per sequence against paged KV.
// q: [batch, num_heads, head_dim]
// out: [batch, num_heads, head_dim]
void paged_attention(
    float* out, const float* q, const float* kv_cache, int batch,
    int num_heads, int num_kv_heads, int head_dim, int block_size,
    const std::vector<std::vector<BlockId>>& block_tables,
    const std::vector<int>& context_lens, float scale);

// Prefill attention for a contiguous prompt chunk (non-paged for simplicity
// on the active window; results are written into paged KV via write_kv_cache).
void write_kv_cache(float* kv_cache, const float* k, const float* v, int layer,
                    int num_tokens, int num_kv_heads, int head_dim,
                    int block_size, const std::vector<BlockId>& block_table,
                    int start_pos, int num_layers, int num_blocks);

DeviceType active_device();
const char* device_name();

}  // namespace llm::kernels
