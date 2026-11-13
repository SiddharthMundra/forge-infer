#include "kernels/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace llm::kernels {
namespace {

inline float dot(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

}  // namespace

void rms_norm(float* out, const float* x, const float* weight, int n,
              float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * inv * weight[i];
  }
}

void rope_inplace(float* q, float* k, int num_tokens, int num_heads,
                  int num_kv_heads, int head_dim, int start_pos, float theta) {
  const int half = head_dim / 2;
  for (int t = 0; t < num_tokens; ++t) {
    const int pos = start_pos + t;
    for (int h = 0; h < num_heads; ++h) {
      float* qh = q + (t * num_heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float freq =
            1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
        const float angle = static_cast<float>(pos) * freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
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
            1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
        const float angle = static_cast<float>(pos) * freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
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
    max_v = std::max(max_v, x[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - max_v);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) {
    x[i] *= inv;
  }
}

void quantize_int8(const float* src, int8_t* dst, float* scales, int rows,
                   int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = src + static_cast<std::size_t>(r) * cols;
    float amax = 0.0f;
    for (int c = 0; c < cols; ++c) {
      amax = std::max(amax, std::fabs(row[c]));
    }
    const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    scales[r] = scale;
    int8_t* out = dst + static_cast<std::size_t>(r) * cols;
    for (int c = 0; c < cols; ++c) {
      const float q = std::round(row[c] / scale);
      out[c] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }
}

void gemm_int8(const int8_t* a, const int8_t* b, const float* a_scales,
               const float* b_scales, float* c, int m, int n, int k) {
  // a: [m,k] row-major, b: [n,k] row-major (B stored as rows = out features)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t acc = 0;
      const int8_t* ai = a + static_cast<std::size_t>(i) * k;
      const int8_t* bj = b + static_cast<std::size_t>(j) * k;
      for (int t = 0; t < k; ++t) {
        acc += static_cast<int32_t>(ai[t]) * static_cast<int32_t>(bj[t]);
      }
      c[static_cast<std::size_t>(i) * n + j] =
          static_cast<float>(acc) * a_scales[i] * b_scales[j];
    }
  }
}

void gemm_f32(const float* a, const float* b, float* c, int m, int n, int k) {
  // a[m,k] * b[n,k]^T -> c[m,n]  (b rows are output features)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[static_cast<std::size_t>(i) * n + j] =
          dot(a + static_cast<std::size_t>(i) * k,
              b + static_cast<std::size_t>(j) * k, k);
    }
  }
}

void write_kv_cache(float* kv_cache, const float* k, const float* v, int layer,
                    int num_tokens, int num_kv_heads, int head_dim,
                    int block_size, const std::vector<BlockId>& block_table,
                    int start_pos, int num_layers, int num_blocks) {
  const std::size_t page =
      static_cast<std::size_t>(block_size) * num_kv_heads * head_dim;
  const std::size_t layer_stride =
      static_cast<std::size_t>(num_blocks) * 2 * page;

  for (int t = 0; t < num_tokens; ++t) {
    const int pos = start_pos + t;
    const int block_idx = pos / block_size;
    const int slot = pos % block_size;
    if (block_idx >= static_cast<int>(block_table.size())) {
      throw std::runtime_error("write_kv_cache: missing block");
    }
    const BlockId bid = block_table[static_cast<std::size_t>(block_idx)];
    float* k_page = kv_cache + static_cast<std::size_t>(layer) * layer_stride +
                    static_cast<std::size_t>(bid) * 2 * page;
    float* v_page = k_page + page;
    const float* kt =
        k + static_cast<std::size_t>(t) * num_kv_heads * head_dim;
    const float* vt =
        v + static_cast<std::size_t>(t) * num_kv_heads * head_dim;
    float* k_slot = k_page + static_cast<std::size_t>(slot) * num_kv_heads *
                                 head_dim;
    float* v_slot = v_page + static_cast<std::size_t>(slot) * num_kv_heads *
                                 head_dim;
    std::copy(kt, kt + num_kv_heads * head_dim, k_slot);
    std::copy(vt, vt + num_kv_heads * head_dim, v_slot);
  }
  (void)num_layers;
}

void paged_attention(float* out, const float* q, const float* kv_cache,
                     int batch, int num_heads, int num_kv_heads, int head_dim,
                     int block_size,
                     const std::vector<std::vector<BlockId>>& block_tables,
                     const std::vector<int>& context_lens, float scale) {
  // Assumes single-layer view: kv_cache points at layer's [num_blocks,2,page]
  // region. Caller passes layer base pointer.
  const int kv_mul = num_heads / num_kv_heads;
  const std::size_t page =
      static_cast<std::size_t>(block_size) * num_kv_heads * head_dim;

  std::vector<float> scores;
  for (int b = 0; b < batch; ++b) {
    const int ctx = context_lens[static_cast<std::size_t>(b)];
    const auto& table = block_tables[static_cast<std::size_t>(b)];
    scores.assign(static_cast<std::size_t>(ctx), 0.0f);

    for (int h = 0; h < num_heads; ++h) {
      const int kv_h = h / kv_mul;
      const float* qh =
          q + (static_cast<std::size_t>(b) * num_heads + h) * head_dim;

      for (int pos = 0; pos < ctx; ++pos) {
        const int block_idx = pos / block_size;
        const int slot = pos % block_size;
        const BlockId bid = table[static_cast<std::size_t>(block_idx)];
        const float* k_page =
            kv_cache + static_cast<std::size_t>(bid) * 2 * page;
        const float* k_slot =
            k_page + static_cast<std::size_t>(slot) * num_kv_heads * head_dim +
            kv_h * head_dim;
        scores[static_cast<std::size_t>(pos)] = scale * dot(qh, k_slot, head_dim);
      }
      softmax_inplace(scores.data(), ctx);

      float* oh =
          out + (static_cast<std::size_t>(b) * num_heads + h) * head_dim;
      std::fill(oh, oh + head_dim, 0.0f);
      for (int pos = 0; pos < ctx; ++pos) {
        const int block_idx = pos / block_size;
        const int slot = pos % block_size;
        const BlockId bid = table[static_cast<std::size_t>(block_idx)];
        const float* v_page =
            kv_cache + static_cast<std::size_t>(bid) * 2 * page + page;
        const float* v_slot =
            v_page + static_cast<std::size_t>(slot) * num_kv_heads * head_dim +
            kv_h * head_dim;
        const float w = scores[static_cast<std::size_t>(pos)];
        for (int d = 0; d < head_dim; ++d) {
          oh[d] += w * v_slot[d];
        }
      }
    }
  }
}

DeviceType active_device() { return DeviceType::kCpu; }

const char* device_name() { return "cpu"; }

}  // namespace llm::kernels
