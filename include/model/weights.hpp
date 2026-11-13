#pragma once

#include "common/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

struct LinearWeights {
  std::vector<float> weight;  // [out, in]
  std::vector<int8_t> weight_int8;
  std::vector<float> scales;  // per-out-row
  int in_features = 0;
  int out_features = 0;
  bool quantized = false;

  void quantize();
};

struct AttentionWeights {
  LinearWeights wq;
  LinearWeights wk;
  LinearWeights wv;
  LinearWeights wo;
};

struct MlpWeights {
  LinearWeights gate;
  LinearWeights up;
  LinearWeights down;
};

struct LayerWeights {
  std::vector<float> attn_norm;
  AttentionWeights attn;
  std::vector<float> ffn_norm;
  MlpWeights mlp;
};

struct ModelWeights {
  ModelConfig config;
  std::vector<float> tok_embeddings;  // [vocab, hidden]
  std::vector<LayerWeights> layers;
  std::vector<float> norm;
  LinearWeights lm_head;

  static std::unique_ptr<ModelWeights> create_random(const ModelConfig& cfg,
                                                     uint32_t seed = 42);
  static std::unique_ptr<ModelWeights> load(const std::string& path);
};

}  // namespace llm
