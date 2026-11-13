#include "model/weights.hpp"

#include "kernels/ops.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace llm {
namespace {

void init_linear(LinearWeights& w, int out_f, int in_f, std::mt19937& rng,
                 bool quantize) {
  w.out_features = out_f;
  w.in_features = in_f;
  w.weight.resize(static_cast<std::size_t>(out_f) * in_f);
  const float stddev = 1.0f / std::sqrt(static_cast<float>(in_f));
  std::normal_distribution<float> dist(0.0f, stddev);
  for (float& v : w.weight) {
    v = dist(rng);
  }
  if (quantize) {
    w.quantize();
  }
}

void init_vec(std::vector<float>& v, int n, float fill = 1.0f) {
  v.assign(static_cast<std::size_t>(n), fill);
}

}  // namespace

void LinearWeights::quantize() {
  weight_int8.resize(weight.size());
  scales.resize(static_cast<std::size_t>(out_features));
  kernels::quantize_int8(weight.data(), weight_int8.data(), scales.data(),
                         out_features, in_features);
  quantized = true;
}

std::unique_ptr<ModelWeights> ModelWeights::create_random(const ModelConfig& cfg,
                                                          uint32_t seed) {
  auto model = std::make_unique<ModelWeights>();
  model->config = cfg;
  std::mt19937 rng(seed);

  model->tok_embeddings.resize(
      static_cast<std::size_t>(cfg.vocab_size) * cfg.hidden_size);
  {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (float& v : model->tok_embeddings) {
      v = dist(rng);
    }
  }

  model->layers.resize(static_cast<std::size_t>(cfg.num_layers));
  for (auto& layer : model->layers) {
    init_vec(layer.attn_norm, cfg.hidden_size);
    init_linear(layer.attn.wq, cfg.num_heads * cfg.head_dim, cfg.hidden_size,
                rng, cfg.use_int8);
    init_linear(layer.attn.wk, cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size,
                rng, cfg.use_int8);
    init_linear(layer.attn.wv, cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size,
                rng, cfg.use_int8);
    init_linear(layer.attn.wo, cfg.hidden_size, cfg.num_heads * cfg.head_dim,
                rng, cfg.use_int8);
    init_vec(layer.ffn_norm, cfg.hidden_size);
    init_linear(layer.mlp.gate, cfg.intermediate_size, cfg.hidden_size, rng,
                cfg.use_int8);
    init_linear(layer.mlp.up, cfg.intermediate_size, cfg.hidden_size, rng,
                cfg.use_int8);
    init_linear(layer.mlp.down, cfg.hidden_size, cfg.intermediate_size, rng,
                cfg.use_int8);
  }

  init_vec(model->norm, cfg.hidden_size);
  init_linear(model->lm_head, cfg.vocab_size, cfg.hidden_size, rng,
              cfg.use_int8);
  return model;
}

std::unique_ptr<ModelWeights> ModelWeights::load(const std::string& path) {
  throw std::runtime_error(
      "ModelWeights::load('" + path +
      "') not implemented yet — use create_random() for the toy path");
}

}  // namespace llm
