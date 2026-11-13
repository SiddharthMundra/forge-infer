#include "model/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace llm {

Sampler::Sampler(uint32_t seed) : rng_(seed) {}

TokenId Sampler::sample(const float* logits, int vocab_size,
                        const SamplingParams& params) {
  std::vector<int> indices(static_cast<std::size_t>(vocab_size));
  std::iota(indices.begin(), indices.end(), 0);

  if (params.top_k > 0 && params.top_k < vocab_size) {
    std::partial_sort(
        indices.begin(), indices.begin() + params.top_k, indices.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });
    indices.resize(static_cast<std::size_t>(params.top_k));
  } else {
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });
  }

  const float temp = std::max(params.temperature, 1e-5f);
  std::vector<float> probs(indices.size());
  float max_v = logits[indices[0]] / temp;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    probs[i] = logits[indices[i]] / temp;
    max_v = std::max(max_v, probs[i]);
  }
  float sum = 0.0f;
  for (float& p : probs) {
    p = std::exp(p - max_v);
    sum += p;
  }
  for (float& p : probs) {
    p /= sum;
  }

  // Nucleus (top-p) truncation.
  if (params.top_p < 1.0f) {
    float cum = 0.0f;
    std::size_t cutoff = probs.size();
    for (std::size_t i = 0; i < probs.size(); ++i) {
      cum += probs[i];
      if (cum >= params.top_p) {
        cutoff = i + 1;
        break;
      }
    }
    probs.resize(cutoff);
    indices.resize(cutoff);
    sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (float& p : probs) {
      p /= sum;
    }
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return static_cast<TokenId>(indices[static_cast<std::size_t>(dist(rng_))]);
}

}  // namespace llm
