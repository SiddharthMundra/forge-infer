#pragma once

#include "common/types.hpp"

#include <random>
#include <vector>

namespace llm {

class Sampler {
 public:
  explicit Sampler(uint32_t seed = 1234);

  TokenId sample(const float* logits, int vocab_size,
                 const SamplingParams& params);

 private:
  std::mt19937 rng_;
};

}  // namespace llm
