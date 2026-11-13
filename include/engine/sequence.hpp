#pragma once

#include "common/types.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace llm {

struct Sequence {
  SeqId id = 0;
  std::string request_id;
  std::string prompt_text;

  std::vector<TokenId> prompt_tokens;
  std::vector<TokenId> output_tokens;
  std::vector<BlockId> block_table;

  SamplingParams sampling;
  SequenceStatus status = SequenceStatus::kWaiting;

  int num_computed_tokens = 0;
  bool arriving = true;

  std::chrono::steady_clock::time_point created_at;
  std::chrono::steady_clock::time_point first_token_at;
  bool has_first_token = false;

  int num_tokens() const {
    return static_cast<int>(prompt_tokens.size() + output_tokens.size());
  }

  int num_prompt_tokens() const {
    return static_cast<int>(prompt_tokens.size());
  }

  bool is_prefill() const { return num_computed_tokens < num_prompt_tokens(); }

  TokenId last_token() const {
    if (!output_tokens.empty()) {
      return output_tokens.back();
    }
    return prompt_tokens.back();
  }

  bool finished() const {
    return status == SequenceStatus::kFinished ||
           status == SequenceStatus::kAborted ||
           static_cast<int>(output_tokens.size()) >= sampling.max_tokens;
  }

  LatencyMetrics metrics() const {
    LatencyMetrics m;
    m.prompt_tokens = num_prompt_tokens();
    m.generated_tokens = static_cast<int>(output_tokens.size());
    if (has_first_token) {
      m.ttft_ms = std::chrono::duration<double, std::milli>(first_token_at -
                                                           created_at)
                      .count();
      const auto now = std::chrono::steady_clock::now();
      const double elapsed_s =
          std::chrono::duration<double>(now - first_token_at).count();
      if (elapsed_s > 0.0 && m.generated_tokens > 1) {
        m.tokens_per_second = (m.generated_tokens - 1) / elapsed_s;
      }
    }
    return m;
  }
};

}  // namespace llm
