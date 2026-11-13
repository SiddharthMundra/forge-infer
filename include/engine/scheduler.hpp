#pragma once

#include "engine/block_manager.hpp"
#include "engine/sequence.hpp"

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llm {

struct ScheduleBatch {
  std::vector<Sequence*> sequences;
  int num_batched_tokens = 0;
  bool contains_prefill = false;
};

// Continuous batching scheduler (vLLM-style):
// admit waiting seqs when KV blocks + token budget allow, step all running
// sequences together each iteration, and free finished ones immediately so
// new work can join mid-generation.
class Scheduler {
 public:
  Scheduler(EngineConfig config, std::shared_ptr<BlockManager> blocks);

  SeqId add_request(std::string request_id, std::string prompt,
                    std::vector<TokenId> prompt_tokens, SamplingParams sampling);

  // Build the next iteration batch. Empty if nothing runnable.
  ScheduleBatch schedule();

  // Returns true if the sequence is now finished.
  bool on_token_generated(SeqId id, TokenId token, bool stop);
  void abort(SeqId id);

  std::optional<Sequence> take_finished(SeqId id);
  std::vector<Sequence> drain_finished();

  int num_waiting() const;
  int num_running() const;
  int num_finished() const;

 private:
  bool try_allocate(Sequence& seq);
  void free_sequence(Sequence& seq);

  EngineConfig config_;
  std::shared_ptr<BlockManager> blocks_;
  SeqId next_id_ = 1;

  std::deque<SeqId> waiting_;
  std::deque<SeqId> running_;
  std::unordered_map<SeqId, Sequence> sequences_;
  std::unordered_map<SeqId, Sequence> finished_;
  mutable std::mutex mu_;
};

}  // namespace llm
