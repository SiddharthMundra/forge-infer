#pragma once

#include "common/types.hpp"
#include "engine/block_manager.hpp"
#include "engine/kv_cache.hpp"
#include "engine/scheduler.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "model/weights.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace llm {

struct GenerateResult {
  std::string request_id;
  std::string text;
  std::vector<TokenId> token_ids;
  bool finished = false;
  LatencyMetrics latency;
};

struct HealthSnapshot {
  bool ready = false;
  std::string device;
  std::string model_name;
};

struct StatsSnapshot {
  int active_sequences = 0;
  int waiting_sequences = 0;
  int kv_blocks_used = 0;
  int kv_blocks_total = 0;
  double gpu_utilization = 0.0;
  int64_t total_tokens_generated = 0;
};

using TokenCallback = std::function<void(const GenerateResult&)>;

class InferenceEngine {
 public:
  explicit InferenceEngine(EngineConfig config);
  ~InferenceEngine();

  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;

  void start();
  void stop();

  // Enqueue a request; optional callback fires on each new token (streaming).
  SeqId submit(std::string request_id, std::string prompt,
               SamplingParams sampling, TokenCallback on_token = nullptr);

  std::optional<GenerateResult> poll_finished(SeqId id);
  GenerateResult generate_blocking(std::string request_id, std::string prompt,
                                   SamplingParams sampling);

  HealthSnapshot health() const;
  StatsSnapshot stats() const;

  const EngineConfig& config() const { return config_; }
  const Tokenizer& tokenizer() const { return *tokenizer_; }

 private:
  void worker_loop();
  void step(ScheduleBatch& batch);
  void run_prefill(Sequence& seq);
  void run_decode_batch(const std::vector<Sequence*>& seqs);

  EngineConfig config_;
  std::shared_ptr<BlockManager> blocks_;
  std::unique_ptr<PagedKvCache> kv_cache_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Sampler> sampler_;

  std::unordered_map<SeqId, TokenCallback> callbacks_;
  mutable std::mutex cb_mu_;

  std::atomic<bool> running_{false};
  std::thread worker_;
  std::mutex wake_mu_;
  std::condition_variable wake_cv_;

  std::atomic<int64_t> total_tokens_generated_{0};
  std::vector<float> hidden_;
  std::vector<float> logits_;
};

}  // namespace llm
