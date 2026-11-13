#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

using TokenId = int32_t;
using SeqId = int64_t;
using BlockId = int32_t;

enum class DeviceType { kCpu, kCuda };

enum class SequenceStatus {
  kWaiting,
  kRunning,
  kSwapped,
  kFinished,
  kAborted,
};

struct ModelConfig {
  std::string name = "toy-llama";
  int vocab_size = 32000;
  int hidden_size = 512;
  int num_layers = 4;
  int num_heads = 8;
  int num_kv_heads = 8;
  int head_dim = 64;
  int intermediate_size = 1376;
  int max_seq_len = 2048;
  float rope_theta = 10000.0f;
  bool use_int8 = true;
};

struct EngineConfig {
  ModelConfig model;
  DeviceType device = DeviceType::kCpu;
  int max_batch_size = 32;
  int max_num_batched_tokens = 2048;
  int block_size = 16;
  int num_kv_blocks = 1024;
  int gpu_memory_utilization_pct = 85;
  std::string listen_addr = "0.0.0.0:50051";
};

struct SamplingParams {
  float temperature = 0.8f;
  float top_p = 0.95f;
  int top_k = 50;
  int max_tokens = 128;
};

struct LatencyMetrics {
  double ttft_ms = 0.0;
  double tokens_per_second = 0.0;
  int prompt_tokens = 0;
  int generated_tokens = 0;
};

}  // namespace llm
