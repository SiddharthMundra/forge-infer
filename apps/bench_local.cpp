#include "engine/inference_engine.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  llm::EngineConfig cfg;
  cfg.model.name = "toy-llama-int8";
  cfg.model.vocab_size = 2048;
  cfg.model.hidden_size = 512;
  cfg.model.num_layers = 4;
  cfg.model.num_heads = 8;
  cfg.model.num_kv_heads = 8;
  cfg.model.head_dim = 64;
  cfg.model.intermediate_size = 1376;
  cfg.model.use_int8 = true;
#if defined(LLM_HAS_CUDA)
  cfg.device = llm::DeviceType::kCuda;
  cfg.max_batch_size = 16;
  cfg.num_kv_blocks = 1024;
#else
  cfg.device = llm::DeviceType::kCpu;
  cfg.model.vocab_size = 512;
  cfg.model.hidden_size = 256;
  cfg.model.num_layers = 2;
  cfg.model.num_heads = 4;
  cfg.model.num_kv_heads = 4;
  cfg.model.intermediate_size = 512;
  cfg.max_batch_size = 8;
  cfg.num_kv_blocks = 256;
#endif
  cfg.block_size = 16;

  llm::InferenceEngine engine(cfg);
  engine.start();

  std::cout << "device=" << engine.health().device
            << " blocks=" << cfg.num_kv_blocks << std::endl;

  const int concurrency = 4;
  llm::SamplingParams sp;
  sp.max_tokens = 16;
  sp.temperature = 0.8f;

  std::vector<llm::SeqId> ids;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < concurrency; ++i) {
    ids.push_back(engine.submit(
        "bench-" + std::to_string(i),
        "hello continuous batching " + std::to_string(i), sp));
  }

  int done = 0;
  double ttft_sum = 0.0;
  while (done < concurrency) {
    for (auto it = ids.begin(); it != ids.end();) {
      if (auto r = engine.poll_finished(*it)) {
        ttft_sum += r->latency.ttft_ms;
        std::cout << r->request_id << " ttft_ms=" << r->latency.ttft_ms
                  << " tok/s=" << r->latency.tokens_per_second << " text=\""
                  << r->text << "\"\n";
        it = ids.erase(it);
        ++done;
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  const auto st = engine.stats();
  std::cout << "wall_ms=" << wall_ms
            << " avg_ttft_ms=" << (ttft_sum / concurrency)
            << " kv_used=" << st.kv_blocks_used << "/" << st.kv_blocks_total
            << " total_tokens=" << st.total_tokens_generated << std::endl;

  engine.stop();
  return 0;
}
