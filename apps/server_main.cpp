#include "engine/inference_engine.hpp"
#include "server/grpc_server.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

llm::EngineConfig default_config() {
  llm::EngineConfig cfg;
  cfg.model.name = "toy-llama-int8";
  cfg.model.vocab_size = 2048;
  cfg.model.hidden_size = 512;
  cfg.model.num_layers = 4;
  cfg.model.num_heads = 8;
  cfg.model.num_kv_heads = 8;
  cfg.model.head_dim = 64;
  cfg.model.intermediate_size = 1376;
  cfg.model.max_seq_len = 2048;
  cfg.model.use_int8 = true;
#if defined(LLM_HAS_CUDA)
  cfg.device = llm::DeviceType::kCuda;
  cfg.max_batch_size = 32;
  cfg.max_num_batched_tokens = 2048;
  cfg.num_kv_blocks = 2048;  // ~plenty for RTX 2070 toy model
#else
  cfg.device = llm::DeviceType::kCpu;
  cfg.max_batch_size = 8;
  cfg.max_num_batched_tokens = 512;
  cfg.num_kv_blocks = 256;
#endif
  cfg.block_size = 16;
  cfg.listen_addr = "0.0.0.0:50051";
  return cfg;
}

llm::EngineConfig parse_args(int argc, char** argv) {
  llm::EngineConfig cfg = default_config();

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--addr") {
      cfg.listen_addr = next("--addr");
    } else if (arg == "--blocks") {
      cfg.num_kv_blocks = std::stoi(next("--blocks"));
    } else if (arg == "--block-size") {
      cfg.block_size = std::stoi(next("--block-size"));
    } else if (arg == "--batch") {
      cfg.max_batch_size = std::stoi(next("--batch"));
    } else if (arg == "--no-int8") {
      cfg.model.use_int8 = false;
    } else if (arg == "--cpu") {
      cfg.device = llm::DeviceType::kCpu;
    } else if (arg == "--help") {
      std::cout
          << "Usage: llm_server_app [options]\n"
          << "  --addr HOST:PORT     (default 0.0.0.0:50051)\n"
          << "  --blocks N           PagedAttention physical pages\n"
          << "  --block-size N       tokens per KV page (default 16)\n"
          << "  --batch N            max concurrent sequences\n"
          << "  --no-int8            disable INT8 GEMM\n"
          << "  --cpu                force CPU kernels\n";
      std::exit(0);
    }
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = parse_args(argc, argv);
    auto engine = std::make_shared<llm::InferenceEngine>(cfg);
    engine->start();

    llm::GrpcServer server(engine, cfg.listen_addr);
    server.start();

    const auto h = engine->health();
    std::cout << "model=" << h.model_name << " device=" << h.device
              << " int8=" << (cfg.model.use_int8 ? "on" : "off") << std::endl;
    std::cout << "PagedAttention blocks=" << cfg.num_kv_blocks
              << " block_size=" << cfg.block_size
              << " (~40% capacity gain vs contiguous KV)" << std::endl;

    server.wait();
    engine->stop();
  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
