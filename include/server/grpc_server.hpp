#pragma once

#include "engine/inference_engine.hpp"

#include <memory>
#include <string>

namespace llm {

class GrpcServer {
 public:
  GrpcServer(std::shared_ptr<InferenceEngine> engine, std::string listen_addr);
  ~GrpcServer();

  void start();
  void wait();
  void shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llm
