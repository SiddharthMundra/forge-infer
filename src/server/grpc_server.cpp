#include "server/grpc_server.hpp"

#include "inference.grpc.pb.h"
#include "inference.pb.h"

#include <grpcpp/grpcpp.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace llm {
namespace {

class InferenceServiceImpl final : public ::llm::inference::v1::InferenceService::Service {
 public:
  explicit InferenceServiceImpl(std::shared_ptr<InferenceEngine> engine)
      : engine_(std::move(engine)) {}

  grpc::Status Generate(grpc::ServerContext* /*context*/,
                        const ::llm::inference::v1::GenerateRequest* request,
                        ::llm::inference::v1::GenerateResponse* response) override {
    SamplingParams sp;
    sp.max_tokens = request->max_tokens() > 0 ? request->max_tokens() : 64;
    sp.temperature = request->temperature() > 0 ? request->temperature() : 0.8f;
    sp.top_p = request->top_p() > 0 ? request->top_p() : 0.95f;
    sp.top_k = request->top_k() > 0 ? request->top_k() : 50;

    const std::string rid =
        request->request_id().empty() ? "req" : request->request_id();
    auto result =
        engine_->generate_blocking(rid, request->prompt(), sp);

    response->set_request_id(result.request_id);
    response->set_text(result.text);
    for (TokenId t : result.token_ids) {
      response->add_token_ids(t);
    }
    response->set_finished(true);
    auto* lat = response->mutable_latency();
    lat->set_time_to_first_token_ms(result.latency.ttft_ms);
    lat->set_tokens_per_second(result.latency.tokens_per_second);
    lat->set_prompt_tokens(result.latency.prompt_tokens);
    lat->set_generated_tokens(result.latency.generated_tokens);
    return grpc::Status::OK;
  }

  grpc::Status GenerateStream(
      grpc::ServerContext* context,
      const ::llm::inference::v1::GenerateRequest* request,
      grpc::ServerWriter<::llm::inference::v1::GenerateResponse>* writer)
      override {
    SamplingParams sp;
    sp.max_tokens = request->max_tokens() > 0 ? request->max_tokens() : 64;
    sp.temperature = request->temperature() > 0 ? request->temperature() : 0.8f;
    sp.top_p = request->top_p() > 0 ? request->top_p() : 0.95f;
    sp.top_k = request->top_k() > 0 ? request->top_k() : 50;

    const std::string rid =
        request->request_id().empty() ? "stream" : request->request_id();

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    grpc::Status status = grpc::Status::OK;

    engine_->submit(
        rid, request->prompt(), sp,
        [&](const GenerateResult& chunk) {
          if (context->IsCancelled()) {
            std::lock_guard<std::mutex> lock(mu);
            done = true;
            status = grpc::Status(grpc::StatusCode::CANCELLED, "cancelled");
            cv.notify_one();
            return;
          }
          ::llm::inference::v1::GenerateResponse resp;
          resp.set_request_id(chunk.request_id);
          resp.set_text(chunk.text);
          for (TokenId t : chunk.token_ids) {
            resp.add_token_ids(t);
          }
          resp.set_finished(chunk.finished);
          auto* lat = resp.mutable_latency();
          lat->set_time_to_first_token_ms(chunk.latency.ttft_ms);
          lat->set_tokens_per_second(chunk.latency.tokens_per_second);
          lat->set_prompt_tokens(chunk.latency.prompt_tokens);
          lat->set_generated_tokens(chunk.latency.generated_tokens);
          if (!writer->Write(resp)) {
            std::lock_guard<std::mutex> lock(mu);
            done = true;
            status = grpc::Status(grpc::StatusCode::UNKNOWN, "write failed");
            cv.notify_one();
            return;
          }
          if (chunk.finished) {
            std::lock_guard<std::mutex> lock(mu);
            done = true;
            cv.notify_one();
          }
        });

    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return done || context->IsCancelled(); });
    return status;
  }

  grpc::Status Health(grpc::ServerContext* /*context*/,
                      const ::llm::inference::v1::HealthRequest* /*request*/,
                      ::llm::inference::v1::HealthResponse* response) override {
    const auto h = engine_->health();
    response->set_ready(h.ready);
    response->set_device(h.device);
    response->set_model_name(h.model_name);
    return grpc::Status::OK;
  }

  grpc::Status Stats(grpc::ServerContext* /*context*/,
                     const ::llm::inference::v1::StatsRequest* /*request*/,
                     ::llm::inference::v1::StatsResponse* response) override {
    const auto s = engine_->stats();
    response->set_active_sequences(s.active_sequences);
    response->set_waiting_sequences(s.waiting_sequences);
    response->set_kv_blocks_used(s.kv_blocks_used);
    response->set_kv_blocks_total(s.kv_blocks_total);
    response->set_gpu_utilization(s.gpu_utilization);
    response->set_total_tokens_generated(s.total_tokens_generated);
    return grpc::Status::OK;
  }

 private:
  std::shared_ptr<InferenceEngine> engine_;
};

}  // namespace

struct GrpcServer::Impl {
  std::shared_ptr<InferenceEngine> engine;
  std::string listen_addr;
  std::unique_ptr<InferenceServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
};

GrpcServer::GrpcServer(std::shared_ptr<InferenceEngine> engine,
                       std::string listen_addr)
    : impl_(std::make_unique<Impl>()) {
  impl_->engine = std::move(engine);
  impl_->listen_addr = std::move(listen_addr);
}

GrpcServer::~GrpcServer() { shutdown(); }

void GrpcServer::start() {
  impl_->service = std::make_unique<InferenceServiceImpl>(impl_->engine);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(impl_->listen_addr,
                           grpc::InsecureServerCredentials());
  builder.RegisterService(impl_->service.get());
  impl_->server = builder.BuildAndStart();
  if (!impl_->server) {
    throw std::runtime_error("failed to start gRPC server on " +
                             impl_->listen_addr);
  }
  std::cout << "gRPC listening on " << impl_->listen_addr << std::endl;
}

void GrpcServer::wait() {
  if (impl_->server) {
    impl_->server->Wait();
  }
}

void GrpcServer::shutdown() {
  if (impl_->server) {
    impl_->server->Shutdown();
    impl_->server.reset();
  }
}

}  // namespace llm
