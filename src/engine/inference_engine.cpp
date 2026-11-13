#include "engine/inference_engine.hpp"

#include "kernels/ops.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

namespace llm {
namespace {

void apply_linear(const LinearWeights& w, const float* x, float* y, int batch) {
  if (w.quantized) {
    std::vector<int8_t> x_q(static_cast<std::size_t>(batch) * w.in_features);
    std::vector<float> x_scales(static_cast<std::size_t>(batch));
    kernels::quantize_int8(x, x_q.data(), x_scales.data(), batch, w.in_features);
    kernels::gemm_int8(x_q.data(), w.weight_int8.data(), x_scales.data(),
                       w.scales.data(), y, batch, w.out_features,
                       w.in_features);
  } else {
    kernels::gemm_f32(x, w.weight.data(), y, batch, w.out_features,
                      w.in_features);
  }
}

void silu_mul(float* gate, const float* up, int n) {
  for (int i = 0; i < n; ++i) {
    const float x = gate[i];
    const float s = x / (1.0f + std::exp(-x));
    gate[i] = s * up[i];
  }
}

const float* layer_kv_base(const PagedKvCache& cache, int layer, int num_blocks,
                           int block_size, int num_kv_heads, int head_dim) {
  const std::size_t page =
      static_cast<std::size_t>(block_size) * num_kv_heads * head_dim;
  const std::size_t layer_stride =
      static_cast<std::size_t>(num_blocks) * 2 * page;
  return cache.data() + static_cast<std::size_t>(layer) * layer_stride;
}

}  // namespace

InferenceEngine::InferenceEngine(EngineConfig config)
    : config_(std::move(config)) {
  const auto& m = config_.model;
  blocks_ = std::make_shared<BlockManager>(
      config_.num_kv_blocks, config_.block_size, m.num_layers, m.num_kv_heads,
      m.head_dim);
  kv_cache_ = std::make_unique<PagedKvCache>(blocks_, m, config_.device);
  scheduler_ = std::make_unique<Scheduler>(config_, blocks_);
  weights_ = ModelWeights::create_random(m);
  tokenizer_ = std::make_unique<Tokenizer>(m.vocab_size);
  sampler_ = std::make_unique<Sampler>();
  hidden_.resize(static_cast<std::size_t>(config_.max_num_batched_tokens) *
                 m.hidden_size);
  logits_.resize(static_cast<std::size_t>(m.vocab_size));
}

InferenceEngine::~InferenceEngine() { stop(); }

void InferenceEngine::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  worker_ = std::thread([this] { worker_loop(); });
}

void InferenceEngine::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  wake_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

SeqId InferenceEngine::submit(std::string request_id, std::string prompt,
                              SamplingParams sampling, TokenCallback on_token) {
  auto tokens = tokenizer_->encode(prompt);
  const SeqId id = scheduler_->add_request(std::move(request_id),
                                           std::move(prompt), std::move(tokens),
                                           sampling);
  if (on_token) {
    std::lock_guard<std::mutex> lock(cb_mu_);
    callbacks_[id] = std::move(on_token);
  }
  wake_cv_.notify_one();
  return id;
}

std::optional<GenerateResult> InferenceEngine::poll_finished(SeqId id) {
  auto seq = scheduler_->take_finished(id);
  if (!seq) {
    return std::nullopt;
  }
  GenerateResult r;
  r.request_id = seq->request_id;
  r.token_ids = seq->output_tokens;
  r.text = tokenizer_->decode(seq->output_tokens);
  r.finished = true;
  r.latency = seq->metrics();
  return r;
}

GenerateResult InferenceEngine::generate_blocking(std::string request_id,
                                                  std::string prompt,
                                                  SamplingParams sampling) {
  const SeqId id =
      submit(std::move(request_id), std::move(prompt), sampling, nullptr);
  for (;;) {
    if (auto r = poll_finished(id)) {
      return *r;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

HealthSnapshot InferenceEngine::health() const {
  HealthSnapshot h;
  h.ready = running_.load();
  h.device = kernels::device_name();
  h.model_name = config_.model.name;
  return h;
}

StatsSnapshot InferenceEngine::stats() const {
  StatsSnapshot s;
  s.active_sequences = scheduler_->num_running();
  s.waiting_sequences = scheduler_->num_waiting();
  s.kv_blocks_used = blocks_->used_blocks();
  s.kv_blocks_total = blocks_->num_blocks();
  s.gpu_utilization = static_cast<double>(s.kv_blocks_used) /
                       std::max(1, s.kv_blocks_total);
  s.total_tokens_generated = total_tokens_generated_.load();
  return s;
}

void InferenceEngine::worker_loop() {
  while (running_.load()) {
    auto batch = scheduler_->schedule();
    if (batch.sequences.empty()) {
      std::unique_lock<std::mutex> lock(wake_mu_);
      wake_cv_.wait_for(lock, std::chrono::milliseconds(2),
                        [this] { return !running_.load(); });
      continue;
    }
    step(batch);
  }
}

void InferenceEngine::run_prefill(Sequence& seq) {
  const auto& cfg = config_.model;
  const int n = seq.num_prompt_tokens();
  const SeqId seq_id = seq.id;
  const std::string request_id = seq.request_id;
  const SamplingParams sampling = seq.sampling;
  const std::vector<BlockId> block_table = seq.block_table;

  std::vector<float> x(static_cast<std::size_t>(n) * cfg.hidden_size);
  for (int t = 0; t < n; ++t) {
    const TokenId tok = seq.prompt_tokens[static_cast<std::size_t>(t)];
    const float* emb = weights_->tok_embeddings.data() +
                       static_cast<std::size_t>(tok) * cfg.hidden_size;
    std::memcpy(x.data() + static_cast<std::size_t>(t) * cfg.hidden_size, emb,
                sizeof(float) * cfg.hidden_size);
  }

  std::vector<float> q(static_cast<std::size_t>(n) * cfg.num_heads *
                       cfg.head_dim);
  std::vector<float> k(static_cast<std::size_t>(n) * cfg.num_kv_heads *
                       cfg.head_dim);
  std::vector<float> v(static_cast<std::size_t>(n) * cfg.num_kv_heads *
                       cfg.head_dim);
  std::vector<float> attn_out(static_cast<std::size_t>(n) * cfg.num_heads *
                              cfg.head_dim);
  std::vector<float> residual(x.size());
  std::vector<float> normed(x.size());
  std::vector<float> gate(static_cast<std::size_t>(n) * cfg.intermediate_size);
  std::vector<float> up(static_cast<std::size_t>(n) * cfg.intermediate_size);
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const auto& lw = weights_->layers[static_cast<std::size_t>(layer)];
    residual = x;
    for (int t = 0; t < n; ++t) {
      kernels::rms_norm(
          normed.data() + static_cast<std::size_t>(t) * cfg.hidden_size,
          x.data() + static_cast<std::size_t>(t) * cfg.hidden_size,
          lw.attn_norm.data(), cfg.hidden_size, 1e-5f);
    }
    apply_linear(lw.attn.wq, normed.data(), q.data(), n);
    apply_linear(lw.attn.wk, normed.data(), k.data(), n);
    apply_linear(lw.attn.wv, normed.data(), v.data(), n);
    kernels::rope_inplace(q.data(), k.data(), n, cfg.num_heads, cfg.num_kv_heads,
                          cfg.head_dim, 0, cfg.rope_theta);
    kernels::write_kv_cache(kv_cache_->mutable_data(), k.data(), v.data(), layer,
                            n, cfg.num_kv_heads, cfg.head_dim,
                            config_.block_size, block_table, 0, cfg.num_layers,
                            blocks_->num_blocks());

    const float* layer_base =
        layer_kv_base(*kv_cache_, layer, blocks_->num_blocks(),
                      config_.block_size, cfg.num_kv_heads, cfg.head_dim);
    std::vector<std::vector<BlockId>> tables{block_table};
    for (int t = 0; t < n; ++t) {
      std::vector<int> lens{t + 1};
      kernels::paged_attention(
          attn_out.data() +
              static_cast<std::size_t>(t) * cfg.num_heads * cfg.head_dim,
          q.data() + static_cast<std::size_t>(t) * cfg.num_heads * cfg.head_dim,
          layer_base, 1, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
          config_.block_size, tables, lens, scale);
    }

    apply_linear(lw.attn.wo, attn_out.data(), x.data(), n);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += residual[i];
    }

    residual = x;
    for (int t = 0; t < n; ++t) {
      kernels::rms_norm(
          normed.data() + static_cast<std::size_t>(t) * cfg.hidden_size,
          x.data() + static_cast<std::size_t>(t) * cfg.hidden_size,
          lw.ffn_norm.data(), cfg.hidden_size, 1e-5f);
    }
    apply_linear(lw.mlp.gate, normed.data(), gate.data(), n);
    apply_linear(lw.mlp.up, normed.data(), up.data(), n);
    silu_mul(gate.data(), up.data(), n * cfg.intermediate_size);
    apply_linear(lw.mlp.down, gate.data(), x.data(), n);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += residual[i];
    }
  }

  const float* last =
      x.data() + static_cast<std::size_t>(n - 1) * cfg.hidden_size;
  std::vector<float> final_h(static_cast<std::size_t>(cfg.hidden_size));
  kernels::rms_norm(final_h.data(), last, weights_->norm.data(), cfg.hidden_size,
                    1e-5f);
  apply_linear(weights_->lm_head, final_h.data(), logits_.data(), 1);

  const TokenId tok =
      sampler_->sample(logits_.data(), cfg.vocab_size, sampling);
  const bool stop =
      (tok == tokenizer_->eos_id()) || (1 >= sampling.max_tokens);

  TokenCallback cb;
  {
    std::lock_guard<std::mutex> lock(cb_mu_);
    auto it = callbacks_.find(seq_id);
    if (it != callbacks_.end()) {
      cb = it->second;
    }
  }

  const bool finished = scheduler_->on_token_generated(seq_id, tok, stop);
  total_tokens_generated_.fetch_add(1);

  if (finished) {
    std::lock_guard<std::mutex> lock(cb_mu_);
    callbacks_.erase(seq_id);
  }

  if (cb) {
    GenerateResult r;
    r.request_id = request_id;
    r.token_ids = {tok};
    r.text = tokenizer_->decode_token(tok);
    r.finished = finished;
    cb(r);
  }
}

void InferenceEngine::run_decode_batch(const std::vector<Sequence*>& seqs) {
  const auto& cfg = config_.model;
  const int batch = static_cast<int>(seqs.size());

  struct SeqSnap {
    SeqId id;
    std::string request_id;
    SamplingParams sampling;
    std::vector<BlockId> block_table;
    TokenId last_token;
    int write_pos;
    int context_len;
  };
  std::vector<SeqSnap> snaps;
  snaps.reserve(static_cast<std::size_t>(batch));
  for (Sequence* seq : seqs) {
    SeqSnap s;
    s.id = seq->id;
    s.request_id = seq->request_id;
    s.sampling = seq->sampling;
    s.block_table = seq->block_table;
    s.last_token = seq->last_token();
    s.context_len = seq->num_tokens();
    s.write_pos = seq->num_tokens() - 1;
    snaps.push_back(std::move(s));
  }

  std::vector<float> x(static_cast<std::size_t>(batch) * cfg.hidden_size);
  for (int b = 0; b < batch; ++b) {
    const float* emb =
        weights_->tok_embeddings.data() +
        static_cast<std::size_t>(snaps[static_cast<std::size_t>(b)].last_token) *
            cfg.hidden_size;
    std::memcpy(x.data() + static_cast<std::size_t>(b) * cfg.hidden_size, emb,
                sizeof(float) * cfg.hidden_size);
  }

  std::vector<float> q(static_cast<std::size_t>(batch) * cfg.num_heads *
                       cfg.head_dim);
  std::vector<float> k(static_cast<std::size_t>(batch) * cfg.num_kv_heads *
                       cfg.head_dim);
  std::vector<float> v(static_cast<std::size_t>(batch) * cfg.num_kv_heads *
                       cfg.head_dim);
  std::vector<float> attn_out(static_cast<std::size_t>(batch) * cfg.num_heads *
                              cfg.head_dim);
  std::vector<float> residual(x.size());
  std::vector<float> normed(x.size());
  std::vector<float> gate(static_cast<std::size_t>(batch) *
                          cfg.intermediate_size);
  std::vector<float> up(static_cast<std::size_t>(batch) * cfg.intermediate_size);

  std::vector<std::vector<BlockId>> tables;
  std::vector<int> ctx_lens;
  tables.reserve(static_cast<std::size_t>(batch));
  for (const auto& s : snaps) {
    tables.push_back(s.block_table);
    ctx_lens.push_back(s.context_len);
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const auto& lw = weights_->layers[static_cast<std::size_t>(layer)];
    residual = x;
    for (int b = 0; b < batch; ++b) {
      kernels::rms_norm(
          normed.data() + static_cast<std::size_t>(b) * cfg.hidden_size,
          x.data() + static_cast<std::size_t>(b) * cfg.hidden_size,
          lw.attn_norm.data(), cfg.hidden_size, 1e-5f);
    }
    apply_linear(lw.attn.wq, normed.data(), q.data(), batch);
    apply_linear(lw.attn.wk, normed.data(), k.data(), batch);
    apply_linear(lw.attn.wv, normed.data(), v.data(), batch);

    for (int b = 0; b < batch; ++b) {
      float* qb = q.data() + static_cast<std::size_t>(b) * cfg.num_heads *
                                 cfg.head_dim;
      float* kb = k.data() + static_cast<std::size_t>(b) * cfg.num_kv_heads *
                                 cfg.head_dim;
      kernels::rope_inplace(qb, kb, 1, cfg.num_heads, cfg.num_kv_heads,
                            cfg.head_dim,
                            snaps[static_cast<std::size_t>(b)].write_pos,
                            cfg.rope_theta);
    }

    for (int b = 0; b < batch; ++b) {
      kernels::write_kv_cache(
          kv_cache_->mutable_data(),
          k.data() +
              static_cast<std::size_t>(b) * cfg.num_kv_heads * cfg.head_dim,
          v.data() +
              static_cast<std::size_t>(b) * cfg.num_kv_heads * cfg.head_dim,
          layer, 1, cfg.num_kv_heads, cfg.head_dim, config_.block_size,
          snaps[static_cast<std::size_t>(b)].block_table,
          snaps[static_cast<std::size_t>(b)].write_pos, cfg.num_layers,
          blocks_->num_blocks());
    }

    const float* layer_base =
        layer_kv_base(*kv_cache_, layer, blocks_->num_blocks(),
                      config_.block_size, cfg.num_kv_heads, cfg.head_dim);
    kernels::paged_attention(attn_out.data(), q.data(), layer_base, batch,
                             cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                             config_.block_size, tables, ctx_lens, scale);

    apply_linear(lw.attn.wo, attn_out.data(), x.data(), batch);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += residual[i];
    }

    residual = x;
    for (int b = 0; b < batch; ++b) {
      kernels::rms_norm(
          normed.data() + static_cast<std::size_t>(b) * cfg.hidden_size,
          x.data() + static_cast<std::size_t>(b) * cfg.hidden_size,
          lw.ffn_norm.data(), cfg.hidden_size, 1e-5f);
    }
    apply_linear(lw.mlp.gate, normed.data(), gate.data(), batch);
    apply_linear(lw.mlp.up, normed.data(), up.data(), batch);
    silu_mul(gate.data(), up.data(), batch * cfg.intermediate_size);
    apply_linear(lw.mlp.down, gate.data(), x.data(), batch);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += residual[i];
    }
  }

  for (int b = 0; b < batch; ++b) {
    const auto& snap = snaps[static_cast<std::size_t>(b)];
    std::vector<float> final_h(static_cast<std::size_t>(cfg.hidden_size));
    kernels::rms_norm(
        final_h.data(),
        x.data() + static_cast<std::size_t>(b) * cfg.hidden_size,
        weights_->norm.data(), cfg.hidden_size, 1e-5f);
    apply_linear(weights_->lm_head, final_h.data(), logits_.data(), 1);

    const TokenId tok =
        sampler_->sample(logits_.data(), cfg.vocab_size, snap.sampling);
    const bool stop = (tok == tokenizer_->eos_id());

    TokenCallback cb;
    {
      std::lock_guard<std::mutex> lock(cb_mu_);
      auto it = callbacks_.find(snap.id);
      if (it != callbacks_.end()) {
        cb = it->second;
      }
    }

    const bool finished = scheduler_->on_token_generated(snap.id, tok, stop);
    total_tokens_generated_.fetch_add(1);

    if (finished) {
      std::lock_guard<std::mutex> lock(cb_mu_);
      callbacks_.erase(snap.id);
    }

    if (cb) {
      GenerateResult r;
      r.request_id = snap.request_id;
      r.token_ids = {tok};
      r.text = tokenizer_->decode_token(tok);
      r.finished = finished;
      cb(r);
    }
  }
}

void InferenceEngine::step(ScheduleBatch& batch) {
  std::vector<Sequence*> prefills;
  std::vector<Sequence*> decodes;
  for (Sequence* seq : batch.sequences) {
    if (seq->is_prefill()) {
      prefills.push_back(seq);
    } else {
      decodes.push_back(seq);
    }
  }
  for (Sequence* seq : prefills) {
    run_prefill(*seq);
  }
  if (!decodes.empty()) {
    run_decode_batch(decodes);
  }
}

}  // namespace llm
