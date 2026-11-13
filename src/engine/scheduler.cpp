#include "engine/scheduler.hpp"

#include <algorithm>
#include <utility>

namespace llm {

Scheduler::Scheduler(EngineConfig config, std::shared_ptr<BlockManager> blocks)
    : config_(std::move(config)), blocks_(std::move(blocks)) {}

SeqId Scheduler::add_request(std::string request_id, std::string prompt,
                             std::vector<TokenId> prompt_tokens,
                             SamplingParams sampling) {
  std::lock_guard<std::mutex> lock(mu_);
  Sequence seq;
  seq.id = next_id_++;
  seq.request_id = std::move(request_id);
  seq.prompt_text = std::move(prompt);
  seq.prompt_tokens = std::move(prompt_tokens);
  seq.sampling = sampling;
  seq.status = SequenceStatus::kWaiting;
  seq.created_at = std::chrono::steady_clock::now();
  const SeqId id = seq.id;
  sequences_.emplace(id, std::move(seq));
  waiting_.push_back(id);
  return id;
}

bool Scheduler::try_allocate(Sequence& seq) {
  const int tokens = seq.num_tokens();
  return blocks_->append_slots(seq.id, tokens, seq.block_table);
}

void Scheduler::free_sequence(Sequence& seq) {
  blocks_->free(seq.block_table);
  seq.block_table.clear();
}

ScheduleBatch Scheduler::schedule() {
  std::lock_guard<std::mutex> lock(mu_);
  ScheduleBatch batch;

  // Promote waiting -> running while budget allows (continuous batching admit).
  while (!waiting_.empty()) {
    const SeqId id = waiting_.front();
    auto it = sequences_.find(id);
    if (it == sequences_.end()) {
      waiting_.pop_front();
      continue;
    }
    Sequence& seq = it->second;
    const int prompt_tokens = seq.num_prompt_tokens();
    if (batch.num_batched_tokens + prompt_tokens >
        config_.max_num_batched_tokens) {
      break;
    }
    if (static_cast<int>(batch.sequences.size()) >= config_.max_batch_size) {
      break;
    }
    if (!try_allocate(seq)) {
      break;
    }
    seq.status = SequenceStatus::kRunning;
    waiting_.pop_front();
    running_.push_back(id);
    batch.sequences.push_back(&seq);
    batch.num_batched_tokens += prompt_tokens;
    batch.contains_prefill = true;
  }

  // Add already-running decode sequences (1 token each).
  for (SeqId id : running_) {
    auto it = sequences_.find(id);
    if (it == sequences_.end()) {
      continue;
    }
    Sequence& seq = it->second;
    if (seq.status != SequenceStatus::kRunning) {
      continue;
    }
    // Already added as fresh prefill above.
    if (std::find(batch.sequences.begin(), batch.sequences.end(), &seq) !=
        batch.sequences.end()) {
      continue;
    }
    if (static_cast<int>(batch.sequences.size()) >= config_.max_batch_size) {
      break;
    }
    if (batch.num_batched_tokens + 1 > config_.max_num_batched_tokens) {
      break;
    }
    // Ensure a slot for the next generated token.
    if (!blocks_->append_slots(seq.id, seq.num_tokens() + 1, seq.block_table)) {
      continue;
    }
    batch.sequences.push_back(&seq);
    batch.num_batched_tokens += 1;
  }

  return batch;
}

bool Scheduler::on_token_generated(SeqId id, TokenId token, bool stop) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = sequences_.find(id);
  if (it == sequences_.end()) {
    return true;
  }
  Sequence& seq = it->second;
  seq.output_tokens.push_back(token);
  if (!seq.has_first_token) {
    seq.has_first_token = true;
    seq.first_token_at = std::chrono::steady_clock::now();
  }
  seq.num_computed_tokens = seq.num_tokens();

  if (stop || seq.finished()) {
    seq.status = SequenceStatus::kFinished;
    free_sequence(seq);
    running_.erase(std::remove(running_.begin(), running_.end(), id),
                   running_.end());
    finished_.emplace(id, std::move(seq));
    sequences_.erase(it);
    return true;
  }
  return false;
}

void Scheduler::abort(SeqId id) {
  std::lock_guard<std::mutex> lock(mu_);
  waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), id),
                 waiting_.end());
  running_.erase(std::remove(running_.begin(), running_.end(), id),
                 running_.end());
  auto it = sequences_.find(id);
  if (it == sequences_.end()) {
    return;
  }
  it->second.status = SequenceStatus::kAborted;
  free_sequence(it->second);
  finished_.emplace(id, std::move(it->second));
  sequences_.erase(it);
}

std::optional<Sequence> Scheduler::take_finished(SeqId id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = finished_.find(id);
  if (it == finished_.end()) {
    return std::nullopt;
  }
  Sequence seq = std::move(it->second);
  finished_.erase(it);
  return seq;
}

std::vector<Sequence> Scheduler::drain_finished() {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<Sequence> out;
  out.reserve(finished_.size());
  for (auto& [_, seq] : finished_) {
    out.push_back(std::move(seq));
  }
  finished_.clear();
  return out;
}

int Scheduler::num_waiting() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(waiting_.size());
}

int Scheduler::num_running() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(running_.size());
}

int Scheduler::num_finished() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(finished_.size());
}

}  // namespace llm
