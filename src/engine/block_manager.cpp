#include "engine/block_manager.hpp"

#include <stdexcept>

namespace llm {

BlockManager::BlockManager(int num_blocks, int block_size, int num_layers,
                           int num_kv_heads, int head_dim)
    : block_size_(block_size),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      blocks_(static_cast<std::size_t>(num_blocks)) {
  if (num_blocks <= 0 || block_size <= 0) {
    throw std::invalid_argument("BlockManager: invalid geometry");
  }
  free_list_.reserve(static_cast<std::size_t>(num_blocks));
  for (int i = 0; i < num_blocks; ++i) {
    blocks_[static_cast<std::size_t>(i)].id = i;
    free_list_.push_back(i);
  }
}

int BlockManager::free_blocks() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(free_list_.size());
}

std::optional<std::vector<BlockId>> BlockManager::allocate(int n) {
  std::lock_guard<std::mutex> lock(mu_);
  if (n <= 0) {
    return std::vector<BlockId>{};
  }
  if (static_cast<int>(free_list_.size()) < n) {
    return std::nullopt;
  }
  std::vector<BlockId> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const BlockId id = free_list_.back();
    free_list_.pop_back();
    blocks_[static_cast<std::size_t>(id)].ref_count = 1;
    out.push_back(id);
  }
  return out;
}

void BlockManager::free(BlockId id) {
  std::lock_guard<std::mutex> lock(mu_);
  if (id < 0 || id >= static_cast<int>(blocks_.size())) {
    return;
  }
  auto& block = blocks_[static_cast<std::size_t>(id)];
  if (block.ref_count <= 0) {
    return;
  }
  --block.ref_count;
  if (block.ref_count == 0) {
    free_list_.push_back(id);
  }
}

void BlockManager::free(const std::vector<BlockId>& ids) {
  for (BlockId id : ids) {
    free(id);
  }
}

bool BlockManager::append_slots(SeqId /*seq_id*/, int num_tokens,
                                std::vector<BlockId>& block_table) {
  const int needed_blocks =
      (num_tokens + block_size_ - 1) / block_size_;
  const int have = static_cast<int>(block_table.size());
  if (needed_blocks <= have) {
    return true;
  }
  auto extra = allocate(needed_blocks - have);
  if (!extra) {
    return false;
  }
  block_table.insert(block_table.end(), extra->begin(), extra->end());
  return true;
}

std::size_t BlockManager::bytes_reserved() const {
  // K + V per layer per block.
  const std::size_t page =
      static_cast<std::size_t>(block_size_) * num_kv_heads_ * head_dim_;
  return static_cast<std::size_t>(num_blocks()) * 2 * num_layers_ * page *
         sizeof(float);
}

}  // namespace llm
