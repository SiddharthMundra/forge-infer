#pragma once

#include "common/types.hpp"

#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llm {

// Physical KV page. Contiguous [block_size, num_kv_heads, head_dim] slots.
struct KvBlock {
  BlockId id = -1;
  int ref_count = 0;
};

// PagedAttention allocator: logical tokens map to fixed-size physical blocks.
// Eliminates contiguous-allocation fragmentation and raises concurrent capacity.
class BlockManager {
 public:
  BlockManager(int num_blocks, int block_size, int num_layers, int num_kv_heads,
               int head_dim);

  int block_size() const { return block_size_; }
  int num_blocks() const { return static_cast<int>(blocks_.size()); }
  int free_blocks() const;
  int used_blocks() const { return num_blocks() - free_blocks(); }

  // Allocate `n` free blocks; returns empty optional if insufficient.
  std::optional<std::vector<BlockId>> allocate(int n);

  void free(const std::vector<BlockId>& ids);
  void free(BlockId id);

  // Grow a sequence's block table to cover `num_tokens` tokens.
  bool append_slots(SeqId seq_id, int num_tokens,
                    std::vector<BlockId>& block_table);

  // Bytes reserved for all KV pages (K+V across layers).
  std::size_t bytes_reserved() const;

 private:
  int block_size_;
  int num_layers_;
  int num_kv_heads_;
  int head_dim_;
  std::vector<KvBlock> blocks_;
  std::vector<BlockId> free_list_;
  mutable std::mutex mu_;
};

}  // namespace llm
