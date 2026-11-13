#pragma once

#include "common/types.hpp"
#include "engine/block_manager.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace llm {

// Physical KV arena: [num_blocks, 2, block_size, num_kv_heads, head_dim]
// Layout matches PagedAttention: each logical token indexes via block_table.
class PagedKvCache {
 public:
  PagedKvCache(std::shared_ptr<BlockManager> blocks, const ModelConfig& model,
               DeviceType device);

  float* mutable_data() { return data_.data(); }
  const float* data() const { return data_.data(); }

  // Offset (in floats) of K or V page for (layer, block_id, is_v).
  std::size_t page_offset(int layer, BlockId block_id, bool is_v) const;

  float* page_ptr(int layer, BlockId block_id, bool is_v);
  const float* page_ptr(int layer, BlockId block_id, bool is_v) const;

  int block_size() const { return block_size_; }
  int num_layers() const { return num_layers_; }
  DeviceType device() const { return device_; }

  std::size_t num_floats() const { return data_.size(); }
  std::size_t bytes() const { return data_.size() * sizeof(float); }

 private:
  std::shared_ptr<BlockManager> blocks_;
  DeviceType device_;
  int block_size_;
  int num_layers_;
  int num_kv_heads_;
  int head_dim_;
  int num_blocks_;
  std::size_t page_stride_;
  std::vector<float> data_;
};

}  // namespace llm
