#include "engine/kv_cache.hpp"

#include <stdexcept>

namespace llm {

PagedKvCache::PagedKvCache(std::shared_ptr<BlockManager> blocks,
                           const ModelConfig& model, DeviceType device)
    : blocks_(std::move(blocks)),
      device_(device),
      block_size_(blocks_->block_size()),
      num_layers_(model.num_layers),
      num_kv_heads_(model.num_kv_heads),
      head_dim_(model.head_dim),
      num_blocks_(blocks_->num_blocks()) {
  page_stride_ = static_cast<std::size_t>(block_size_) * num_kv_heads_ *
                 head_dim_;
  // Layout: [num_layers][num_blocks][2][page]
  const std::size_t total =
      static_cast<std::size_t>(num_layers_) * num_blocks_ * 2 * page_stride_;
  data_.assign(total, 0.0f);
}

std::size_t PagedKvCache::page_offset(int layer, BlockId block_id,
                                      bool is_v) const {
  if (layer < 0 || layer >= num_layers_ || block_id < 0 ||
      block_id >= num_blocks_) {
    throw std::out_of_range("PagedKvCache::page_offset");
  }
  const std::size_t layer_stride = static_cast<std::size_t>(num_blocks_) * 2 *
                                   page_stride_;
  return static_cast<std::size_t>(layer) * layer_stride +
         static_cast<std::size_t>(block_id) * 2 * page_stride_ +
         (is_v ? page_stride_ : 0);
}

float* PagedKvCache::page_ptr(int layer, BlockId block_id, bool is_v) {
  return data_.data() + page_offset(layer, block_id, is_v);
}

const float* PagedKvCache::page_ptr(int layer, BlockId block_id,
                                    bool is_v) const {
  return data_.data() + page_offset(layer, block_id, is_v);
}

}  // namespace llm
