#pragma once

#include "common/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

// Minimal whitespace / byte-fallback tokenizer for the toy model path.
// Swap for SentencePiece / tiktoken when loading a real checkpoint.
class Tokenizer {
 public:
  explicit Tokenizer(int vocab_size = 32000);

  std::vector<TokenId> encode(const std::string& text) const;
  std::string decode(const std::vector<TokenId>& tokens) const;
  std::string decode_token(TokenId id) const;

  TokenId bos_id() const { return bos_id_; }
  TokenId eos_id() const { return eos_id_; }
  TokenId pad_id() const { return pad_id_; }
  int vocab_size() const { return vocab_size_; }

 private:
  int vocab_size_;
  TokenId bos_id_ = 1;
  TokenId eos_id_ = 2;
  TokenId pad_id_ = 0;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::vector<std::string> id_to_token_;
};

}  // namespace llm
