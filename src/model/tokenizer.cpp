#include "model/tokenizer.hpp"

#include <sstream>

namespace llm {

Tokenizer::Tokenizer(int vocab_size) : vocab_size_(vocab_size) {
  id_to_token_.assign(static_cast<std::size_t>(vocab_size_), "<unk>");
  id_to_token_[static_cast<std::size_t>(pad_id_)] = "<pad>";
  id_to_token_[static_cast<std::size_t>(bos_id_)] = "<bos>";
  id_to_token_[static_cast<std::size_t>(eos_id_)] = "<eos>";

  // Reserve printable ASCII as single-byte tokens for a usable decode path.
  for (int c = 32; c < 127 && c < vocab_size_; ++c) {
    const std::string s(1, static_cast<char>(c));
    token_to_id_[s] = c;
    id_to_token_[static_cast<std::size_t>(c)] = s;
  }
}

std::vector<TokenId> Tokenizer::encode(const std::string& text) const {
  std::vector<TokenId> out;
  out.push_back(bos_id_);
  for (unsigned char ch : text) {
    if (ch < vocab_size_) {
      out.push_back(static_cast<TokenId>(ch));
    } else {
      out.push_back(static_cast<TokenId>(3));  // unk
    }
  }
  return out;
}

std::string Tokenizer::decode_token(TokenId id) const {
  if (id < 0 || id >= vocab_size_) {
    return "";
  }
  if (id == bos_id_ || id == eos_id_ || id == pad_id_) {
    return "";
  }
  return id_to_token_[static_cast<std::size_t>(id)];
}

std::string Tokenizer::decode(const std::vector<TokenId>& tokens) const {
  std::ostringstream oss;
  for (TokenId id : tokens) {
    oss << decode_token(id);
  }
  return oss.str();
}

}  // namespace llm
