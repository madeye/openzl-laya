// Copyright (c) Meta Platforms, Inc. and affiliates.
// Byte-fallback BPE tokenizer matching HuggingFace `tokenizers` for the
// mmBERT/Gemma tokenizer.json shipped with the Laya checkpoint: a Replace
// normalizer (" " -> U+2581), a Metaspace pre-tokenizer (prepend always,
// split on U+2581), added-token matching with lstrip/rstrip, and BPE merges
// with byte fallback and fused unknowns.
#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace openzl::laya {

class Tokenizer {
   public:
    explicit Tokenizer(const std::string& tokenizerJsonPath);

    /// Token ids of `text` without special tokens (add_special_tokens=False).
    std::vector<int> encode(const std::string& text) const;

    int maskId() const
    {
        return maskId_;
    }
    int clsId() const
    {
        return clsId_;
    }
    int sepId() const
    {
        return sepId_;
    }
    int padId() const
    {
        return padId_;
    }
    int unkId() const
    {
        return unkId_;
    }
    size_t vocabSize() const
    {
        return vocab_.size();
    }

   private:
    struct AddedToken {
        std::u32string scalars;
        int id;
        bool lstrip;
        bool rstrip;
    };
    struct MergeInfo {
        int rank;
        int merged;
    };

    void encodeSegment(const std::u32string& segment, std::vector<int>& ids)
            const;
    std::vector<int> encodePiece(const std::u32string& piece) const;
    std::vector<int> bpe(const std::u32string& piece) const;
    int lookup(const std::string& token) const;
    int requiredToken(const std::string& token) const;

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<uint64_t, MergeInfo> merges_;
    std::vector<int> byteTokens_; // 256 entries, -1 when missing
    std::unordered_map<char32_t, std::vector<AddedToken>> addedByFirst_;
    int maskId_, clsId_, sepId_, padId_, unkId_;

    // Piece cache: statistics prompts repeat many pieces.
    mutable std::mutex cacheMutex_;
    mutable std::unordered_map<std::u32string, std::vector<int>> cache_;
};

std::u32string decodeUtf8(const std::string& text);
std::string encodeUtf8(char32_t scalar);

} // namespace openzl::laya
