// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "cli/laya/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <queue>
#include <stdexcept>

#include "tools/json.hpp"

namespace openzl::laya {
namespace {
constexpr char32_t kSpaceMarker = 0x2581;

bool isWhitespace(char32_t c)
{
    // Rust char::is_whitespace: the Unicode White_Space property.
    return c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0
            || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028
            || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

uint64_t pairKey(int left, int right)
{
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}
} // namespace

std::u32string decodeUtf8(const std::string& text)
{
    std::u32string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        auto b0 = static_cast<unsigned char>(text[i]);
        char32_t c;
        size_t n;
        if (b0 < 0x80) {
            c = b0;
            n = 1;
        } else if ((b0 & 0xE0) == 0xC0) {
            c = b0 & 0x1F;
            n = 2;
        } else if ((b0 & 0xF0) == 0xE0) {
            c = b0 & 0x0F;
            n = 3;
        } else if ((b0 & 0xF8) == 0xF0) {
            c = b0 & 0x07;
            n = 4;
        } else {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        if (i + n > text.size()) {
            out.push_back(0xFFFD);
            break;
        }
        bool valid = true;
        for (size_t j = 1; j < n; ++j) {
            auto b = static_cast<unsigned char>(text[i + j]);
            if ((b & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            c = (c << 6) | (b & 0x3F);
        }
        if (!valid) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        out.push_back(c);
        i += n;
    }
    return out;
}

std::string encodeUtf8(char32_t c)
{
    std::string s;
    if (c < 0x80) {
        s.push_back(char(c));
    } else if (c < 0x800) {
        s.push_back(char(0xC0 | (c >> 6)));
        s.push_back(char(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        s.push_back(char(0xE0 | (c >> 12)));
        s.push_back(char(0x80 | ((c >> 6) & 0x3F)));
        s.push_back(char(0x80 | (c & 0x3F)));
    } else {
        s.push_back(char(0xF0 | (c >> 18)));
        s.push_back(char(0x80 | ((c >> 12) & 0x3F)));
        s.push_back(char(0x80 | ((c >> 6) & 0x3F)));
        s.push_back(char(0x80 | (c & 0x3F)));
    }
    return s;
}

Tokenizer::Tokenizer(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open " + path);
    nlohmann::json root = nlohmann::json::parse(file);
    const auto& model   = root.at("model");
    if (model.at("type") != "BPE" || model.at("byte_fallback") != true)
        throw std::runtime_error(
                "tokenizer.json must describe a byte-fallback BPE model");
    const auto& vocab = model.at("vocab");
    vocab_.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it)
        vocab_.emplace(it.key(), it.value().get<int>());
    const auto& merges = model.at("merges");
    merges_.reserve(merges.size());
    int rank = 0;
    for (const auto& entry : merges) {
        std::string left, right;
        if (entry.is_array()) {
            left  = entry.at(0).get<std::string>();
            right = entry.at(1).get<std::string>();
        } else {
            auto text  = entry.get<std::string>();
            auto space = text.find(' ');
            if (space == std::string::npos)
                throw std::runtime_error("merge entry without a separator");
            left  = text.substr(0, space);
            right = text.substr(space + 1);
        }
        int a = lookup(left), b = lookup(right), ab = lookup(left + right);
        if (a >= 0 && b >= 0 && ab >= 0)
            merges_.emplace(pairKey(a, b), MergeInfo{ rank, ab });
        ++rank;
    }
    byteTokens_.assign(256, -1);
    for (int value = 0; value < 256; ++value) {
        char name[8];
        snprintf(name, sizeof(name), "<0x%02X>", value);
        byteTokens_[size_t(value)] = lookup(name);
    }
    if (root.contains("added_tokens")) {
        for (const auto& entry : root.at("added_tokens")) {
            auto content = entry.at("content").get<std::string>();
            if (content.empty())
                continue;
            AddedToken token{ decodeUtf8(content),
                              entry.at("id").get<int>(),
                              entry.value("lstrip", false),
                              entry.value("rstrip", false) };
            vocab_.emplace(content, token.id);
            addedByFirst_[token.scalars[0]].push_back(std::move(token));
        }
        // Longest content first so overlapping added tokens resolve like the
        // Rust matcher.
        for (auto& [first, tokens] : addedByFirst_)
            std::stable_sort(
                    tokens.begin(), tokens.end(), [](auto& a, auto& b) {
                        return a.scalars.size() > b.scalars.size();
                    });
    }
    maskId_ = requiredToken("<mask>");
    clsId_  = requiredToken("<bos>");
    sepId_  = requiredToken("<eos>");
    padId_  = requiredToken("<pad>");
    unkId_  = requiredToken("<unk>");
}

int Tokenizer::lookup(const std::string& token) const
{
    auto it = vocab_.find(token);
    return it == vocab_.end() ? -1 : it->second;
}

int Tokenizer::requiredToken(const std::string& token) const
{
    int id = lookup(token);
    if (id < 0)
        throw std::runtime_error("tokenizer.json has no " + token + " token");
    return id;
}

std::vector<int> Tokenizer::encode(const std::string& text) const
{
    std::vector<int> ids;
    if (text.empty())
        return ids;
    const auto scalars = decodeUtf8(text);
    std::u32string segment;
    size_t index = 0;
    while (index < scalars.size()) {
        auto candidates         = addedByFirst_.find(scalars[index]);
        const AddedToken* match = nullptr;
        if (candidates != addedByFirst_.end()) {
            for (const auto& token : candidates->second) {
                if (index + token.scalars.size() <= scalars.size()
                    && scalars.compare(
                               index, token.scalars.size(), token.scalars)
                            == 0) {
                    match = &token;
                    break;
                }
            }
        }
        if (match) {
            if (match->lstrip)
                while (!segment.empty() && isWhitespace(segment.back()))
                    segment.pop_back();
            encodeSegment(segment, ids);
            segment.clear();
            ids.push_back(match->id);
            index += match->scalars.size();
            if (match->rstrip)
                while (index < scalars.size() && isWhitespace(scalars[index]))
                    ++index;
            continue;
        }
        segment.push_back(scalars[index]);
        ++index;
    }
    encodeSegment(segment, ids);
    return ids;
}

void Tokenizer::encodeSegment(
        const std::u32string& segment,
        std::vector<int>& ids) const
{
    if (segment.empty())
        return;
    // Replace normalizer, then Metaspace with prepend_scheme "always" and
    // split on the marker (each piece keeps its leading marker).
    std::u32string scalars;
    scalars.reserve(segment.size() + 1);
    for (char32_t c : segment)
        scalars.push_back(c == U' ' ? kSpaceMarker : c);
    if (scalars.front() != kSpaceMarker)
        scalars.insert(scalars.begin(), kSpaceMarker);
    std::u32string piece;
    for (char32_t c : scalars) {
        if (c == kSpaceMarker && !piece.empty()) {
            auto encoded = encodePiece(piece);
            ids.insert(ids.end(), encoded.begin(), encoded.end());
            piece.clear();
        }
        piece.push_back(c);
    }
    if (!piece.empty()) {
        auto encoded = encodePiece(piece);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
    }
}

std::vector<int> Tokenizer::encodePiece(const std::u32string& piece) const
{
    {
        std::lock_guard<std::mutex> guard(cacheMutex_);
        auto it = cache_.find(piece);
        if (it != cache_.end())
            return it->second;
    }
    auto ids = bpe(piece);
    std::lock_guard<std::mutex> guard(cacheMutex_);
    if (cache_.size() >= 8192)
        cache_.clear();
    cache_.emplace(piece, ids);
    return ids;
}

std::vector<int> Tokenizer::bpe(const std::u32string& piece) const
{
    // Initial symbols: vocabulary characters, byte-fallback tokens, or fused
    // unknowns, exactly as tokenizers' BPE::merge_word.
    std::vector<int> symbol;
    symbol.reserve(piece.size());
    bool unknownRun = false;
    for (char32_t c : piece) {
        const auto utf8 = encodeUtf8(c);
        int id          = lookup(utf8);
        if (id >= 0) {
            symbol.push_back(id);
            unknownRun = false;
            continue;
        }
        bool bytesKnown = true;
        for (unsigned char b : utf8)
            bytesKnown = bytesKnown && byteTokens_[b] >= 0;
        if (bytesKnown) {
            for (unsigned char b : utf8)
                symbol.push_back(byteTokens_[b]);
            unknownRun = false;
        } else if (!unknownRun) {
            symbol.push_back(unkId_);
            unknownRun = true;
        }
    }
    const size_t n = symbol.size();
    if (n < 2)
        return symbol;
    std::vector<int> prev(n), next(n);
    std::vector<bool> alive(n, true);
    for (size_t i = 0; i < n; ++i) {
        prev[i] = int(i) - 1;
        next[i] = i + 1 < n ? int(i) + 1 : -1;
    }
    struct Candidate {
        int rank, pos, left, right;
        bool operator>(const Candidate& o) const
        {
            return rank != o.rank ? rank > o.rank : pos > o.pos;
        }
    };
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> heap;
    auto propose = [&](int pos) {
        if (pos < 0 || next[size_t(pos)] < 0)
            return;
        int right = next[size_t(pos)];
        auto it   = merges_.find(
                pairKey(symbol[size_t(pos)], symbol[size_t(right)]));
        if (it != merges_.end())
            heap.push(
                    { it->second.rank,
                      pos,
                      symbol[size_t(pos)],
                      symbol[size_t(right)] });
    };
    for (size_t i = 0; i + 1 < n; ++i)
        propose(int(i));
    while (!heap.empty()) {
        auto top = heap.top();
        heap.pop();
        const auto pos = size_t(top.pos);
        if (!alive[pos] || next[pos] < 0)
            continue;
        const auto right = size_t(next[pos]);
        if (symbol[pos] != top.left || symbol[right] != top.right)
            continue; // stale after another merge
        symbol[pos]  = merges_.at(pairKey(top.left, top.right)).merged;
        alive[right] = false;
        next[pos]    = next[right];
        if (next[pos] >= 0)
            prev[size_t(next[pos])] = int(pos);
        propose(prev[pos]);
        propose(int(pos));
    }
    std::vector<int> ids;
    for (int i = 0; i >= 0; i = next[size_t(i)])
        ids.push_back(symbol[size_t(i)]);
    return ids;
}

} // namespace openzl::laya
