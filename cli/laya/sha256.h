// Copyright (c) Meta Platforms, Inc. and affiliates.
// Minimal SHA-256 (FIPS 180-4) for asset verification. On macOS, file()
// uses CommonCrypto, which runs on the ARMv8 SHA-256 instructions and hashes
// the model weights several times faster.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

#ifdef __APPLE__
#    include <CommonCrypto/CommonDigest.h>
#endif

namespace openzl::laya {

class Sha256 {
   public:
    Sha256()
    {
        state_ = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    }

    void update(const uint8_t* data, size_t size)
    {
        total_ += size;
        while (size) {
            size_t take = std::min(size, size_t(64) - fill_);
            std::memcpy(buffer_.data() + fill_, data, take);
            fill_ += take;
            data += take;
            size -= take;
            if (fill_ == 64) {
                transform(buffer_.data());
                fill_ = 0;
            }
        }
    }

    std::string hexdigest()
    {
        const uint64_t bits = total_ * 8;
        buffer_[fill_++]    = 0x80;
        if (fill_ > 56) {
            while (fill_ < 64)
                buffer_[fill_++] = 0;
            transform(buffer_.data());
            fill_ = 0;
        }
        while (fill_ < 56)
            buffer_[fill_++] = 0;
        for (int i = 0; i < 8; ++i)
            buffer_[size_t(56 + i)] = uint8_t(bits >> (56 - 8 * i));
        transform(buffer_.data());
        static const char* hex = "0123456789abcdef";
        std::string out;
        for (uint32_t word : state_)
            for (int i = 28; i >= 0; i -= 4)
                out.push_back(hex[(word >> i) & 0xF]);
        return out;
    }

    static std::string file(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return "";
        std::array<char, 1 << 20> chunk{};
#ifdef __APPLE__
        CC_SHA256_CTX context;
        CC_SHA256_Init(&context);
        while (in.read(chunk.data(), std::streamsize(chunk.size()))
               || in.gcount() > 0)
            CC_SHA256_Update(&context, chunk.data(), CC_LONG(in.gcount()));
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
        CC_SHA256_Final(digest.data(), &context);
        static const char* hex = "0123456789abcdef";
        std::string out;
        for (unsigned char byte : digest) {
            out.push_back(hex[byte >> 4]);
            out.push_back(hex[byte & 0xF]);
        }
        return out;
#else
        Sha256 hash;
        while (in.read(chunk.data(), std::streamsize(chunk.size()))
               || in.gcount() > 0)
            hash.update(
                    reinterpret_cast<const uint8_t*>(chunk.data()),
                    size_t(in.gcount()));
        return hash.hexdigest();
#endif
    }

   private:
    static uint32_t rotr(uint32_t x, int n)
    {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t* block)
    {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
            0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
            0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
            0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
            0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(block[4 * i]) << 24
                    | uint32_t(block[4 * i + 1]) << 16
                    | uint32_t(block[4 * i + 2]) << 8
                    | uint32_t(block[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 =
                    rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 =
                    rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3],
                 e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch  = (e & f) ^ (~e & g);
            uint32_t t1  = h + s1 + ch + k[i] + w[i];
            uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2  = s0 + maj;
            h            = g;
            g            = f;
            f            = e;
            e            = d + t1;
            d            = c;
            c            = b;
            b            = a;
            a            = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> buffer_{};
    size_t fill_    = 0;
    uint64_t total_ = 0;
};

} // namespace openzl::laya
