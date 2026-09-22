// Copyright (c) Meta Platforms, Inc. and affiliates.
// Minimal safetensors reader: 8-byte little-endian header length, JSON
// header, then raw tensor bytes.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/json.hpp"

namespace openzl::laya {

struct TensorInfo {
    std::string dtype;
    std::vector<size_t> shape;
    size_t begin = 0, end = 0;
    size_t elements() const
    {
        size_t n = 1;
        for (auto d : shape)
            n *= d;
        return n;
    }
};

class SafeTensors {
   public:
    explicit SafeTensors(const std::string& path)
            : file_(path, std::ios::binary)
    {
        if (!file_)
            throw std::runtime_error("cannot open " + path);
        uint64_t headerSize = 0;
        file_.read(reinterpret_cast<char*>(&headerSize), 8);
        std::string header(headerSize, '\0');
        file_.read(header.data(), std::streamsize(headerSize));
        if (!file_)
            throw std::runtime_error("truncated safetensors header");
        dataOffset_ = 8 + headerSize;
        auto json   = nlohmann::json::parse(header);
        for (auto it = json.begin(); it != json.end(); ++it) {
            if (it.key() == "__metadata__")
                continue;
            TensorInfo info;
            info.dtype = it.value().at("dtype").get<std::string>();
            info.shape = it.value().at("shape").get<std::vector<size_t>>();
            info.begin = it.value().at("data_offsets").at(0).get<size_t>();
            info.end   = it.value().at("data_offsets").at(1).get<size_t>();
            tensors_.emplace(it.key(), info);
        }
    }

    const TensorInfo& info(const std::string& name) const
    {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("missing tensor " + name);
        return it->second;
    }

    bool has(const std::string& name) const
    {
        return tensors_.count(name) != 0;
    }

    /// Raw bytes of a tensor.
    std::vector<uint8_t> bytes(const std::string& name)
    {
        const auto& t = info(name);
        std::vector<uint8_t> out(t.end - t.begin);
        file_.seekg(std::streamoff(dataOffset_ + t.begin));
        file_.read(
                reinterpret_cast<char*>(out.data()),
                std::streamsize(out.size()));
        if (!file_)
            throw std::runtime_error("truncated tensor " + name);
        return out;
    }

    /// Tensor converted to fp32 (supports F16, BF16, F32).
    std::vector<float> floats(const std::string& name)
    {
        const auto& t = info(name);
        auto raw      = bytes(name);
        std::vector<float> out(t.elements());
        if (t.dtype == "F32") {
            std::memcpy(out.data(), raw.data(), raw.size());
        } else if (t.dtype == "F16") {
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = halfToFloat(
                        uint16_t(raw[2 * i]) | uint16_t(raw[2 * i + 1]) << 8);
        } else if (t.dtype == "BF16") {
            for (size_t i = 0; i < out.size(); ++i) {
                uint32_t bits = uint32_t(
                                        uint16_t(raw[2 * i])
                                        | uint16_t(raw[2 * i + 1]) << 8)
                        << 16;
                std::memcpy(&out[i], &bits, 4);
            }
        } else {
            throw std::runtime_error(
                    "unsupported dtype " + t.dtype + " for " + name);
        }
        return out;
    }

    /// Tensor as IEEE half bits (supports F16 directly, F32/BF16 by rounding).
    std::vector<uint16_t> halves(const std::string& name)
    {
        const auto& t = info(name);
        if (t.dtype == "F16") {
            auto raw = bytes(name);
            std::vector<uint16_t> out(t.elements());
            std::memcpy(out.data(), raw.data(), raw.size());
            return out;
        }
        auto f = floats(name);
        std::vector<uint16_t> out(f.size());
        for (size_t i = 0; i < f.size(); ++i)
            out[i] = floatToHalf(f[i]);
        return out;
    }

    static float halfToFloat(uint16_t h)
    {
        uint32_t sign = uint32_t(h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t man  = h & 0x3FF;
        uint32_t bits;
        if (exp == 0) {
            if (man == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while (!(man & 0x400)) {
                    man <<= 1;
                    --exp;
                }
                man &= 0x3FF;
                bits = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000 | (man << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }

    static uint16_t floatToHalf(float f)
    {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp   = int32_t((x >> 23) & 0xFF) - 127 + 15;
        uint32_t man  = x & 0x7FFFFF;
        if (((x >> 23) & 0xFF) == 0xFF)
            return uint16_t(sign | 0x7C00 | (man ? 0x200 : 0));
        if (exp >= 31)
            return uint16_t(sign | 0x7C00);
        if (exp <= 0) {
            if (exp < -10)
                return uint16_t(sign);
            man |= 0x800000;
            uint32_t shift = uint32_t(14 - exp);
            uint32_t half  = man >> shift;
            uint32_t rem   = man & ((1u << shift) - 1);
            uint32_t mid   = 1u << (shift - 1);
            if (rem > mid || (rem == mid && (half & 1)))
                ++half;
            return uint16_t(sign | half);
        }
        uint32_t half = uint32_t(exp << 10) | (man >> 13);
        uint32_t rem  = man & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (half & 1)))
            ++half;
        return uint16_t(sign | half);
    }

   private:
    std::ifstream file_;
    size_t dataOffset_ = 0;
    std::map<std::string, TensorInfo> tensors_;
};

} // namespace openzl::laya
