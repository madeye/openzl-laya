// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "cli/args/CompressArgs.h"
#include "tools/json.hpp"

namespace openzl::cli::laya {
using Json = nlohmann::json;
#ifdef __APPLE__
// FluidInference/laya-coreml (Core ML conversion used by the macOS worker).
inline constexpr const char* revision =
        "7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc";
inline constexpr const char* defaultComputeUnits = "cpu_and_ne";
inline constexpr const char* defaultPrecision    = "e8";
#else
// convaiinnovations/laya-multilingual (PyTorch checkpoint used by the Linux
// worker).
inline constexpr const char* revision =
        "052592a15d198d9ad47da779604259b10b47b7aa";
inline constexpr const char* defaultComputeUnits = "cuda";
inline constexpr const char* defaultPrecision    = "bfloat16";
#endif
inline const std::vector<std::string> candidates = {
    "numeric",       "fieldlz",  "range_fieldlz", "range_zstd",
    "delta_fieldlz", "tokenize", "zstd"
};
struct IntegerType {
    size_t width;
    bool isSigned;
    bool bigEndian;
};
IntegerType integerType(const std::string& profile);
std::vector<poly::string_view> samples(poly::string_view data, size_t width);
std::vector<poly::string_view> probeWindows(
        poly::string_view data,
        size_t width,
        size_t limit,
        size_t chunkBytes);
Json statistics(
        const std::vector<poly::string_view>& windows,
        IntegerType type);
std::vector<size_t>
validateResponse(const Json& response, const std::string& id, double threshold);
std::shared_ptr<Compressor> candidate(const CompressArgs& args, size_t index);
Json route(CompressArgs& args);
void writeReport(
        const CompressArgs& args,
        Json report,
        size_t size,
        double compressionMs);
void saveCompressor(const CompressArgs& args);
} // namespace openzl::cli::laya
