// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "cli/utils/laya.h"
#include <gtest/gtest.h>

using namespace openzl::cli::laya;

TEST(Laya, Samples)
{
    std::string data(1024 * 1024, '\0');
    auto windows = samples(data, 8);
    ASSERT_EQ(windows.size(), 3);
    EXPECT_EQ(windows.front().data(), data.data());
    EXPECT_EQ(
            windows.back().data() + windows.back().size(),
            data.data() + data.size());
    for (size_t i = 0; i < windows.size(); ++i) {
        EXPECT_EQ(size_t(windows[i].data() - data.data()) % 8, 0);
        EXPECT_EQ(windows[i].size(), 65536);
        if (i)
            EXPECT_GE(
                    windows[i].data(),
                    windows[i - 1].data() + windows[i - 1].size());
    }
    EXPECT_EQ(samples(std::string(100000, 'a'), 8).size(), 1);
    EXPECT_THROW(samples("123", 2), std::exception);
}
TEST(Laya, ChunkAlignedProbes)
{
    std::string data(5 * 65536 + 8, '\0');
    auto windows = probeWindows(data, 8, 1024, 65536);
    ASSERT_EQ(windows.size(), 3);
    EXPECT_EQ(windows[0].data(), data.data());
    EXPECT_EQ(windows[1].data(), data.data() + 2 * 65536);
    EXPECT_EQ(windows[2].data(), data.data() + 5 * 65536);
    EXPECT_EQ(windows[2].size(), 8);
    EXPECT_EQ(probeWindows(data, 8, 1024, data.size()).size(), 1);
    EXPECT_TRUE(probeWindows("", 8, 1024, 65536).empty());
    EXPECT_THROW(probeWindows(data, 8, 0, 65536), std::exception);
    EXPECT_THROW(probeWindows(data, 8, 1024, 1), std::exception);
}
TEST(Laya, IntegerTypes)
{
    EXPECT_EQ(integerType("le-u64").width, 8);
    EXPECT_TRUE(integerType("be-i32").isSigned);
    EXPECT_TRUE(integerType("be-i32").bigEndian);
    EXPECT_EQ(integerType("i8").width, 1);
    EXPECT_THROW(integerType("serial"), std::exception);
}
TEST(Laya, SignedExtremesAndEndian)
{
    for (bool big : { false, true }) {
        std::string data(24, '\0');
        std::array<uint64_t, 3> values{ uint64_t(1) << 63,
                                        0,
                                        (uint64_t(1) << 63) - 1 };
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 8; ++j)
                data[i * 8 + j] = char(values[i] >> (8 * (big ? 7 - j : j)));
        auto stats = statistics(samples(data, 8), { 8, true, big });
        EXPECT_DOUBLE_EQ(stats["range"].get<double>(), std::ldexp(1., 64));
        EXPECT_DOUBLE_EQ(stats["monotonicity"].get<double>(), 1.);
        EXPECT_DOUBLE_EQ(stats["zero_fraction"].get<double>(), 1. / 3);
        EXPECT_DOUBLE_EQ(stats["cardinality_ratio"].get<double>(), 1.);
    }
}
TEST(Laya, Responses)
{
    Json r{ { "version", 1 },
            { "id", "test" },
            { "revision", revision },
            { "truncated", false },
            { "candidates", candidates },
            { "probabilities", { 1., 0., 0., 0., 0., 0., 0. } },
            { "confidence", 1. },
            { "action_probability", 0. },
            { "selected", "numeric" } };
    EXPECT_EQ(
            validateResponse(r, "test", .8),
            (std::vector<size_t>{ 0, 1, 2, 3, 4, 5, 6 }));
    r["probabilities"] = std::vector<double>(7, 1. / 7);
    r["confidence"]    = 0.;
    EXPECT_EQ(
            validateResponse(r, "test", .8),
            (std::vector<size_t>{ 0, 1, 2, 3, 4, 5, 6 }));
    r["truncated"] = true;
    EXPECT_THROW(validateResponse(r, "test", .8), std::exception);
    r["truncated"]        = false;
    r["probabilities"][0] = -1;
    EXPECT_THROW(validateResponse(r, "test", .8), std::exception);
    EXPECT_THROW(validateResponse(r, "wrong", .8), std::exception);
}

#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "tools/arg/arg_parser.h"

TEST(Laya, CandidateRoundTripsAndSerialization)
{
    using namespace openzl;
    using namespace openzl::cli;
    for (auto profile : { "u8",
                          "i8",
                          "le-u16",
                          "le-i16",
                          "be-u16",
                          "be-i16",
                          "le-u32",
                          "le-i32",
                          "be-u32",
                          "be-i32",
                          "le-u64",
                          "le-i64",
                          "be-u64",
                          "be-i64" }) {
        for (auto chunk : { "32768", "65536" }) {
            arg::ArgParser parser;
            GlobalArgs::addArgs(parser);
            ProfileArgs::addArgs(parser);
            CompressArgs::addArgs(parser);
            std::vector<std::string> strings{
                "zli",          "compress", "/dev/null", "--profile", profile,
                "--chunk-size", chunk,      "-o",        "/dev/null", "--force"
            };
            strings.insert(
                    strings.end(),
                    { "--level", std::string(chunk) == "32768" ? "1" : "9" });
            std::vector<char*> argv;
            for (auto& s : strings)
                argv.push_back(s.data());
            CompressArgs args(parser.parse(int(argv.size()), argv.data()));
            std::string data(131072, '\0');
            auto type = integerType(profile);
            for (size_t offset = 0; offset < data.size();
                 offset += type.width) {
                uint64_t value = (offset / type.width) % 257;
                if ((offset / type.width) % 31 == 0)
                    value = UINT64_MAX;
                for (size_t j = 0; j < type.width; ++j)
                    data[offset + j] = char(
                            value
                            >> (8 * (type.bigEndian ? type.width - 1 - j : j)));
            }
            for (size_t i = 0; i < candidates.size(); ++i) {
                SCOPED_TRACE(
                        std::string(profile) + "/" + chunk + "/"
                        + candidates[i]);
                auto comp = candidate(args, i);
                Compressor restored;
                restored.deserialize(comp->serialize());
                CCtx cctx;
                cctx.refCompressor(restored);
                cctx.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
                cctx.setParameter(CParam::PermissiveCompression, 0);
                DCtx dctx;
                EXPECT_EQ(
                        dctx.decompressSerial(cctx.compressSerial(data)), data);
            }
        }
    }
}
TEST(Laya, NearbyUnsignedExtremes)
{
    std::string data(24, '\0');
    for (size_t i = 0; i < 3; ++i) {
        auto value = UINT64_MAX - i;
        for (size_t j = 0; j < 8; ++j)
            data[i * 8 + j] = char(value >> (8 * j));
    }
    auto stats = statistics(samples(data, 8), { 8, false, false });
    EXPECT_DOUBLE_EQ(stats["range"].get<double>(), 2.);
    EXPECT_DOUBLE_EQ(stats["delta_p50"].get<double>(), 1.);
    EXPECT_EQ(stats["max_run"], 1);
}
