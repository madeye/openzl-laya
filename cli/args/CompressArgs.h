// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "openzl/cpp/Compressor.hpp"

#include "tools/io/Input.h"
#include "tools/io/Output.h"

#include "cli/args/ArgsUtils.h"
#include "cli/args/GlobalArgs.h"
#include "cli/utils/util.h"
#include "tools/io/InputFile.h"
#include "tools/io/OutputFile.h"

namespace openzl::cli {

struct CompressArgs : public GlobalArgs, public ProfileArgs {
    static void addArgs(arg::ArgParser& parser)
    {
        // Add the command
        parser.addCommand(Cmd::COMPRESS, "compress", 'c');

        // Add the args
        parser.addCommandFlag(
                cmd(),
                "laya",
                0,
                false,
                "Route integer compression using local Laya (experimental).");
        parser.addCommandFlag(
                cmd(),
                "laya-size-guard",
                0,
                false,
                "Compare full output against numeric and retain the smaller frame.");
        for (const auto* flag : { "laya-context",
                                  "laya-confidence",
                                  "laya-timeout-ms",
                                  "laya-report",
                                  "laya-save-compressor" }) {
            parser.addCommandFlag(
                    cmd(),
                    flag,
                    0,
                    true,
                    "Local Laya routing option (see doc/laya.md).");
        }
        parser.addCommandPositional(cmd(), kInput, "Input file path.");
        parser.addCommandFlag(cmd(), kOutput, 'o', true, "Output file path.");
        parser.addCommandFlag(
                cmd(), kForce, 'f', false, "Overwrite output file.");
        parser.addCommandFlag(
                cmd(),
                kCompressor,
                'c',
                true,
                "Compress with the given serialized compressor file.");
        parser.addCommandFlag(
                cmd(),
                kLevel,
                'l',
                true,
                "Compression level (default: 6; higher favors compression "
                "ratio).");
        parser.addCommandFlag(
                cmd(),
                kTrainInline,
                0,
                false,
                "Train the compressor on the input file before compressing.");
        parser.addCommandFlag(
                cmd(),
                kTrainInlineTestLimit,
                0,
                true,
                // "Lime limit spent on inline training, in seconds, for
                // testing."
                "");
        parser.addCommandFlag(
                cmd(),
                kTrace,
                0,
                true,
                "Record a trace of the compression to be visualized with streamdump. Writes a CBOR file to the provided path.");
        parser.addCommandFlag(
                cmd(),
                kTraceStreamsDir,
                0,
                true,
                "Directory to write trace streamdump to.");
        parser.addCommandFlag(
                cmd(),
                kStrict,
                0,
                false,
                "Enforce strict mode compression. Fail on errors instead of falling back to generic compression.");
        parser.addCommandFlag(
                cmd(),
                kNoStreamPreview,
                0,
                false,
                "Omit stream preview data from the trace CBOR output. Requires --trace.");
        parser.addCommandFlag(
                cmd(),
                kStoreOnExpansion,
                0,
                false,
                "Enable anti-inflation guard (replace expanding chunks with STORE). This is the default.");
        parser.addCommandFlag(
                cmd(),
                kNoStoreOnExpansion,
                0,
                false,
                "Disable anti-inflation guard (do not replace expanding chunks with STORE).");
        parser.addCommandFlag(
                cmd(),
                kDictBundle,
                'D',
                true,
                "Path to a fat dict bundle (.zd) file to load for compression.");
    }

    explicit CompressArgs(const arg::ParsedArgs& parsed)
            : GlobalArgs(parsed), ProfileArgs(parsed)
    {
        laya          = parsed.cmdHasFlag(cmd(), "laya");
        layaSizeGuard = parsed.cmdHasFlag(cmd(), "laya-size-guard");
        if (layaSizeGuard && !laya)
            throw InvalidArgsException("--laya-size-guard requires --laya");
        for (const auto* flag : { "laya-context",
                                  "laya-confidence",
                                  "laya-timeout-ms",
                                  "laya-report",
                                  "laya-save-compressor" }) {
            if (!laya && parsed.cmdHasFlag(cmd(), flag)) {
                throw InvalidArgsException(
                        std::string("--") + flag + " requires --laya");
            }
        }
        if (laya) {
#ifndef OPENZL_ENABLE_LAYA
            throw InvalidArgsException(
                    "This build does not enable Laya; configure OPENZL_ENABLE_LAYA on macOS 14+ Apple Silicon or Linux.");
#endif
            if (parsed.cmdHasFlag(cmd(), kCompressor)
                || parsed.cmdHasFlag(cmd(), kTrainInline)
                || parsed.cmdHasFlag(cmd(), kDictBundle) || !map().empty()) {
                throw InvalidArgsException(
                        "--laya requires an integer profile without custom compressors, profile arguments, training, or dictionaries.");
            }
            layaContext = parsed.cmdFlag(cmd(), "laya-context").value_or("");
            if (layaContext.size() > 4096)
                throw InvalidArgsException("--laya-context exceeds 4096 bytes");
            if (auto value = parsed.cmdFlag(cmd(), "laya-confidence")) {
                size_t consumed = 0;
                layaConfidence  = std::stod(*value, &consumed);
                if (consumed != value->size() || !std::isfinite(layaConfidence)
                    || layaConfidence < 0 || layaConfidence > 1)
                    throw InvalidArgsException(
                            "--laya-confidence must be between 0 and 1");
            }
            if (auto value = parsed.cmdFlag(cmd(), "laya-timeout-ms")) {
                layaTimeoutMs = util::checkedstoiExact(*value);
                if (layaTimeoutMs < 1 || layaTimeoutMs > 60000)
                    throw InvalidArgsException(
                            "--laya-timeout-ms must be 1..60000");
            }
            layaReport         = parsed.cmdFlag(cmd(), "laya-report");
            layaSaveCompressor = parsed.cmdFlag(cmd(), "laya-save-compressor");
        }
        // Create the compressor
        auto bundlePath = parsed.cmdFlag(cmd(), kDictBundle);
        if (bundlePath) {
            tools::io::InputFile bundleInput(bundlePath.value());
            dictBundleData = bundleInput.contents();
        }
        setVerbosityLevel(verbosity);
        const auto levelArg = parsed.cmdFlag(cmd(), kLevel);
        if (levelArg) {
            compressionLevel = util::checkedstoiExact(levelArg.value());
            setRequestedCompressionLevel(compressionLevel.value());
        }
        setCompressor(createCompressorFromArgs(
                *this, parsed.cmdFlag(cmd(), kCompressor), dictBundleData));

        // Get the input and output files
        auto inputPath = parsed.cmdPositional(cmd(), kInput);
        auto outputPath =
                parsed.cmdFlag(cmd(), kOutput).value_or(inputPath + ".zl");
        for (const auto& sidecar : { layaReport, layaSaveCompressor }) {
            if (sidecar) {
                auto canonical = [](const std::string& p) {
                    return std::filesystem::weakly_canonical(p);
                };
                auto sameFile = [&](const std::string& a,
                                    const std::string& b) {
                    return canonical(a) == canonical(b)
                            || (std::filesystem::exists(a)
                                && std::filesystem::exists(b)
                                && std::filesystem::equivalent(a, b));
                };
                if (sameFile(*sidecar, inputPath)
                    || sameFile(*sidecar, outputPath)
                    || (layaReport && layaSaveCompressor
                        && sameFile(*layaReport, *layaSaveCompressor)))
                    throw InvalidArgsException(
                            "Laya output paths must be distinct from input and each other");
                checkOutput(*sidecar, parsed.cmdHasFlag(cmd(), kForce));
            }
        }
        checkOutput(outputPath, parsed.cmdHasFlag(cmd(), kForce));
        input  = std::make_unique<tools::io::InputFile>(std::move(inputPath));
        output = std::make_unique<tools::io::OutputFile>(std::move(outputPath));

        trainInline = parsed.cmdHasFlag(cmd(), kTrainInline);
        if (parsed.cmdHasFlag(cmd(), kTrainInlineTestLimit)) {
            trainInlineTestLimit = util::checkedstoul(
                    parsed.cmdFlag(cmd(), kTrainInlineTestLimit).value());
        }

        if (parsed.cmdHasFlag(cmd(), kTrace)) {
            auto path   = parsed.cmdFlag(cmd(), kTrace).value();
            traceOutput = std::make_shared<tools::io::OutputFile>(path);
        }

        traceStreamsDir = parsed.cmdFlag(cmd(), kTraceStreamsDir);
        strict          = parsed.cmdHasFlag(cmd(), kStrict);
        streamPreview   = !parsed.cmdHasFlag(cmd(), kNoStreamPreview);

        if (!streamPreview && !traceOutput) {
            throw InvalidArgsException(
                    "--no-stream-preview requires --trace to be specified.");
        }
        if (parsed.cmdHasFlag(cmd(), kNoStoreOnExpansion)) {
            storeOnExpansion = false;
        } else if (parsed.cmdHasFlag(cmd(), kStoreOnExpansion)) {
            storeOnExpansion = true;
        }
    }

    static Cmd cmd()
    {
        return Cmd::COMPRESS;
    }

    std::shared_ptr<tools::io::Input> input;
    std::shared_ptr<tools::io::Output> output;

    bool laya          = false;
    bool layaSizeGuard = false;
    std::string layaContext;
    double layaConfidence = 0.8;
    int layaTimeoutMs     = 2000;
    std::optional<std::string> layaReport;
    std::optional<std::string> layaSaveCompressor;
    bool trainInline{};
    poly::optional<size_t> trainInlineTestLimit;

    std::shared_ptr<tools::io::Output> traceOutput;
    std::optional<std::string> traceStreamsDir;
    bool strict           = false;
    bool streamPreview    = true;
    bool storeOnExpansion = true;
    std::optional<int> compressionLevel;
    std::string dictBundleData;

   private:
    inline static const std::string kInput      = "input";
    inline static const std::string kOutput     = "output";
    inline static const std::string kForce      = "force";
    inline static const std::string kCompressor = "compressor";
    inline static const std::string kLevel      = "level";

    inline static const std::string kVerbose     = "verbose";
    inline static const std::string kRecursive   = "recursive";
    inline static const std::string kTrainInline = "train-inline";
    inline static const std::string kTrainInlineTestLimit =
            "train-inline-test-limit";
    inline static const std::string kTrace            = "trace";
    inline static const std::string kTraceStreamsDir  = "trace-streams-dir";
    inline static const std::string kStrict           = "strict";
    inline static const std::string kNoStreamPreview  = "no-stream-preview";
    inline static const std::string kStoreOnExpansion = "store-on-expansion";
    inline static const std::string kNoStoreOnExpansion =
            "no-store-on-expansion";
    inline static const std::string kDictBundle = "dict-bundle";
};

} // namespace openzl::cli
