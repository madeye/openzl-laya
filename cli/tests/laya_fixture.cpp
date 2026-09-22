// Copyright (c) Meta Platforms, Inc. and affiliates.
// Test-only graph export, independent of the model and production routing
// policy.
#include <filesystem>
#include <fstream>
#include "cli/utils/laya.h"
#include "tools/arg/arg_parser.h"

int main(int argc, char** argv)
{
    using namespace openzl::cli;
    if (argc != 2)
        return 1;
    const std::filesystem::path output(argv[1]);
    std::filesystem::create_directories(output);
    openzl::arg::ArgParser parser;
    GlobalArgs::addArgs(parser);
    ProfileArgs::addArgs(parser);
    CompressArgs::addArgs(parser);
    std::vector<std::string> strings = { "zli",       "compress", "/dev/null",
                                         "--profile", "le-u64",   "-o",
                                         "/dev/null", "--force" };
    std::vector<char*> args;
    for (auto& s : strings)
        args.push_back(s.data());
    CompressArgs options(parser.parse(int(args.size()), args.data()));
    for (size_t i = 0; i < laya::candidates.size(); ++i) {
        const auto serialized = laya::candidate(options, i)->serialize();
        std::ofstream file(
                output / (laya::candidates[i] + ".compressor"),
                std::ios::binary);
        file.write(serialized.data(), std::streamsize(serialized.size()));
        if (!file)
            return 1;
    }
}
