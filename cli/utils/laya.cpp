// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "cli/utils/laya.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <set>
#include <thread>

#include "cli/utils/profile_graphs.h"
#include "openzl/codecs/zl_conversion.h"
#include "openzl/codecs/zl_delta.h"
#include "openzl/codecs/zl_generic.h"
#include "openzl/codecs/zl_lz.h"
#include "openzl/codecs/zl_range_pack.h"
#include "openzl/codecs/zl_segmenters.h"
#include "openzl/codecs/zl_tokenize.h"
#include "openzl/codecs/zl_zstd.h"
#include "openzl/cpp/CCtx.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "tools/logger/Logger.h"

#ifdef OPENZL_ENABLE_LAYA
#    include <arpa/inet.h>
#    include <fcntl.h>
#    include <poll.h>
#    include <spawn.h>
#    include <sys/socket.h>
#    include <sys/stat.h>
#    include <sys/un.h>
#    include <sys/wait.h>
#    include <unistd.h>
#    ifdef __APPLE__
#        include <crt_externs.h>
#        include <mach-o/dyld.h>
#    else
extern char** environ;
#    endif
#endif

namespace openzl::cli::laya {
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count();
}
IntegerType integerType(const std::string& profile)
{
    std::smatch match;
    if (!std::regex_match(
                profile,
                match,
                std::regex("(?:(le|be)-([iu])(16|32|64)|([iu])(8))"))) {
        throw InvalidArgsException(
                "--laya supports only integer profiles (u8, i8, le/be-u/i16/32/64)");
    }
    return { match[3].matched ? size_t(std::stoul(match[3]) / 8) : 1,
             (match[2].matched ? match[2].str() : match[4].str()) == "i",
             match[1] == "be" };
}
std::vector<poly::string_view> samples(poly::string_view data, size_t width)
{
    if (width == 0 || data.size() % width)
        throw InvalidArgsException("Integer input must be element-aligned");
    if (data.empty())
        return {};
    size_t length = std::min(size_t(65536), data.size());
    length -= length % width;
    std::vector<poly::string_view> result;
    for (size_t offset : { size_t(0),
                           ((data.size() - length) / 2 / width) * width,
                           data.size() - length }) {
        if (result.empty()
            || data.data() + offset
                    >= result.back().data() + result.back().size())
            result.emplace_back(data.data() + offset, length);
    }
    return result;
}
// Probe actual chunk starts, with a bounded prefix at each scale. A short
// final chunk is retained; never concatenate unrelated regions into one frame.
std::vector<poly::string_view> probeWindows(
        poly::string_view data,
        size_t width,
        size_t limit,
        size_t chunkBytes)
{
    if (!width || !limit || !chunkBytes || data.size() % width)
        throw InvalidArgsException("Invalid integer probe geometry");
    chunkBytes -= chunkBytes % width;
    limit -= limit % width;
    if (!chunkBytes || !limit)
        throw InvalidArgsException("Probe must contain at least one element");
    if (data.empty())
        return {};
    const size_t chunks = 1 + (data.size() - 1) / chunkBytes;
    std::vector<poly::string_view> result;
    for (size_t index : { size_t(0), (chunks - 1) / 2, chunks - 1 }) {
        const size_t offset = index * chunkBytes;
        if (!result.empty() && result.back().data() == data.data() + offset)
            continue;
        result.emplace_back(
                data.data() + offset,
                std::min({ limit, chunkBytes, data.size() - offset }));
    }
    return result;
}
Json statistics(const std::vector<poly::string_view>& windows, IntegerType type)
{
    std::set<uint64_t> distinct;
    std::array<size_t, 256> bytes{};
    std::vector<long double> deltas;
    size_t count = 0, zeros = 0, increasing = 0, decreasing = 0, pairs = 0,
           runs = 0, maxRun = 0, byteCount = 0;
    uint64_t minOrdered = UINT64_MAX, maxOrdered = 0;
    long double minimum = std::numeric_limits<long double>::infinity(),
                maximum = -minimum;
    for (auto window : windows) {
        uint64_t previous = 0;
        size_t run        = 0;
        for (size_t i = 0; i < window.size(); i += type.width) {
            uint64_t bits = 0;
            for (size_t j = 0; j < type.width; ++j) {
                auto byte = static_cast<unsigned char>(window[i + j]);
                ++bytes[byte];
                ++byteCount;
                bits |= uint64_t(byte)
                        << (8 * (type.bigEndian ? type.width - 1 - j : j));
            }
            distinct.insert(bits);
            // Avoid signed conversion and subtraction overflow, including
            // INT64_MIN.
            long double value = static_cast<long double>(bits);
            if (type.isSigned && (bits & (uint64_t(1) << (8 * type.width - 1))))
                value -= std::ldexp(1.0L, int(8 * type.width));
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            ++count;
            zeros += bits == 0;
            uint64_t ordered = bits
                    ^ (type.isSigned ? (uint64_t(1) << (8 * type.width - 1))
                                     : 0);
            minOrdered = std::min(minOrdered, ordered);
            maxOrdered = std::max(maxOrdered, ordered);
            if (i) {
                ++pairs;
                increasing += ordered >= previous;
                decreasing += ordered <= previous;
                deltas.push_back(
                        static_cast<long double>(
                                ordered >= previous ? ordered - previous
                                                    : previous - ordered));
            }
            if (!i || ordered != previous) {
                ++runs;
                run = 0;
            }
            maxRun   = std::max(maxRun, ++run);
            previous = ordered;
        }
    }
    std::sort(deltas.begin(), deltas.end());
    auto nonzero = std::upper_bound(deltas.begin(), deltas.end(), 0.0L);
    const auto nonzeroCount = size_t(deltas.end() - nonzero);
    auto quantile           = [&](double q) {
        return deltas.empty()
                          ? 0.0
                          : double(deltas[size_t(q * double(deltas.size() - 1))]);
    };
    double entropy = 0;
    for (auto n : bytes)
        if (n) {
            double p = double(n) / double(byteCount);
            entropy -= p * std::log2(p);
        }
    return { { "elements", count },
             { "width", type.width },
             { "signed", type.isSigned },
             { "big_endian", type.bigEndian },
             { "cardinality_ratio",
               count ? double(distinct.size()) / double(count) : 0 },
             { "monotonicity",
               pairs ? double(std::max(increasing, decreasing)) / double(pairs)
                     : 1 },
             { "delta_p50", quantile(0.5) },
             { "delta_p90", quantile(0.9) },
             { "delta_p99", quantile(0.99) },
             { "nonzero_delta_fraction",
               pairs ? double(nonzeroCount) / pairs : 0 },
             { "nonzero_delta_p50",
               nonzeroCount ? double(nonzero[nonzeroCount / 2]) : 0 },
             { "mean_run", runs ? double(count) / double(runs) : 0 },
             { "max_run", maxRun },
             { "zero_fraction", count ? double(zeros) / double(count) : 0 },
             { "min", count ? double(minimum) : 0 },
             { "max", count ? double(maximum) : 0 },
             { "range", count ? double(maxOrdered - minOrdered) : 0 },
             { "byte_entropy", entropy } };
}
std::vector<size_t>
validateResponse(const Json& response, const std::string& id, double threshold)
{
    if (response.at("version") != 1 || response.at("id") != id
        || response.at("revision") != revision)
        throw std::runtime_error("worker protocol mismatch");
    if (response.contains("error"))
        throw std::runtime_error(response.at("error").get<std::string>());
    if (response.at("truncated").get<bool>())
        throw std::runtime_error("truncated state");
    auto probabilities =
            response.at("probabilities").get<std::vector<double>>();
    if (response.at("candidates") != candidates
        || probabilities.size() != candidates.size())
        throw std::runtime_error("invalid candidates");
    double sum = 0;
    for (double p : probabilities) {
        if (!std::isfinite(p) || p < 0 || p > 1)
            throw std::runtime_error("invalid probability");
        sum += p;
    }
    if (std::abs(sum - 1) > 0.0001)
        throw std::runtime_error("probabilities do not sum to one");
    double confidence = response.at("confidence").get<double>();
    double action     = response.at("action_probability").get<double>();
    double entropy    = 0;
    for (double p : probabilities)
        if (p > 0)
            entropy -= p * std::log(p);
    if (!std::isfinite(confidence) || confidence < 0 || confidence > 1
        || std::abs(
                   confidence
                   - (1 - entropy / std::log(double(candidates.size()))))
                > 0.0001
        || !std::isfinite(action) || action < 0 || action > 1)
        throw std::runtime_error("invalid confidence or action probability");
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return probabilities[a] > probabilities[b];
    });
    auto selected = std::find(
            candidates.begin(),
            candidates.end(),
            response.at("selected").get<std::string>());
    if (selected == candidates.end()
        || probabilities[size_t(selected - candidates.begin())]
                < probabilities[order.front()])
        throw std::runtime_error("invalid selected candidate");
    (void)threshold; // Confidence is diagnostic, never permission to skip
                     // trials.
    return order;
}
std::shared_ptr<Compressor> candidate(const CompressArgs& args, size_t index)
{
    if (index == 0)
        return args.compressor();
    auto type  = integerType(*args.name());
    auto comp  = std::make_shared<Compressor>();
    auto* c    = comp->get();
    auto field = ZL_Compressor_registerFieldLZGraph(c);
    auto delta = ZL_Compressor_registerStaticGraph_fromNode1o(
            c, ZL_NODE_DELTA_INT, field);
    ZL_GraphID graph;
    switch (index) {
        case 1:
            graph = field;
            break;
        case 2:
            graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                    c, ZL_NODE_RANGE_PACK, field);
            break;
        case 3:
            graph = ZL_Compressor_registerStaticGraph_fromNode1o(
                    c, ZL_NODE_RANGE_PACK, ZL_GRAPH_ZSTD);
            break;
        case 4:
            graph = delta;
            break;
        case 5:
            graph = ZL_Compressor_registerTokenizeGraph(
                    c, ZL_Type_numeric, true, delta, field);
            break;
        case 6:
            graph = ZL_GRAPH_ZSTD;
            break;
        default:
            throw std::runtime_error("unknown candidate");
    }
    graph = ZL_Compressor_registerStaticGraph_fromNode1o(
            c,
            type.bigEndian ? ZL_Node_convertSerialToNumBE(type.width * 8)
                           : ZL_Node_convertSerialToNumLE(type.width * 8),
            graph);
    graph = ZL_Compressor_buildNumFromSerialSegmenter(
            c,
            type.width,
            args.chunkSize().value_or(ZL_DEFAULT_SEGMENTER_CHUNK_BYTE_SIZE),
            graph);
    comp->selectStartingGraph(graph);
    if (args.compressionLevel)
        comp->setParameter(CParam::CompressionLevel, *args.compressionLevel);
    return comp;
}
#ifdef OPENZL_ENABLE_LAYA
namespace {
struct FD {
    int fd;
    ~FD()
    {
        if (fd >= 0)
            close(fd);
    }
};
std::string runtimePath()
{
    return "/tmp/openzl-laya-" + std::to_string(getuid());
}
void checkRuntime()
{
    auto path = runtimePath();
    if (mkdir(path.c_str(), 0700) && errno != EEXIST)
        throw std::runtime_error("cannot create private runtime directory");
    struct stat st{};
    if (lstat(path.c_str(), &st) || !S_ISDIR(st.st_mode)
        || st.st_uid != getuid() || (st.st_mode & 0777) != 0700)
        throw std::runtime_error("unsafe Laya runtime directory");
}
int connectWorker()
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    auto path          = runtimePath() + "/worker.sock";
    std::copy(path.begin(), path.end(), address.sun_path);
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address))) {
        close(fd);
        return -1;
    }
#    ifdef SO_NOSIGPIPE
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#    endif
    fcntl(fd, F_SETFL, O_NONBLOCK);
    return fd;
}
#    ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#    else
constexpr int kSendFlags = 0;
#    endif
// The directory holding zli as invoked, so a symlinked zli finds the worker
// beside the link rather than beside the build's object cache.
std::filesystem::path executableDirectory()
{
    std::filesystem::path executable;
#    ifdef __APPLE__
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    _NSGetExecutablePath(buffer.data(), &size);
    executable = buffer.c_str();
#    else
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    std::string argv0;
    std::getline(cmdline, argv0, '\0');
    if (argv0.find('/') != std::string::npos) {
        executable = argv0;
    } else if (!argv0.empty()) {
        std::string path = getenv("PATH") ? getenv("PATH") : "";
        for (size_t start = 0; start <= path.size();) {
            size_t end = path.find(':', start);
            if (end == std::string::npos)
                end = path.size();
            auto candidate =
                    std::filesystem::path(path.substr(start, end - start))
                    / argv0;
            std::error_code ec;
            if (!path.substr(start, end - start).empty()
                && std::filesystem::is_regular_file(candidate, ec)) {
                executable = candidate;
                break;
            }
            start = end + 1;
        }
    }
    if (executable.empty()) {
        std::error_code ec;
        executable = std::filesystem::read_symlink("/proc/self/exe", ec);
    }
#    endif
    return std::filesystem::weakly_canonical(executable.parent_path());
}
char** environmentBlock()
{
#    ifdef __APPLE__
    return *_NSGetEnviron();
#    else
    return environ;
#    endif
}
void transfer(
        int fd,
        char* buffer,
        size_t count,
        bool writing,
        Clock::time_point deadline)
{
    while (count) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - Clock::now())
                                 .count();
        if (remaining <= 0)
            throw std::runtime_error("worker decision timeout");
        pollfd p{ fd, short(writing ? POLLOUT : POLLIN), 0 };
        if (poll(&p, 1, int(remaining)) < 0 && errno == EINTR)
            continue;
        auto n = writing ? send(fd, buffer, count, kSendFlags)
                         : recv(fd, buffer, count, 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
        if (n <= 0)
            throw std::runtime_error("worker disconnected");
        buffer += n;
        count -= size_t(n);
    }
}
Json requestWorker(Json request, double& startupMs, int timeout)
{
    checkRuntime();
    FD socket{ connectWorker() };
    if (socket.fd < 0) {
        auto start = Clock::now();
        // Make exposes zli as a symlink into its object cache. Resolve the
        // containing directory, not the executable's final symlink component.
        auto worker  = (executableDirectory() / "openzl-laya-worker").string();
        char* argv[] = { worker.data(), const_cast<char*>("start"), nullptr };
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(
                &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_addopen(
                &actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
        pid_t pid;
        int result = posix_spawn(
                &pid,
                worker.c_str(),
                &actions,
                nullptr,
                argv,
                environmentBlock());
        posix_spawn_file_actions_destroy(&actions);
        if (result)
            throw std::runtime_error(
                    "worker unavailable; install openzl-laya-worker beside zli");
        int status = 0;
        while (waitpid(pid, &status, WNOHANG) == 0) {
            if (elapsed(start) > 60000) {
                kill(pid, SIGTERM);
                waitpid(pid, &status, 0);
                startupMs = elapsed(start);
                throw std::runtime_error("worker startup timeout");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        startupMs = elapsed(start);
        socket.fd = connectWorker();
        if (socket.fd < 0)
            throw std::runtime_error(
                    "worker startup failed; run openzl-laya-worker prepare (or inspect start output)");
    }
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
    request["deadline_ms"] =
            std::chrono::duration<double, std::milli>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count()
            + timeout;
    auto wire = request.dump();
    if (wire.size() > 65536)
        throw std::runtime_error("request too large");
    uint32_t length = htonl(uint32_t(wire.size()));
    transfer(socket.fd, reinterpret_cast<char*>(&length), 4, true, deadline);
    transfer(socket.fd, wire.data(), wire.size(), true, deadline);
    transfer(socket.fd, reinterpret_cast<char*>(&length), 4, false, deadline);
    length = ntohl(length);
    if (!length || length > 65536)
        throw std::runtime_error("invalid response length");
    wire.resize(length);
    transfer(socket.fd, wire.data(), length, false, deadline);
    return Json::parse(wire);
}
} // namespace
#endif
Json route(CompressArgs& args)
{
    auto start = Clock::now();
    auto type  = integerType(args.name().value_or(""));
    auto data  = args.input->contents();
    Json report{ { "revision", revision },
                 { "compute_units", defaultComputeUnits },
                 { "precision", defaultPrecision },
                 { "bucket", 1024 },
                 { "selected", "numeric" },
                 { "startup_ms", 0 },
                 { "inference_ms", nullptr },
                 { "probabilities", nullptr },
                 { "action_probability", nullptr },
                 { "truncated", nullptr },
                 { "fallback_reason", nullptr },
                 { "sample_sizes", Json::array() } };
    // Let normal compression retain its empty/misaligned-input semantics.
    if (data.size() < 1024 * 1024 || data.size() % type.width) {
        report["fallback_reason"] = "small_or_misaligned_input";
    } else {
        auto windows = samples(data, type.width);
        for (auto window : windows)
            report["sample_sizes"].push_back(window.size());
        report["statistics"] = statistics(windows, type);
        const size_t chunkBytes =
                args.chunkSize().value_or(ZL_DEFAULT_SEGMENTER_CHUNK_BYTE_SIZE);
        report["statistics"]["file_bytes"]  = data.size();
        report["statistics"]["chunk_bytes"] = chunkBytes;
        report["statistics"]["compression_level"] =
                args.compressionLevel.value_or(6);
        for (size_t i = 0; i < windows.size(); ++i)
            report["statistics"]["sample_offset_" + std::to_string(i)] =
                    size_t(windows[i].data() - data.data());
        std::vector<size_t> eligible(candidates.size());
        std::iota(eligible.begin(), eligible.end(), 0);
        Json response;
        double startupMs = 0;
        try {
#ifdef OPENZL_ENABLE_LAYA
            const auto id = std::to_string(getpid()) + "-"
                    + std::to_string(Clock::now().time_since_epoch().count());
            response = requestWorker(
                    { { "version", 1 },
                      { "id", id },
                      { "command", "decide" },
                      { "candidates", candidates },
                      { "statistics", report["statistics"] },
                      { "context", args.layaContext } },
                    startupMs,
                    args.layaTimeoutMs);
            report["worker_response"] = response;
            for (const auto* key : { "probabilities",
                                     "confidence",
                                     "action_probability",
                                     "truncated",
                                     "inference_ms",
                                     "queue_ms",
                                     "compute_units",
                                     "precision",
                                     "bucket" })
                if (response.contains(key))
                    report[key] = response[key];
            eligible = validateResponse(response, id, args.layaConfidence);
            if (response.at("confidence").get<double>() < args.layaConfidence)
                report["fallback_reason"] = "low_confidence";
#else
            throw std::runtime_error("Laya disabled");
#endif
        } catch (const std::exception& e) {
            report["fallback_reason"] = e.what();
            tools::logger::Logger::log(
                    tools::logger::INFO, "Laya fallback: ", e.what());
        }
        report["startup_ms"]  = startupMs;
        report["trial_order"] = Json::array();
        for (auto index : eligible)
            report["trial_order"].push_back(candidates[index]);
        report["probe_stages"] = Json::array();
        std::array<bool, 7> excluded{};
        size_t selected = 0;
        // Frozen experimental budget: <=3 chunk-aligned windows at each of
        // 64 KiB, 1 MiB, and min(actual chunk, 16 MiB). All candidates remain
        // eligible at larger scales, even when they lost a smaller probe.
        const size_t cap =
                std::min({ chunkBytes, data.size(), size_t(16 * 1024 * 1024) });
        report["probe_cap_bytes"] = cap;
        report["probe_cap_below_chunk"] =
                cap < std::min(chunkBytes, data.size());
        size_t previousLimit = 0;
        bool escalate        = false;
        for (size_t limit : { size_t(65536), size_t(1024 * 1024), cap }) {
            limit = std::min(limit, cap);
            if (limit <= previousLimit)
                continue;
            const auto probes =
                    probeWindows(data, type.width, limit, chunkBytes);
            Json stage         = { { "limit_bytes", limit },
                                   { "samples", Json::array() },
                                   { "output_bytes", Json::object() } };
            size_t sampleBytes = 0;
            for (auto probe : probes) {
                sampleBytes += probe.size();
                stage["samples"].push_back(
                        { { "offset", size_t(probe.data() - data.data()) },
                          { "bytes", probe.size() } });
            }
            const auto stageStart          = Clock::now();
            const size_t previousSelection = selected;
            size_t best                    = std::numeric_limits<size_t>::max();
            for (auto index : eligible) {
                if (excluded[index])
                    continue;
                try {
                    auto comp = candidate(args, index);
                    CCtx cctx;
                    cctx.setParameter(CParam::StickyParameters, 1);
                    cctx.refCompressor(*comp);
                    cctx.setParameter(
                            CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
                    cctx.setParameter(CParam::PermissiveCompression, 0);
                    cctx.setParameter(
                            CParam::StoreOnExpansion,
                            args.storeOnExpansion ? ZL_TernaryParam_enable
                                                  : ZL_TernaryParam_disable);
                    if (args.compressionLevel)
                        cctx.setParameter(
                                CParam::CompressionLevel,
                                *args.compressionLevel);
                    size_t total = 0;
                    DCtx dctx;
                    for (auto probe : probes) {
                        std::string compressed(
                                2 * ZL_compressBound(probe.size()) + 1024,
                                '\0');
                        compressed.resize(
                                cctx.compressSerial(compressed, probe));
                        if (poly::string_view(dctx.decompressSerial(compressed))
                            != probe)
                            throw std::runtime_error(
                                    "sample round trip mismatch");
                        total += compressed.size();
                    }
                    stage["output_bytes"][candidates[index]] = total;
                    // Stable ties independent of model trial order; numeric
                    // first.
                    if (total < best || (total == best && index < selected)) {
                        best     = total;
                        selected = index;
                    }
                } catch (const std::exception& e) {
                    excluded[index]                       = true;
                    report["excluded"][candidates[index]] = e.what();
                }
            }
            if (best == std::numeric_limits<size_t>::max())
                throw std::runtime_error(
                        "No Laya candidate passed sample round trips: "
                        + report["excluded"].dump());
            const bool highlyCompressible =
                    double(sampleBytes) / double(best) >= 128;
            const bool longRuns =
                    report["statistics"]["mean_run"].get<double>() >= 32;
            const bool changed = previousLimit && previousSelection != selected;
            escalate           = highlyCompressible || longRuns || changed;
            stage["selected"]  = candidates[selected];
            stage["elapsed_ms"]           = elapsed(stageStart);
            stage["escalate"]             = escalate;
            report["sample_output_bytes"] = stage["output_bytes"];
            report["probe_stages"].push_back(std::move(stage));
            previousLimit = limit;
            if (!escalate)
                break;
        }
        auto chosen = candidate(args, selected);
        args.setCompressor(chosen);
        report["selected"] = candidates[selected];
    }
    report["routing_ms"] = elapsed(start);
    return report;
}
void saveCompressor(const CompressArgs& args)
{
    if (args.layaSaveCompressor) {
        auto serialized = args.compressor()->serialize();
        std::ofstream file(*args.layaSaveCompressor, std::ios::binary);
        file.write(serialized.data(), std::streamsize(serialized.size()));
        if (!file)
            throw std::runtime_error("Cannot write Laya compressor");
    }
}
void writeReport(
        const CompressArgs& args,
        Json report,
        size_t size,
        double compressionMs)
{
    if (!args.layaReport)
        return;
    report["final_size"]     = size;
    report["compression_ms"] = compressionMs;
    std::ofstream file(*args.layaReport);
    file << report.dump(2) << '\n';
    if (!file)
        throw std::runtime_error("Cannot write Laya report");
}
} // namespace openzl::cli::laya
