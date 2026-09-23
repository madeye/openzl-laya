// Copyright (c) Meta Platforms, Inc. and affiliates.
// Local Laya worker: a private Unix-socket protocol serving routing decisions
// to zli. The worker, tokenizer and prompt construction are shared; the model
// backend is native CUDA on Linux (the upstream
// convaiinnovations/laya-multilingual checkpoint) and Core ML on macOS (the
// FluidInference/laya-coreml conversion). No Python, PyTorch or Swift runtime
// is involved.
//
// Usage: openzl-laya-worker prepare|start|status|stop|serve|check
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli/laya/model.h"
#include "cli/laya/sha256.h"
#include "cli/laya/tokenizer.h"
#include "tools/json.hpp"

#ifdef __APPLE__
#    include <crt_externs.h>
#    include <mach-o/dyld.h>
#    include <mach/mach.h>
#    define environ (*_NSGetEnviron())
#else
extern char** environ;
#endif

namespace openzl::laya {
namespace fs = std::filesystem;
using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;

#ifdef __APPLE__
// Core ML conversion: tokenizer plus the e8 (int8 embeddings, fp16 encoder)
// 1024-token bucket.
constexpr const char* kModelRepo = "FluidInference/laya-coreml";
constexpr const char* kModelRevision =
        "7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc";
#    define LAYA_BUNDLE "laya_multilingual_e8_L1024_options32.mlmodelc"
const std::vector<std::pair<std::string, std::string>> kArtifacts = {
    { "tokenizer.json",
      "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f" },
    { LAYA_BUNDLE "/analytics/coremldata.bin",
      "4f7e24f6023b0404bd3230182ba4950cdb414d88e837edfea395df68952917af" },
    { LAYA_BUNDLE "/coremldata.bin",
      "3686bdd1170aceb97cb879ffcfc5d63e5e310606fa60c7415f296106307a96b0" },
    { LAYA_BUNDLE "/model.mil",
      "4ee32f43aac0e4fe5ef66a3317fbd055b3c27802db38502061997901f012fb79" },
    { LAYA_BUNDLE "/weights/weight.bin",
      "441cefaa5768572327ba89214566c6cb9a27aaacd479523b453f442c62f08eb2" },
};
#    undef LAYA_BUNDLE
constexpr const char* kTokenizerPath = "tokenizer.json";
// An absolute path: downloads never resolve curl through PATH.
constexpr const char* kCurl = "/usr/bin/curl";
#else
constexpr const char* kModelRepo = "convaiinnovations/laya-multilingual";
constexpr const char* kModelRevision =
        "052592a15d198d9ad47da779604259b10b47b7aa";
const std::vector<std::pair<std::string, std::string>> kArtifacts = {
    { "model.safetensors",
      "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204" },
    { "rl_agent_config.json",
      "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6" },
    { "encoder/config.json",
      "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4" },
    { "tokenizer/tokenizer.json",
      "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f" },
    { "tokenizer/tokenizer_config.json",
      "424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7" },
};
constexpr const char* kTokenizerPath = "tokenizer/tokenizer.json";
constexpr const char* kCurl          = "curl";
#endif
const std::vector<std::string> kCandidates   = { "numeric",       "fieldlz",
                                                 "range_fieldlz", "range_zstd",
                                                 "delta_fieldlz", "tokenize",
                                                 "zstd" };
const std::vector<std::string> kDescriptions = {
    "numeric",
    "FieldLZ",
    "range pack FieldLZ",
    "range pack Zstd",
    "delta FieldLZ",
    "tokenize delta alphabet FieldLZ",
    "Zstd"
};
constexpr const char* kInstructions = "Choose smallest compressed size.";
constexpr size_t kMaximumMessage    = 65536;
constexpr int kMaximumConnections   = 8;
constexpr double kMessageDeadlineS  = 2.0;
constexpr double kIdleExitS         = 600.0;
constexpr int kSkipExitCode         = 77;

struct WorkerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ------------------------------------------------------------------- paths

std::string runtimeDir()
{
    return "/tmp/openzl-laya-" + std::to_string(getuid());
}
std::string socketPath()
{
    return runtimeDir() + "/worker.sock";
}
std::string lockPath()
{
    return runtimeDir() + "/worker.lock";
}
std::string logPath()
{
    return runtimeDir() + "/worker.log";
}
fs::path home()
{
    return getenv("HOME") ? fs::path(getenv("HOME")) : fs::path(".");
}
fs::path cacheRoot()
{
#ifdef __APPLE__
    return home() / "Library" / "Application Support" / "OpenZL" / "Laya";
#else
    const char* xdg = getenv("XDG_CACHE_HOME");
    fs::path base   = xdg && *xdg ? fs::path(xdg) : home() / ".cache";
    return base / "openzl" / "laya";
#endif
}
fs::path cacheDir()
{
    return cacheRoot() / kModelRevision;
}
double nowMs()
{
    return std::chrono::duration<double, std::milli>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
}

// ----------------------------------------------------------------- runtime

class ProcessLock {
   public:
    ProcessLock(const std::string& path, bool wait = false)
    {
        fd_ = open(
                path.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd_ < 0)
            throw WorkerError("cannot open process lock");
        struct stat info{};
        if (fstat(fd_, &info) || info.st_uid != getuid()
            || !S_ISREG(info.st_mode)
            || flock(fd_, LOCK_EX | (wait ? 0 : LOCK_NB))) {
            close(fd_);
            fd_ = -1;
            throw WorkerError("worker/prepare already running or unsafe lock");
        }
    }
    ~ProcessLock()
    {
        release();
    }
    void release()
    {
        if (fd_ >= 0)
            close(fd_);
        fd_ = -1;
    }

   private:
    int fd_ = -1;
};

void ensureRuntime()
{
    const auto path = runtimeDir();
    if (mkdir(path.c_str(), 0700) && errno != EEXIST)
        throw WorkerError("cannot create runtime directory");
    struct stat info{};
    if (lstat(path.c_str(), &info) || info.st_uid != getuid()
        || !S_ISDIR(info.st_mode) || (info.st_mode & 0777) != 0700)
        throw WorkerError("unsafe runtime directory");
}

size_t residentBytes()
{
#ifdef __APPLE__
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info),
                &count)
        != KERN_SUCCESS)
        return 0;
    return size_t(info.resident_size);
#else
    std::ifstream statm("/proc/self/statm");
    size_t size = 0, resident = 0;
    statm >> size >> resident;
    return resident * size_t(sysconf(_SC_PAGESIZE));
#endif
}

/// Path of this executable, for respawning it as the server.
std::string selfPath()
{
#ifdef __APPLE__
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size))
        throw WorkerError("cannot locate worker executable");
    return fs::canonical(buffer.c_str()).string();
#else
    return fs::read_symlink("/proc/self/exe").string();
#endif
}

// ------------------------------------------------------------------ sockets

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0; // macOS: SO_NOSIGPIPE is set per socket
#endif

/// Marks a new descriptor close-on-exec and suppresses SIGPIPE on it where
/// send() has no MSG_NOSIGNAL.
int prepareSocket(int fd)
{
    if (fd < 0)
        return fd;
    fcntl(fd, F_SETFD, FD_CLOEXEC);
#ifdef SO_NOSIGPIPE
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    return fd;
}
int unixSocket()
{
#ifdef SOCK_CLOEXEC
    return prepareSocket(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
#else
    return prepareSocket(socket(AF_UNIX, SOCK_STREAM, 0));
#endif
}
int acceptClient(int listener)
{
#ifdef __APPLE__
    return prepareSocket(accept(listener, nullptr, nullptr));
#else
    return prepareSocket(accept4(listener, nullptr, nullptr, SOCK_CLOEXEC));
#endif
}

// --------------------------------------------------------------- messaging

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
            throw WorkerError(
                    writing ? "message write deadline exceeded"
                            : "message read deadline exceeded");
        pollfd p{ fd, short(writing ? POLLOUT : POLLIN), 0 };
        int ready = poll(&p, 1, int(std::min<long>(remaining, 2000)));
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready == 0)
            continue;
        ssize_t n = writing ? send(fd, buffer, count, kSendFlags)
                            : recv(fd, buffer, count, 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
        if (n <= 0)
            throw WorkerError("client disconnected or timed out");
        buffer += n;
        count -= size_t(n);
    }
}

std::string receive(int fd)
{
    auto deadline = Clock::now()
            + std::chrono::milliseconds(int(kMessageDeadlineS * 1000));
    uint32_t header;
    transfer(fd, reinterpret_cast<char*>(&header), 4, false, deadline);
    const size_t size = __builtin_bswap32(header);
    if (size == 0 || size > kMaximumMessage)
        throw WorkerError("invalid message size");
    std::string payload(size, '\0');
    transfer(fd, payload.data(), size, false, deadline);
    return payload;
}

void sendMessage(int fd, const Json& payload)
{
    auto data = payload.dump();
    if (data.size() > kMaximumMessage)
        throw WorkerError("message too large");
    uint32_t header = __builtin_bswap32(uint32_t(data.size()));
    std::string frame(reinterpret_cast<char*>(&header), 4);
    frame += data;
    transfer(
            fd,
            frame.data(),
            frame.size(),
            true,
            Clock::now()
                    + std::chrono::milliseconds(int(kMessageDeadlineS * 1000)));
}

Json baseResponse(const std::string& id)
{
    return { { "version", 1 },
             { "id", id },
             { "revision", kModelRevision },
             { "pid", getpid() },
             { "resident_bytes", residentBytes() } };
}

std::string uuid()
{
    static thread_local std::mt19937_64 rng{ std::random_device{}() };
    char text[40];
    uint64_t a = rng(), b = rng();
    snprintf(
            text,
            sizeof(text),
            "%08x-%04x-%04x-%04x-%012llx",
            uint32_t(a >> 32),
            uint32_t(a >> 16) & 0xFFFF,
            uint32_t(a) & 0xFFFF,
            uint32_t(b >> 48) & 0xFFFF,
            (unsigned long long)(b & 0xFFFFFFFFFFFFULL));
    return text;
}

void validateRequest(const Json& request)
{
    if (!request.is_object())
        throw WorkerError("invalid request");
    auto id = request.value("id", Json());
    if (request.value("version", 0) != 1 || !id.is_string()
        || id.get<std::string>().empty() || id.get<std::string>().size() > 128)
        throw WorkerError("invalid request");
    const auto command = request.value("command", "");
    if (command != "status" && command != "stop" && command != "decide")
        throw WorkerError("invalid request");
    if (command == "decide") {
        const auto& statistics = request.value("statistics", Json());
        const auto& context    = request.value("context", Json(""));
        const auto& deadline   = request.value("deadline_ms", Json());
        const double now       = nowMs();
        if (request.value("candidates", Json()) != Json(kCandidates)
            || !statistics.is_object() || statistics.empty()
            || !context.is_string() || context.get<std::string>().size() > 4096
            || !deadline.is_number() || !std::isfinite(deadline.get<double>())
            || deadline.get<double>() <= now
            || deadline.get<double>() > now + 61000)
            throw WorkerError("invalid or expired decision request");
        for (const auto& [key, value] : statistics.items()) {
            if (value.is_boolean())
                continue;
            if (!value.is_number() || !std::isfinite(value.get<double>()))
                throw WorkerError("nonfinite statistic");
        }
    }
}

int connectWorker()
{
    int fd = unixSocket();
    if (fd < 0)
        throw WorkerError("socket failed");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    auto path          = socketPath();
    std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address))) {
        close(fd);
        throw WorkerError("worker not running");
    }
    return fd;
}

Json control(const std::string& command)
{
    int fd = connectWorker();
    Json response;
    try {
        const auto id = uuid();
        sendMessage(
                fd, { { "version", 1 }, { "id", id }, { "command", command } });
        response = Json::parse(receive(fd));
        if (response.value("id", "") != id || response.value("version", 0) != 1
            || response.value("revision", "") != kModelRevision)
            throw WorkerError("bad control response");
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
    return response;
}

// ------------------------------------------------------------------ assets

void verifyAssets()
{
    for (const auto& [relative, expected] : kArtifacts)
        if (Sha256::file((cacheDir() / relative).string()) != expected)
            throw WorkerError(
                    "Missing/corrupt " + relative
                    + "; run openzl-laya-worker prepare");
}

int runProcess(const std::vector<std::string>& argv)
{
    std::vector<char*> args;
    for (const auto& a : argv)
        args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    pid_t pid;
    if (posix_spawnp(
                &pid, argv[0].c_str(), nullptr, nullptr, args.data(), environ))
        return -1;
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

void prepare()
{
    fs::create_directories(cacheRoot());
    ProcessLock lock((cacheRoot() / "prepare.lock").string(), true);
    Json manifest = Json::object();
    for (const auto& [relative, expected] : kArtifacts)
        manifest[relative] = expected;
    try {
        verifyAssets();
        std::ofstream(cacheDir() / "sha256.json") << manifest.dump(2) << '\n';
        std::cout << "Assets verified: " << cacheDir().string() << '\n';
        return;
    } catch (const WorkerError&) {
    }
    const fs::path staging = cacheRoot() / (".prepare-" + uuid());
    fs::create_directories(staging);
    try {
        for (const auto& [relative, expected] : kArtifacts) {
            std::cout << "Downloading " << relative << std::endl;
            const fs::path destination = staging / relative;
            fs::create_directories(destination.parent_path());
            // System curl honors HTTPS_PROXY/ALL_PROXY/NO_PROXY; arguments
            // never pass through a shell.
            const int status = runProcess(
                    { kCurl,
                      "--fail",
                      "--location",
                      "--silent",
                      "--show-error",
                      "--proto",
                      "=https",
                      "--proto-redir",
                      "=https",
                      "--connect-timeout",
                      "30",
                      "--max-time",
                      "1800",
                      "--retry",
                      "2",
                      "--output",
                      destination.string(),
                      std::string("https://huggingface.co/") + kModelRepo
                              + "/resolve/" + kModelRevision + "/"
                              + relative });
            if (status != 0 || Sha256::file(destination.string()) != expected)
                throw WorkerError(
                        "Artifact download/hash check failed: " + relative);
        }
        std::ofstream(staging / "sha256.json") << manifest.dump(2) << '\n';
        if (fs::exists(cacheDir()))
            fs::remove_all(cacheDir());
        fs::rename(staging, cacheDir());
    } catch (...) {
        fs::remove_all(staging);
        throw;
    }
    std::cout << "Prepared pinned " << kModelRepo << "@"
              << std::string(kModelRevision).substr(0, 12)
              << " assets: " << cacheDir().string() << '\n';
}

// --------------------------------------------------------------- inference

struct Answer {
    std::vector<double> probabilities;
    double confidence        = 0;
    double actionProbability = 0;
    bool truncated           = false;
    int tokens               = 0;
    int paddedTokens         = 0;
    float deviceMs           = 0;
};

class Inference {
   public:
    Inference()
            : tokenizer_((cacheDir() / kTokenizerPath).string()),
              model_(cacheDir().string(), tokenizer_.padId())
    {
        buildHead();
        if (int(markers_.size()) != int(kDescriptions.size()))
            throw WorkerError("choice question exceeds prompt head budget");
    }

    std::vector<int> encode(const std::string& text) const
    {
        return tokenizer_.encode(text);
    }
    const std::vector<int>& head() const
    {
        return head_;
    }
    const std::vector<int>& markers() const
    {
        return markers_;
    }
    const Model& model() const
    {
        return model_;
    }

    int paddedLength(int tokens) const
    {
        return model_.paddedLength(tokens);
    }

    /// Port of laya `build_sequence` (state part) and FluidUse's builder.
    void sequence(
            const std::string& state,
            std::vector<int>& ids,
            bool& truncated) const
    {
        auto stateIds = encode(replaceMask(state));
        const size_t room =
                size_t(std::max(0, model_.maxLength() - int(head_.size()) - 1));
        truncated = stateIds.size() > room;
        if (truncated)
            stateIds.resize(room);
        ids = head_;
        ids.insert(ids.end(), stateIds.begin(), stateIds.end());
        ids.push_back(tokenizer_.sepId());
        if (int(ids.size()) > model_.maxLength())
            ids.resize(size_t(model_.maxLength()));
    }

    Answer answer(const std::string& state, bool useGraph = true)
    {
        std::vector<int> ids;
        Answer out;
        sequence(state, ids, out.truncated);
        out.tokens       = int(ids.size());
        out.paddedTokens = paddedLength(out.tokens);
        auto result  = model_.infer(ids, markers_, out.paddedTokens, useGraph);
        out.deviceMs = result.milliseconds;
        const int count    = int(result.logits.size());
        const double scale = std::max(1e-3, double(model_.temperature(count)));
        double peak        = -1e30;
        for (float z : result.logits)
            peak = std::max(peak, double(z) / scale);
        double total = 0;
        out.probabilities.resize(size_t(count));
        for (int i = 0; i < count; ++i) {
            out.probabilities[size_t(i)] =
                    std::exp(double(result.logits[size_t(i)]) / scale - peak);
            total += out.probabilities[size_t(i)];
        }
        double entropy = 0;
        for (auto& p : out.probabilities) {
            p /= total;
            entropy -= p * std::log(std::max(p, 1e-12));
        }
        out.confidence        = count < 2
                       ? 1.0
                       : std::min(
                          1.0,
                          std::max(0.0, 1 - entropy / std::log(double(count))));
        out.actionProbability = result.actionProbability;
        return out;
    }

    Json decide(const Json& request, double arrivalMs)
    {
        Json response = baseResponse(request.value("id", ""));
        try {
            validateRequest(request);
            const double now = nowMs();
            if (now >= request.at("deadline_ms").get<double>())
                throw WorkerError("queue deadline exceeded");
            // Stable key order, compact separators, non-ASCII kept as-is:
            // identical bytes to json.dumps(sort_keys=True, separators=(",",
            // ":")).
            std::string state = canonical(request.at("statistics"))
                    + "\nContext: " + request.value("context", "");
            const auto start = Clock::now();
            auto result      = answer(state);
            response["inference_ms"] =
                    std::chrono::duration<double, std::milli>(
                            Clock::now() - start)
                            .count();
            response["device_ms"] = result.deviceMs;
            response["queue_ms"]  = now - arrivalMs;
            size_t best           = 0;
            for (size_t i = 1; i < result.probabilities.size(); ++i)
                if (result.probabilities[i] > result.probabilities[best])
                    best = i;
            response["selected"]           = kCandidates[best];
            response["candidates"]         = kCandidates;
            response["probabilities"]      = result.probabilities;
            response["confidence"]         = result.confidence;
            response["action_probability"] = result.actionProbability;
            response["truncated"]          = result.truncated;
            response["token_count"]        = result.tokens;
            response["padded_tokens"]      = result.paddedTokens;
        } catch (const std::exception& error) {
            response["error"] = error.what();
        }
        response["compute_units"] = model_.computeUnits();
        response["precision"]     = model_.precision();
        response["bucket"]        = model_.maxLength();
        response["backend"]       = model_.backend();
        return response;
    }

    void warmup()
    {
        // Statistics prompts span roughly 300-450 tokens; record those CUDA
        // graphs (Core ML runs one fixed length, so this warms it once).
        std::set<int> lengths;
        for (int n : { 200, 320, 384, 448, 512 })
            lengths.insert(paddedLength(n));
        for (int length : lengths) {
            const int target = std::min(length, model_.maxLength()) - 1;
            std::string state;
            std::vector<int> ids;
            bool truncated;
            for (int words = std::max(1, target - int(head_.size()) - 1);;
                 ++words) {
                state.clear();
                for (int w = 0; w < words; ++w)
                    state += w ? " a" : "a";
                sequence(state, ids, truncated);
                if (int(ids.size()) >= target)
                    break;
            }
            for (int repeat = 0; repeat < 2; ++repeat)
                answer(state);
        }
    }

    /// Byte-identical to Python's json.dumps(value, sort_keys=True,
    /// separators=(",", ":"), ensure_ascii=False): nlohmann objects already
    /// iterate in key order, but its float layout differs from Python's repr
    /// around 1e15-1e16, so numbers are formatted here.
   public:
    static std::string canonical(const Json& value)
    {
        std::string out;
        canonicalInto(value, out);
        return out;
    }

    static std::string pythonFloat(double v)
    {
        if (std::isnan(v))
            return "NaN";
        if (std::isinf(v))
            return v > 0 ? "Infinity" : "-Infinity";
        if (v == 0)
            return std::signbit(v) ? "-0.0" : "0.0";
        // Shortest round-trip digits in scientific form: d.ddddde[+-]xx
        char buffer[64];
        auto result = std::to_chars(
                buffer,
                buffer + sizeof(buffer),
                v,
                std::chars_format::scientific);
        std::string text(buffer, result.ptr);
        std::string sign;
        if (text[0] == '-') {
            sign = "-";
            text.erase(0, 1);
        }
        const auto e       = text.find('e');
        std::string digits = text.substr(0, e);
        digits.erase(
                std::remove(digits.begin(), digits.end(), '.'), digits.end());
        const int exponent = std::stoi(text.substr(e + 1));
        const int decpt    = exponent + 1; // value = 0.digits * 10^decpt
        std::string body;
        if (decpt <= -4 || decpt > 16) {
            body = digits.substr(0, 1);
            if (digits.size() > 1)
                body += "." + digits.substr(1);
            char exp[8];
            snprintf(
                    exp,
                    sizeof(exp),
                    "e%c%02d",
                    exponent < 0 ? '-' : '+',
                    std::abs(exponent));
            body += exp;
        } else if (decpt <= 0) {
            body = "0." + std::string(size_t(-decpt), '0') + digits;
        } else if (size_t(decpt) < digits.size()) {
            body = digits.substr(0, size_t(decpt)) + "."
                    + digits.substr(size_t(decpt));
        } else {
            body = digits + std::string(size_t(decpt) - digits.size(), '0')
                    + ".0";
        }
        return sign + body;
    }

    static void canonicalInto(const Json& value, std::string& out)
    {
        switch (value.type()) {
            case Json::value_t::object: {
                out += '{';
                bool first = true;
                for (auto it = value.begin(); it != value.end(); ++it) {
                    if (!first)
                        out += ',';
                    first = false;
                    out += Json(it.key()).dump(
                            -1, ' ', false, Json::error_handler_t::replace);
                    out += ':';
                    canonicalInto(it.value(), out);
                }
                out += '}';
                break;
            }
            case Json::value_t::array: {
                out += '[';
                bool first = true;
                for (const auto& item : value) {
                    if (!first)
                        out += ',';
                    first = false;
                    canonicalInto(item, out);
                }
                out += ']';
                break;
            }
            case Json::value_t::number_float:
                out += pythonFloat(value.get<double>());
                break;
            default:
                out += value.dump(
                        -1, ' ', false, Json::error_handler_t::replace);
        }
    }

   private:
    static std::string replaceMask(std::string text)
    {
        for (size_t pos; (pos = text.find("<mask>")) != std::string::npos;)
            text.replace(pos, 6, " ");
        return text;
    }

    void buildHead()
    {
        const int headMax = model_.headMaxLength();
        auto headIds      = encode(
                std::string("choice question: ") + replaceMask(kInstructions));
        std::vector<std::vector<int>> options;
        for (const auto& option : kDescriptions) {
            auto ids = encode(" " + replaceMask(option));
            if (ids.size() > 48)
                ids.resize(48);
            ids.insert(ids.begin(), tokenizer_.maskId());
            options.push_back(ids);
        }
        auto used = [&] {
            size_t n = 0;
            for (auto& o : options)
                n += o.size();
            return int(n);
        };
        int budget = headMax - used();
        if (budget < 16) {
            const int per = std::max(
                    4, (headMax - 16) / std::max<int>(1, int(options.size())));
            for (auto& o : options)
                if (int(o.size()) > per)
                    o.resize(size_t(per));
            budget = headMax - used();
        }
        if (int(headIds.size()) > std::max(8, budget))
            headIds.resize(size_t(std::max(8, budget)));
        head_ = { tokenizer_.clsId() };
        head_.insert(head_.end(), headIds.begin(), headIds.end());
        head_.push_back(tokenizer_.sepId());
        for (const auto& o : options) {
            markers_.push_back(int(head_.size()));
            head_.insert(head_.end(), o.begin(), o.end());
        }
        head_.push_back(tokenizer_.sepId());
    }

    Tokenizer tokenizer_;
    Model model_;
    std::vector<int> head_;
    std::vector<int> markers_;
};

// ------------------------------------------------------------------ server

struct ServerState {
    std::mutex mutex;
    int active             = 0;
    Clock::time_point last = Clock::now();
    bool stopping          = false;

    bool enter()
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (active >= kMaximumConnections || stopping)
            return false;
        ++active;
        last = Clock::now();
        return true;
    }
    void leave()
    {
        std::lock_guard<std::mutex> guard(mutex);
        --active;
        last = Clock::now();
    }
    void stop()
    {
        std::lock_guard<std::mutex> guard(mutex);
        stopping = true;
    }
    bool shouldStop()
    {
        std::lock_guard<std::mutex> guard(mutex);
        return active == 0
                && (stopping
                    || std::chrono::duration<double>(Clock::now() - last)
                                    .count()
                            >= kIdleExitS);
    }
};

struct Job {
    int fd;
    Json request;
    double arrivalMs;
};

void serve()
{
    ensureRuntime();
    ProcessLock lock(lockPath());
    // Only the lock owner removes stale sockets; never unlink another
    // worker's socket.
    struct stat info{};
    if (lstat(socketPath().c_str(), &info) == 0) {
        if (!S_ISSOCK(info.st_mode) || info.st_uid != getuid())
            throw WorkerError("refusing to remove non-socket runtime entry");
        if (unlink(socketPath().c_str()))
            throw WorkerError("cannot remove stale socket");
    } else if (errno != ENOENT) {
        throw WorkerError("cannot inspect runtime socket");
    }
    verifyAssets();
    const auto started = Clock::now();
    Inference inference;
    std::cout << "model ready after "
              << std::chrono::duration<double, std::milli>(
                         Clock::now() - started)
                         .count()
              << " ms on " << inference.model().deviceName() << std::endl;
    // Warm the usual prompt lengths (CUDA graphs, Core ML kernels) before
    // readiness.
    inference.warmup();
    std::cout << "loaded " << kModelRepo << "@"
              << std::string(kModelRevision).substr(0, 12) << " on "
              << inference.model().computeUnits() << " ("
              << inference.model().precision() << ") in "
              << std::chrono::duration<double, std::milli>(
                         Clock::now() - started)
                         .count()
              << " ms" << std::endl;

    int listener = unixSocket();
    if (listener < 0)
        throw WorkerError("socket failed");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(
            address.sun_path,
            socketPath().c_str(),
            sizeof(address.sun_path) - 1);
    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address))
        || listen(listener, 16))
        throw WorkerError("cannot bind worker socket");
    chmod(socketPath().c_str(), 0600);

    ServerState state;
    std::mutex jobsMutex;
    std::condition_variable jobsReady;
    std::queue<Job> jobs;
    std::atomic<bool> done{ false };

    auto finish = [&](int fd) {
        close(fd);
        state.leave();
    };
    auto handle = [&](int fd, double arrivalMs) {
        Json request;
        try {
            request = Json::parse(receive(fd));
            validateRequest(request);
        } catch (const std::exception&) {
            finish(fd);
            return;
        }
        const auto command = request.value("command", "");
        if (command != "decide") {
            try {
                sendMessage(fd, baseResponse(request.value("id", "")));
            } catch (const std::exception&) {
            }
            if (command == "stop")
                state.stop();
            finish(fd);
            return;
        }
        {
            std::lock_guard<std::mutex> guard(jobsMutex);
            jobs.push({ fd, std::move(request), arrivalMs });
        }
        jobsReady.notify_one();
    };
    std::thread acceptor([&] {
        while (!state.shouldStop()) {
            pollfd p{ listener, POLLIN, 0 };
            if (poll(&p, 1, 250) <= 0)
                continue;
            int fd = acceptClient(listener);
            if (fd < 0)
                continue;
            if (!state.enter()) {
                // Close excess connections immediately; clients benchmark
                // locally.
                close(fd);
                continue;
            }
            const double arrivalMs = nowMs();
            std::thread(handle, fd, arrivalMs).detach();
        }
        done = true;
        jobsReady.notify_all();
    });
    // One FIFO inference consumer on this thread.
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> guard(jobsMutex);
            jobsReady.wait(guard, [&] { return done || !jobs.empty(); });
            if (jobs.empty())
                break;
            job = std::move(jobs.front());
            jobs.pop();
        }
        try {
            sendMessage(job.fd, inference.decide(job.request, job.arrivalMs));
        } catch (const std::exception&) {
        }
        finish(job.fd);
    }
    acceptor.join();
    close(listener);
    unlink(socketPath().c_str());
}

void start()
{
    try {
        auto status = control("status");
        std::cout << "running pid=" << status.value("pid", 0) << '\n';
        return;
    } catch (const WorkerError&) {
    }
    const std::string self = selfPath();
    int log =
            open(logPath().c_str(),
                 O_CREAT | O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC,
                 0600);
    if (log < 0)
        throw WorkerError("cannot open worker log");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(
            &actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, log, STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, log, STDERR_FILENO);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSID);
    std::vector<char*> argv{ const_cast<char*>(self.c_str()),
                             const_cast<char*>("serve"),
                             nullptr };
    pid_t pid;
    const int spawned = posix_spawn(
            &pid, self.c_str(), &actions, &attributes, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    close(log);
    if (spawned)
        throw WorkerError("cannot spawn worker");
    const auto started = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - started).count() < 59) {
        try {
            auto status = control("status");
            std::cout << "running pid=" << status.value("pid", 0) << '\n';
            return;
        } catch (const WorkerError&) {
        }
        // A competing starter may own the lock; wait for that worker too.
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid
            && (!WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
            try {
                ProcessLock probe(lockPath());
                probe.release();
                throw WorkerError(
                        "Worker startup failed; run openzl-laya-worker prepare; see "
                        + logPath());
            } catch (const WorkerError& error) {
                if (std::string(error.what()).rfind("Worker startup failed", 0)
                    == 0)
                    throw;
            }
        }
        usleep(50000);
    }
    throw WorkerError("startup timed out; see " + logPath());
}

void stop()
{
    std::cout << control("stop").dump() << '\n';
    const auto started = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - started).count() < 60) {
        try {
            ProcessLock probe(lockPath());
            return;
        } catch (const WorkerError&) {
            usleep(50000);
        }
    }
    throw WorkerError("shutdown still draining work after 60 seconds");
}

// ------------------------------------------------------------------- check

int check(
        const std::string& referencePath,
        const std::string& tokenizerReferencePath)
{
    try {
        verifyAssets();
    } catch (const WorkerError& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return kSkipExitCode;
    }
    Inference inference;
    int failures = 0;
    {
        std::ifstream file(tokenizerReferencePath);
        Json fixture = Json::parse(file);
        int ok       = 0;
        for (auto it = fixture.begin(); it != fixture.end(); ++it) {
            if (inference.encode(it.key())
                == it.value().get<std::vector<int>>())
                ++ok;
            else
                ++failures;
        }
        std::cout << "tokenizer: " << ok << " strings match, " << failures
                  << " differ\n";
    }
    std::ifstream file(referencePath);
    Json fixture = Json::parse(file);
    if (fixture.at("head").get<std::vector<int>>() != inference.head()
        || fixture.at("markers").get<std::vector<int>>()
                != inference.markers()) {
        std::cout << "prompt head differs from the reference\n";
        ++failures;
    }
    double maxProbability = 0, maxConfidence = 0, maxAction = 0;
    int orderings = 0, selections = 0, total = 0;
    for (auto it = fixture.at("states").begin();
         it != fixture.at("states").end();
         ++it) {
        const auto& entry = it.value();
        std::vector<int> ids;
        bool truncated;
        const auto state = entry.at("state").get<std::string>();
        if (const auto split = state.find("\nContext: ");
            split != std::string::npos) {
            const auto rebuilt =
                    Inference::canonical(Json::parse(state.substr(0, split)));
            if (rebuilt != state.substr(0, split)) {
                std::cout << "canonical JSON differs for " << it.key()
                          << ":\n  " << rebuilt.substr(0, 200) << "\n";
                ++failures;
            }
        }
        inference.sequence(state, ids, truncated);
        if (ids != entry.at("ids").get<std::vector<int>>()
            || truncated != entry.at("truncated").get<bool>()) {
            std::cout << "sequence differs for " << it.key() << '\n';
            ++failures;
            continue;
        }
        auto answer   = inference.answer(entry.at("state").get<std::string>());
        auto expected = entry.at("probabilities").get<std::vector<double>>();
        auto order    = [](const std::vector<double>& p) {
            std::vector<int> o(p.size());
            for (size_t i = 0; i < o.size(); ++i)
                o[i] = int(i);
            std::sort(o.begin(), o.end(), [&](int a, int b) {
                return p[size_t(a)] > p[size_t(b)];
            });
            return o;
        };
        for (size_t i = 0; i < expected.size(); ++i)
            maxProbability = std::max(
                    maxProbability,
                    std::fabs(answer.probabilities[i] - expected[i]));
        maxConfidence = std::max(
                maxConfidence,
                std::fabs(
                        answer.confidence
                        - entry.at("confidence").get<double>()));
        maxAction = std::max(
                maxAction,
                std::fabs(
                        answer.actionProbability
                        - entry.at("action_probability").get<double>()));
        orderings += order(answer.probabilities) == order(expected);
        selections += order(answer.probabilities)[0] == order(expected)[0];
        ++total;
    }
    std::cout << "model: " << total << " states, max |dp| " << maxProbability
              << ", max |dconfidence| " << maxConfidence << ", max |daction| "
              << maxAction << ", identical orderings " << orderings << "/"
              << total << ", identical selections " << selections << "/"
              << total << '\n';
    // The CUDA backend reproduces the fp32 reference closely. The Core ML e8
    // conversion (int8 embeddings, fp16 encoder) reorders near-tied
    // low-probability candidates, so it is held to the decision itself.
    const bool coreml = inference.model().backend() == "coreml";
    if (coreml ? selections != total || maxProbability > 0.05
               : orderings != total || maxProbability > 0.01)
        ++failures;
    return failures ? 1 : 0;
}

int main(int argc, char** argv)
{
    const std::string command = argc > 1 ? argv[1] : "help";
    // Clients may disconnect mid-reply; a write must fail, not kill us.
    signal(SIGPIPE, SIG_IGN);
    try {
        ensureRuntime();
        if (command == "prepare")
            prepare();
        else if (command == "serve")
            serve();
        else if (command == "start")
            start();
        else if (command == "status")
            std::cout << control("status").dump() << '\n';
        else if (command == "stop")
            stop();
        else if (command == "check" && argc > 3)
            return check(argv[2], argv[3]);
        else
            std::cout
                    << "Usage: openzl-laya-worker prepare|start|status|stop|check REFERENCE TOKENIZER_REFERENCE\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

} // namespace openzl::laya

int main(int argc, char** argv)
{
    return openzl::laya::main(argc, argv);
}
