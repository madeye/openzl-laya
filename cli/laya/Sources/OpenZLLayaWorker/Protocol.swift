import Darwin
import Foundation

let candidateIDs = ["numeric", "fieldlz", "range_fieldlz", "range_zstd", "delta_fieldlz", "tokenize", "zstd"]
let maximumMessage = 65536
let runtime = "/tmp/openzl-laya-\(getuid())"
let socketPath = "\(runtime)/worker.sock"

struct Request: Codable {
    var version: Int
    var id: String
    var command: String
    var candidates: [String]?
    var statistics: [String: Stat]?
    var context: String?
    var deadline_ms: Double?

    func validate() throws {
        guard version == 1, !id.isEmpty, id.utf8.count <= 128,
              ["status", "stop", "decide"].contains(command)
        else { throw WorkerError("invalid request") }
        if command == "decide" {
            guard candidates == candidateIDs, let statistics, !statistics.isEmpty,
                  (context ?? "").utf8.count <= 4096, let deadline_ms, deadline_ms.isFinite,
                  deadline_ms > Date().timeIntervalSince1970 * 1000,
                  deadline_ms <= Date().timeIntervalSince1970 * 1000 + 61000
            else { throw WorkerError("invalid or expired decision request") }
        }
    }
}

enum Stat: Codable {
    case number(Double)
    case boolean(Bool)
    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let b = try? value.decode(Bool.self) { self = .boolean(b) }
        else {
            let n = try value.decode(Double.self)
            guard n.isFinite else { throw WorkerError("nonfinite statistic") }
            self = .number(n)
        }
    }

    func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case let .number(n): try value.encode(n)
        case let .boolean(b): try value.encode(b)
        }
    }
}

func residentBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.resident_size : 0
}

struct Response: Codable {
    var version = 1
    var id: String
    var revision = modelRevision
    var error: String?
    var pid = getpid()
    var resident_bytes = residentBytes()
    var selected: String?
    var candidates: [String]?
    var probabilities: [Float]?
    var confidence: Float?
    var action_probability: Float?
    var truncated: Bool?
    var inference_ms: Double?
    var queue_ms: Double?
    var compute_units = "all"
}

final class ProcessLock {
    private var fd: Int32
    init(path: String, wait: Bool = false) throws {
        fd = Darwin.open(path, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw WorkerError("cannot open process lock") }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_uid == getuid(), info.st_mode & S_IFMT == S_IFREG,
              flock(fd, LOCK_EX | (wait ? 0 : LOCK_NB)) == 0
        else { Darwin.close(fd); fd = -1; throw WorkerError("worker/prepare already running or unsafe lock") }
    }

    func close() {
        if fd >= 0 { Darwin.close(fd); fd = -1 }
    }

    deinit { close() }
}

func ensureRuntime() throws {
    if mkdir(runtime, 0o700) != 0, errno != EEXIST { throw WorkerError("cannot create runtime directory") }
    var info = stat()
    guard lstat(runtime, &info) == 0, info.st_uid == getuid(), info.st_mode & S_IFMT == S_IFDIR,
          info.st_mode & 0o777 == 0o700
    else { throw WorkerError("unsafe runtime directory") }
}

func withAddress<T>(_ body: (UnsafePointer<sockaddr>, socklen_t) throws -> T) rethrows -> T {
    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
    withUnsafeMutableBytes(of: &address.sun_path) { destination in
        socketPath.utf8CString.withUnsafeBytes { source in destination.copyBytes(from: source) }
    }
    return try withUnsafePointer(to: &address) { pointer in
        try pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { try body($0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
    }
}

func configure(_ fd: Int32) {
    var one: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one)))
    var timeout = timeval(tv_sec: 2, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
}

func readExactly(_ fd: Int32, count: Int, deadline: ContinuousClock.Instant) throws -> Data {
    var result = Data(count: count)
    try result.withUnsafeMutableBytes { bytes in
        var offset = 0
        while offset < count {
            let remaining = ContinuousClock.now.duration(to: deadline)
            guard remaining > .zero else { throw WorkerError("message read deadline exceeded") }
            let milliseconds = Int32(min(2000, max(1, remaining.components.seconds * 1000 + remaining.components.attoseconds / 1_000_000_000_000_000)))
            var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            guard poll(&descriptor, 1, milliseconds) > 0 else { throw WorkerError("message read deadline exceeded") }
            let n = recv(fd, bytes.baseAddress!.advanced(by: offset), count - offset, 0)
            if n < 0, errno == EINTR { continue }
            guard n > 0 else { throw WorkerError("client disconnected or timed out") }
            offset += n
        }
    }
    return result
}

func receive(_ fd: Int32) throws -> Data {
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    let header = try readExactly(fd, count: 4, deadline: deadline)
    let size = header.reduce(0) { ($0 << 8) | Int($1) }
    guard size > 0, size <= maximumMessage else { throw WorkerError("invalid message size") }
    return try readExactly(fd, count: size, deadline: deadline)
}

func sendResponse(_ fd: Int32, _ response: Response) throws {
    try sendData(fd, JSONEncoder().encode(response))
}

func sendData(_ fd: Int32, _ data: Data) throws {
    guard data.count <= maximumMessage else { throw WorkerError("message too large") }
    var length = UInt32(data.count).bigEndian
    var frame = withUnsafeBytes(of: &length) { Data($0) }; frame.append(data)
    try frame.withUnsafeBytes { bytes in
        var offset = 0
        while offset < frame.count {
            let n = send(fd, bytes.baseAddress!.advanced(by: offset), frame.count - offset, 0)
            if n < 0, errno == EINTR { continue }
            guard n > 0 else { throw WorkerError("client disconnected") }
            offset += n
        }
    }
}

func control(_ command: String) throws -> Response {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { throw WorkerError("socket failed") }
    defer { Darwin.close(fd) }
    configure(fd)
    guard withAddress({ connect(fd, $0, $1) }) == 0 else { throw WorkerError("worker not running") }
    let request = Request(version: 1, id: UUID().uuidString, command: command)
    try sendData(fd, JSONEncoder().encode(request))
    let response = try JSONDecoder().decode(Response.self, from: receive(fd))
    guard response.id == request.id, response.version == 1, response.revision == modelRevision else { throw WorkerError("bad control response") }
    return response
}
