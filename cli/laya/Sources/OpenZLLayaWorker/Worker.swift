import CoreML
import Darwin
import FluidUse
import Foundation

actor Inference {
    let manager: LayaManager
    let question: LayaQuestion
    init(manager: LayaManager) throws {
        self.manager = manager
        let descriptions = ["numeric", "FieldLZ", "range pack FieldLZ", "range pack Zstd", "delta FieldLZ", "tokenize delta alphabet FieldLZ", "Zstd"]
        question = .choice("Choose smallest compressed size.", options: descriptions)
        // FluidUse silently budgets the prompt head. Refuse any such truncation.
        let tokens = manager.encode("choice question: \(question.instructions)").count
            + descriptions.reduce(0) { $0 + 1 + manager.encode(" " + $1).count }
        guard tokens <= manager.headMaxLength else { throw WorkerError("choice question exceeds prompt head budget") }
    }

    func warmup() async throws {
        _ = try await manager.answer(state: "{}", question: question)
    }

    func decide(_ request: Request, arrival: Double) async -> Response {
        var response = Response(id: request.id)
        do {
            try request.validate()
            let now = Date().timeIntervalSince1970 * 1000
            guard now < (request.deadline_ms ?? 0) else { throw WorkerError("queue deadline exceeded") }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let statistics = try encoder.encode(request.statistics ?? [:])
            let state = String(decoding: statistics, as: UTF8.self) + "\nContext: " + (request.context ?? "")
            let start = ContinuousClock.now
            let answer = try await manager.answer(state: state, question: question)
            let duration = start.duration(to: .now)
            response.inference_ms = Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
            response.queue_ms = now - arrival
            response.selected = candidateIDs[answer.selectedIndex]
            response.candidates = candidateIDs
            response.probabilities = answer.probabilities
            response.confidence = answer.confidence
            response.action_probability = answer.actionProbability
            response.truncated = answer.stateWasTruncated
        } catch { response.error = String(describing: error) }
        return response
    }
}

/// One FIFO inference consumer. Actor reentrancy must not admit a backlog inside LayaManager.
final class ServerState: @unchecked Sendable {
    private let lock = NSLock()
    private var active = 0
    private var last = ContinuousClock.now
    private var stopping = false
    func enter() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard active < 8, !stopping else { return false }
        active += 1; last = .now; return true
    }

    func leave() {
        lock.lock(); active -= 1; last = .now; lock.unlock()
    }

    func stop() {
        lock.lock(); stopping = true; lock.unlock()
    }

    func shouldStop(now: ContinuousClock.Instant = .now) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return active == 0 && (stopping || last.duration(to: now) >= .seconds(600))
    }
}

func serve() async throws {
    try ensureRuntime()
    let lock = try ProcessLock(path: "\(runtime)/worker.lock")
    defer { lock.close() }
    // Only the lock owner removes stale sockets; never unlink another worker's socket.
    var socketInfo = stat()
    if lstat(socketPath, &socketInfo) == 0 {
        guard socketInfo.st_mode & S_IFMT == S_IFSOCK, socketInfo.st_uid == getuid() else {
            throw WorkerError("refusing to remove non-socket runtime entry")
        }
        guard unlink(socketPath) == 0 else { throw WorkerError("cannot remove stale socket") }
    } else if errno != ENOENT { throw WorkerError("cannot inspect runtime socket") }
    try verifyAssets()
    let manager = try await LayaManager.load(from: cacheDirectory(), configuration: .init(lengths: [1024], computeUnits: [1024: .all], precision: "e8"))
    let inference = try Inference(manager: manager)
    // Compile prediction kernels within the startup allowance, before readiness.
    try await inference.warmup()
    let listener = socket(AF_UNIX, SOCK_STREAM, 0)
    guard listener >= 0 else { throw WorkerError("socket failed") }
    defer { Darwin.close(listener); unlink(socketPath) }
    guard withAddress({ bind(listener, $0, $1) }) == 0, listen(listener, 16) == 0 else { throw WorkerError("cannot bind worker socket") }
    chmod(socketPath, 0o600)
    let state = ServerState()
    let inferenceQueue = DispatchQueue(label: "openzl.laya.inference")
    while !state.shouldStop() {
        var descriptor = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
        if poll(&descriptor, 1, 250) <= 0 { continue }
        let fd = accept(listener, nil, nil)
        if fd < 0 { continue }
        configure(fd)
        guard state.enter() else {
            // Close excess connections immediately; clients benchmark locally.
            Darwin.close(fd); continue
        }
        let arrival = Date().timeIntervalSince1970 * 1000
        DispatchQueue.global().async {
            do {
                let request = try JSONDecoder().decode(Request.self, from: receive(fd))
                try request.validate()
                if request.command != "decide" {
                    try sendResponse(fd, Response(id: request.id))
                    if request.command == "stop" { state.stop() }
                    Darwin.close(fd); state.leave(); return
                }
                inferenceQueue.async {
                    // Block only a dedicated dispatch thread, never Swift's cooperative executor.
                    let done = DispatchSemaphore(value: 0)
                    Task {
                        let response = await inference.decide(request, arrival: arrival)
                        try? sendResponse(fd, response)
                        Darwin.close(fd); state.leave(); done.signal()
                    }
                    done.wait()
                }
            } catch {
                Darwin.close(fd); state.leave()
            }
        }
    }
}

@main struct Worker {
    static func main() async {
        do {
            try ensureRuntime()
            switch CommandLine.arguments.dropFirst().first ?? "help" {
            case "prepare": try await prepare()
            case "serve": try await serve()
            case "start":
                if let status = try? control("status") { print("running pid=\(status.pid)"); return }
                let process = Process()
                process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
                process.arguments = ["serve"]
                process.standardInput = FileHandle.nullDevice
                let log = "\(runtime)/worker.log"
                let fd = Darwin.open(log, O_CREAT | O_WRONLY | O_APPEND | O_NOFOLLOW, 0o600)
                guard fd >= 0 else { throw WorkerError("cannot open worker log") }
                let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
                process.standardOutput = handle; process.standardError = handle
                try process.run()
                let start = ContinuousClock.now
                while start.duration(to: .now) < .seconds(59) {
                    if let status = try? control("status") { print("running pid=\(status.pid)"); return }
                    // A competing starter may own the lock; wait for that worker too.
                    if !process.isRunning, process.terminationStatus != 0,
                       let probe = try? ProcessLock(path: "\(runtime)/worker.lock")
                    {
                        probe.close()
                        throw WorkerError("Worker startup failed; run openzl-laya-worker prepare; see \(log)")
                    }
                    try await Task.sleep(for: .milliseconds(50))
                }
                throw WorkerError("startup timed out; see \(log)")
            case "status", "stop":
                let response = try control(CommandLine.arguments[1])
                try print(String(decoding: JSONEncoder().encode(response), as: UTF8.self))
                if CommandLine.arguments[1] == "stop" {
                    let start = ContinuousClock.now
                    while start.duration(to: .now) < .seconds(60) {
                        if let lock = try? ProcessLock(path: "\(runtime)/worker.lock") {
                            lock.close()
                            return
                        }
                        try await Task.sleep(for: .milliseconds(50))
                    }
                    throw WorkerError("shutdown still draining work after 60 seconds")
                }
            default: print("Usage: openzl-laya-worker prepare|start|status|stop")
            }
        } catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(1)
        }
    }
}
