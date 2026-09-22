import CryptoKit
import FluidUse
import Foundation

let modelRevision = "7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc"
let modelBundle = "laya_multilingual_e8_L1024_options32.mlmodelc"
let artifacts: [(String, String)] = [
    ("tokenizer.json", "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"),
    ("\(modelBundle)/analytics/coremldata.bin", "4f7e24f6023b0404bd3230182ba4950cdb414d88e837edfea395df68952917af"),
    ("\(modelBundle)/coremldata.bin", "3686bdd1170aceb97cb879ffcfc5d63e5e310606fa60c7415f296106307a96b0"),
    ("\(modelBundle)/model.mil", "4ee32f43aac0e4fe5ef66a3317fbd055b3c27802db38502061997901f012fb79"),
    ("\(modelBundle)/weights/weight.bin", "441cefaa5768572327ba89214566c6cb9a27aaacd479523b453f442c62f08eb2"),
]

struct WorkerError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) {
        self.description = description
    }
}

func cacheDirectory() -> URL {
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/OpenZL/Laya/\(modelRevision)")
}

func digest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty {
        hash.update(data: bytes)
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

func verifyAssets() throws {
    for (path, expected) in artifacts {
        guard (try? digest(cacheDirectory().appendingPathComponent(path))) == expected else {
            throw WorkerError("Missing/corrupt \(path); run openzl-laya-worker prepare")
        }
    }
}

func prepare() async throws {
    let root = cacheDirectory()
    let fm = FileManager.default
    try fm.createDirectory(at: root.deletingLastPathComponent(), withIntermediateDirectories: true)
    let lock = try ProcessLock(path: root.deletingLastPathComponent().appendingPathComponent("prepare.lock").path, wait: true)
    defer { lock.close() }
    if (try? verifyAssets()) != nil {
        try JSONEncoder().encode(Dictionary(uniqueKeysWithValues: artifacts)).write(to: root.appendingPathComponent("sha256.json"), options: .atomic)
        print("Assets verified: \(root.path)")
        return
    }
    let staging = root.deletingLastPathComponent().appendingPathComponent(".prepare-\(UUID().uuidString)")
    try fm.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: staging) }
    for (path, expected) in artifacts {
        print("Downloading \(path)")
        let destination = staging.appendingPathComponent(path)
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        // System curl honors the shell's HTTPS_PROXY/ALL_PROXY/NO_PROXY settings.
        // Arguments are passed directly, never interpreted by a shell.
        let download = Process()
        download.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        download.arguments = [
            "--fail", "--location", "--silent", "--show-error",
            "--proto", "=https", "--proto-redir", "=https",
            "--connect-timeout", "30", "--max-time", "600", "--retry", "2",
            "--output", destination.path,
            "https://huggingface.co/FluidInference/laya-coreml/resolve/\(modelRevision)/\(path)",
        ]
        try download.run()
        download.waitUntilExit()
        guard download.terminationStatus == 0, try digest(destination) == expected else {
            throw WorkerError("Artifact download/hash check failed: \(path)")
        }
    }
    let manifest = Dictionary(uniqueKeysWithValues: artifacts)
    try JSONEncoder().encode(manifest).write(to: staging.appendingPathComponent("sha256.json"))
    if fm.fileExists(atPath: root.path) { try fm.removeItem(at: root) }
    try fm.moveItem(at: staging, to: root)
    print("Prepared pinned e8/1024 assets: \(root.path)")
}
