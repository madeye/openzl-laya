import Foundation
@testable import OpenZLLayaWorker
import XCTest

final class ProtocolTests: XCTestCase {
    func testRequestValidation() throws {
        var request = Request(version: 1, id: "test", command: "decide", candidates: candidateIDs,
                              statistics: ["elements": .number(8192)], context: "", deadline_ms: Date().timeIntervalSince1970 * 1000 + 2000)
        XCTAssertNoThrow(try request.validate())
        request.candidates = ["unknown"]
        XCTAssertThrowsError(try request.validate())
        request.candidates = candidateIDs
        request.deadline_ms = 0
        XCTAssertThrowsError(try request.validate())
        request.command = "status"
        request.version = 2
        XCTAssertThrowsError(try request.validate())
    }

    func testProtocolRoundTrip() throws {
        var sockets: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &sockets), 0)
        defer { close(sockets[0]); close(sockets[1]) }
        let response = Response(id: "hello")
        try sendResponse(sockets[0], response)
        let decoded = try JSONDecoder().decode(Response.self, from: receive(sockets[1]))
        XCTAssertEqual(decoded.id, "hello")
        XCTAssertEqual(decoded.revision, modelRevision)
    }

    func testOversizedMessages() throws {
        XCTAssertThrowsError(try sendData(-1, Data(count: maximumMessage + 1)))
    }

    func testOversizedIncomingMessage() throws {
        var sockets: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &sockets), 0)
        defer { close(sockets[0]); close(sockets[1]) }
        var length = UInt32(maximumMessage + 1).bigEndian
        _ = withUnsafeBytes(of: &length) { send(sockets[0], $0.baseAddress, 4, 0) }
        XCTAssertThrowsError(try receive(sockets[1]))
    }

    func testProcessLock() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        let first = try ProcessLock(path: path.path)
        XCTAssertThrowsError(try ProcessLock(path: path.path))
        first.close()
        XCTAssertNoThrow(try ProcessLock(path: path.path))
    }

    func testAdmissionLimit() {
        let state = ServerState()
        for _ in 0 ..< 8 {
            XCTAssertTrue(state.enter())
        }
        XCTAssertFalse(state.enter())
        state.leave()
        XCTAssertTrue(state.enter())
        state.stop()
        XCTAssertFalse(state.enter())
        for _ in 0 ..< 8 {
            state.leave()
        }
        XCTAssertTrue(state.shouldStop())
    }

    func testIdleShutdown() {
        let state = ServerState()
        XCTAssertFalse(state.shouldStop())
        XCTAssertTrue(state.shouldStop(now: .now.advanced(by: .seconds(601))))
        XCTAssertTrue(state.enter())
        XCTAssertFalse(state.shouldStop(now: .now.advanced(by: .seconds(601))))
        state.leave()
    }

    func testDigest() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("abc".utf8).write(to: file)
        XCTAssertEqual(try digest(file), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }
}
