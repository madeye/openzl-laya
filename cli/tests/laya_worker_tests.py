#!/usr/bin/env python3
"""Real local worker lifecycle tests; requires prepared pinned assets."""

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

WORKER = str(Path(sys.argv[1]).resolve())
SOCKET = f"/tmp/openzl-laya-{os.getuid()}/worker.sock"
IDS = [
    "numeric",
    "fieldlz",
    "range_fieldlz",
    "range_zstd",
    "delta_fieldlz",
    "tokenize",
    "zstd",
]
CACHE = (
    Path.home()
    / "Library/Application Support/OpenZL/Laya/7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc"
)


def command(verb, check=True):
    return subprocess.run(
        [WORKER, verb], capture_output=True, text=True, timeout=65, check=check
    )


def stop():
    command("stop", False)
    for _ in range(100):
        if not Path(SOCKET).exists():
            return
        time.sleep(0.05)
    raise AssertionError("worker did not stop")


def pid():
    return json.loads(command("status").stdout)["pid"]


def request(context="", timeout_ms=60000):
    return {
        "version": 1,
        "id": "lifecycle-test",
        "command": "decide",
        "candidates": IDS,
        "statistics": {
            "width": 8,
            "cardinality_ratio": 1,
            "monotonicity": 1,
            "delta_p50": 1,
        },
        "context": context,
        "deadline_ms": time.time() * 1000 + timeout_ms,
    }


def receive(connection, size):
    data = b""
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def send(payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(65)
        connection.connect(SOCKET)
        data = json.dumps(payload).encode()
        connection.sendall(struct.pack("!I", len(data)) + data)
        size = struct.unpack("!I", receive(connection, 4))[0]
        return json.loads(receive(connection, size))


stop()
# Concurrent starts must converge to a single loaded process.
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    starts = list(pool.map(lambda _: command("start"), range(4)))
assert len({result.stdout.strip() for result in starts}) == 1
original = pid()
response = send(request())
assert "error" not in response, response
assert response["candidates"] == IDS
second = send(request())
assert second["selected"] == response["selected"]
assert (
    max(abs(a - b) for a, b in zip(second["probabilities"], response["probabilities"]))
    < 0.0001
)
assert pid() == original
# An abandoned request cannot kill another client's work.
with socket.socket(socket.AF_UNIX) as connection:
    connection.connect(SOCKET)
    data = json.dumps(request(timeout_ms=1)).encode()
    connection.sendall(struct.pack("!I", len(data)) + data)
assert "error" not in send(request())
assert pid() == original
# Saturated readers are bounded, and release on aggregate read deadlines.
held = []
for _ in range(8):
    connection = socket.socket(socket.AF_UNIX)
    connection.connect(SOCKET)
    held.append(connection)
time.sleep(0.1)
with socket.socket(socket.AF_UNIX) as excess:
    excess.settimeout(1)
    excess.connect(SOCKET)
    assert excess.recv(1) == b""
for connection in held:
    connection.close()
time.sleep(0.2)
assert pid() == original
# Large state must be explicitly reported as truncated.
response = send(request(context="x " * 2048))
assert response["truncated"] is True, response
# Crash leaves a stale socket; the next lock owner recovers it.
os.kill(original, signal.SIGKILL)
time.sleep(0.2)
command("start")
assert pid() != original
stop()
# Missing and corrupt assets must not trigger implicit downloads.
tokenizer = CACHE / "tokenizer.json"
saved = tokenizer.with_suffix(".test-backup")
tokenizer.rename(saved)
try:
    assert command("start", False).returncode != 0
    tokenizer.write_text("corrupt")
    assert command("start", False).returncode != 0
finally:
    tokenizer.unlink(missing_ok=True)
    saved.rename(tokenizer)
command("prepare")
command("start")
assert "error" not in send(request())
stop()
print(
    "PASS: concurrent startup, reuse, abandonment, saturation, truncation, crash/stale socket, missing/corrupt assets, stop"
)
