#!/usr/bin/env python3
"""Laya protocol/fallback CLI tests. Uses an isolated fake worker in the private socket.

Run only with the real worker stopped. Never replaces a live worker socket.
"""

import contextlib
import json
import math
import os
import random
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ZLI = str(Path(sys.argv.pop(1)).resolve())
DISABLED_ZLI = os.environ.get("OPENZL_LAYA_DISABLED_ZLI", ZLI)
RUNTIME = Path(f"/tmp/openzl-laya-{os.getuid()}")
CANDIDATES = [
    "numeric",
    "fieldlz",
    "range_fieldlz",
    "range_zstd",
    "delta_fieldlz",
    "tokenize",
    "zstd",
]
REVISION = "7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc"


def export_candidates(directory):
    fixture = os.environ.get(
        "OPENZL_LAYA_FIXTURE", str(Path(ZLI).parent / "tests" / "laya_fixture")
    )
    subprocess.run([fixture, str(directory)], check=True, capture_output=True)
    return [Path(directory) / (c + ".compressor") for c in CANDIDATES]


def read_exact(connection, size):
    data = b""
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            raise EOFError
        data += part
    return data


@contextlib.contextmanager
def worker(transform):
    import fcntl

    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    with (RUNTIME / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RUNTIME / "worker.sock"
        path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(path))
        server.listen()
        server.settimeout(0.1)
        stopped = threading.Event()
        seen = []

        def serve():
            while not stopped.is_set():
                try:
                    client, _ = server.accept()
                except TimeoutError:
                    continue
                with client:
                    try:
                        size = struct.unpack("!I", read_exact(client, 4))[0]
                        request = json.loads(read_exact(client, size))
                        seen.append(request)
                        response = {
                            "version": 1,
                            "id": request["id"],
                            "revision": REVISION,
                            "candidates": CANDIDATES,
                            "selected": "numeric",
                            "probabilities": [1, 0, 0, 0, 0, 0, 0],
                            "confidence": 1,
                            "action_probability": 0.5,
                            "truncated": False,
                            "inference_ms": 1,
                        }
                        response = transform(response)
                        if response is None:
                            continue
                        data = json.dumps(response).encode()
                        client.sendall(struct.pack("!I", len(data)) + data)
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        pass

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            yield seen
        finally:
            stopped.set()
            thread.join()
            server.close()
            path.unlink(missing_ok=True)


class LayaIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.bin"
        self.source.write_bytes(b"".join(struct.pack("<Q", n) for n in range(131072)))

    def compress(self, *flags, expected=0, executable=ZLI, profile="le-u64"):
        report = self.root / "report.json"
        output = self.root / "output.zl"
        command = [
            executable,
            "compress",
            str(self.source),
            "--profile",
            profile,
            "--laya",
            "--laya-report",
            str(report),
            "-o",
            str(output),
            "--force",
            *flags,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=70)
        self.assertEqual(result.returncode, expected, result.stderr)
        if expected:
            return None
        restored = self.root / "restored.bin"
        subprocess.run(
            [DISABLED_ZLI, "decompress", str(output), "-o", str(restored), "--force"],
            check=True,
            capture_output=True,
        )
        self.assertEqual(restored.read_bytes(), self.source.read_bytes())
        return json.loads(report.read_text())

    def test_all_candidates_export(self):
        for index, candidate in enumerate(CANDIDATES):

            def choose(response):
                response["selected"] = candidate
                response["probabilities"] = [int(i == index) for i in range(7)]
                return response

            exported = self.root / "selected.compressor"
            with worker(choose) as requests:
                report = self.compress(
                    "--laya-save-compressor", str(exported), "--chunk-size", "32768"
                )
            self.assertEqual(report["trial_order"][0], candidate)
            self.assertEqual(len(report["sample_output_bytes"]), 7)
            self.assertEqual(len(requests), 1)
            self.assertNotIn("input", requests[0])
            subprocess.run(
                [
                    DISABLED_ZLI,
                    "compress",
                    str(self.source),
                    "--compressor",
                    str(exported),
                    "-o",
                    str(self.root / "reuse.zl"),
                    "--force",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    DISABLED_ZLI,
                    "decompress",
                    str(self.root / "reuse.zl"),
                    "-o",
                    str(self.root / "reuse.bin"),
                    "--force",
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(
                (self.root / "reuse.bin").read_bytes(), self.source.read_bytes()
            )

    def test_fallbacks(self):
        for key, value in [
            ("truncated", True),
            ("selected", "unknown"),
            ("id", "wrong"),
            ("probabilities", [math.nan] * 7),
            ("version", 2),
        ]:
            with worker(lambda r: {**r, key: value}):
                report = self.compress()
            self.assertIsNotNone(report["fallback_reason"])
            self.assertEqual(len(report["sample_output_bytes"]), 7)
        with worker(lambda r: None):
            self.assertEqual(len(self.compress()["sample_output_bytes"]), 7)

    def test_deadline(self):
        def slow(response):
            time.sleep(0.2)
            return response

        with worker(slow):
            report = self.compress("--laya-timeout-ms", "10")
        self.assertIn("timeout", report["fallback_reason"])

    def test_low_confidence(self):
        with worker(lambda r: {**r, "probabilities": [1 / 7] * 7, "confidence": 0}):
            report = self.compress()
        self.assertEqual(report["fallback_reason"], "low_confidence")
        self.assertEqual(set(report["sample_output_bytes"]), set(CANDIDATES))

    def test_ties_and_expansion(self):
        self.source.write_bytes(random.Random(314159).randbytes(1048576))
        with worker(lambda r: {**r, "truncated": True}):
            report = self.compress()
        sizes = report["sample_output_bytes"]
        self.assertEqual(sizes["numeric"], min(sizes.values()))
        self.assertEqual(report["selected"], "numeric")
        with worker(lambda r: {**r, "truncated": True}):
            report = self.compress("--no-store-on-expansion", "--level", "9")
        self.assertEqual(len(report["sample_output_bytes"]), 7)

    def test_small_input(self):
        self.source.write_bytes(bytes(64))
        with worker(lambda r: r) as requests:
            report = self.compress()
        self.assertEqual(requests, [])
        self.assertEqual(report["selected"], "numeric")

    def test_adaptive_runs(self):
        self.source.write_bytes(
            b"".join(struct.pack("<Q", n // 1024 % 17) for n in range(524288))
        )
        with worker(lambda r: r):
            report = self.compress()
        self.assertEqual(report["selected"], "delta_fieldlz")
        self.assertEqual(len(report["probe_stages"]), 3)
        self.assertEqual(
            report["probe_stages"][-1]["samples"], [{"offset": 0, "bytes": 4194304}]
        )
        self.assertEqual(set(report["sample_output_bytes"]), set(CANDIDATES))

    def test_guard_corrects_unrepresentative_prefix(self):
        rng = random.Random(173205)
        # One 32 MiB chunk exceeds the 16 MiB probe budget. Its first half
        # is monotonic; the unseen second half contains random categorical values.
        data = bytearray(struct.pack("<" + "Q" * 2097152, *range(2097152)))
        data.extend(
            struct.pack(
                "<" + "Q" * 2097152, *(rng.randrange(32) for _ in range(2097152))
            )
        )
        self.source.write_bytes(data[:33554432])
        with worker(lambda r: r):
            proposed = self.compress("--chunk-size", "33554432")
        self.assertNotEqual(proposed["selected"], "numeric")
        exported = self.root / "guarded.compressor"
        with worker(lambda r: r):
            guarded = self.compress(
                "--laya-size-guard",
                "--laya-save-compressor",
                str(exported),
                "--chunk-size",
                "33554432",
                "--trace",
                str(self.root / "guard.trace"),
            )
        self.assertTrue((self.root / "guard.trace").stat().st_size > 0)
        self.assertTrue(guarded["size_guard"]["used_numeric"])
        self.assertEqual(guarded["selected"], "numeric")
        self.assertLess(guarded["final_size"], proposed["final_size"])
        reuse = self.root / "guard-reuse.zl"
        subprocess.run(
            [
                DISABLED_ZLI,
                "compress",
                str(self.source),
                "--compressor",
                str(exported),
                "-o",
                str(reuse),
                "--force",
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(reuse.read_bytes(), (self.root / "output.zl").read_bytes())

    def test_guard_keeps_better_candidate(self):
        with worker(lambda r: r):
            guarded = self.compress("--laya-size-guard")
        self.assertEqual(guarded["selected"], "delta_fieldlz")
        self.assertFalse(guarded["size_guard"]["used_numeric"])
        self.assertLess(guarded["final_size"], guarded["size_guard"]["numeric_bytes"])

    def test_guard_integer_options(self):
        for profile in ("i8", "be-i16", "le-i32", "be-u64"):
            with worker(lambda r: r):
                report = self.compress(
                    "--laya-size-guard",
                    "--chunk-size",
                    "32768",
                    "--level",
                    "9",
                    "--no-store-on-expansion",
                    "--strict",
                    profile=profile,
                )
            self.assertLessEqual(
                report["final_size"], report["size_guard"]["numeric_bytes"]
            )

    def test_hardlinked_sidecar(self):
        sidecar = self.root / "hardlink"
        os.link(self.source, sidecar)
        self.compress("--laya-report", str(sidecar), expected=1)
        self.assertEqual(self.source.stat().st_size, 1048576)

    def test_worker_beside_executable_symlink(self):
        import fcntl

        RUNTIME.mkdir(mode=0o700, exist_ok=True)
        with (RUNTIME / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (RUNTIME / "worker.sock").unlink(missing_ok=True)
            executable = self.root / "zli"
            executable.symlink_to(ZLI)
            companion = self.root / "openzl-laya-worker"
            companion.write_text('#!/bin/sh\n: > "$0.started"\nexit 1\n')
            companion.chmod(0o700)
            report = self.compress(executable=str(executable))
            self.assertTrue(companion.with_suffix(".started").exists())
            self.assertIn("startup failed", report["fallback_reason"])

    def test_invalid_options(self):
        missing_mode = subprocess.run(
            [
                ZLI,
                "compress",
                str(self.source),
                "--profile",
                "le-u64",
                "--laya-size-guard",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(missing_mode.returncode, 0)
        self.assertIn("requires --laya", missing_mode.stderr)
        self.compress(profile="serial", expected=1)
        for flags in [
            ("--train-inline",),
            ("--laya-confidence", "nan"),
            ("--laya-timeout-ms", "0"),
            ("--laya-report", str(self.source)),
        ]:
            self.compress(*flags, expected=1)


if __name__ == "__main__":
    unittest.main()
