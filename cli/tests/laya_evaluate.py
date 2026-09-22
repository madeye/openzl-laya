#!/usr/bin/env python3
"""Frozen synthetic evaluation; see doc/laya.md. Worker must initially be stopped."""

import json
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import tempfile
import time

# The integration module consumes argv[1], the absolute zli path.
from laya_integration_tests import CANDIDATES, ZLI, export_candidates, worker

SEED = 271828
ELEMENTS = 524288
output_path = Path(sys.argv[1])
worker_path = str(Path(ZLI).with_name("openzl-laya-worker"))
results = {
    "seed": SEED,
    "development_seed": 314159,
    "elements": ELEMENTS,
    "profile": "le-u64",
    "datasets": {},
}


def run(args):
    start = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return time.perf_counter() - start, result.stderr + result.stdout


def measure(source, compressed, flags):
    wall, log = run(
        [ZLI, "compress", str(source), "-o", str(compressed), "--force", *flags]
    )
    restored = compressed.with_suffix(".restored")
    decode_wall, decode_log = run(
        [ZLI, "decompress", str(compressed), "-o", str(restored), "--force"]
    )
    assert restored.read_bytes() == source.read_bytes()

    def speed(text):
        matches = re.findall(r"([0-9.]+) MB/s", text)
        return float(matches[-1]) if matches else None

    return {
        "size": compressed.stat().st_size,
        "encode_wall_s": wall,
        "decode_wall_s": decode_wall,
        "encode_MB_s": speed(log),
        "decode_MB_s": speed(decode_log),
    }


with tempfile.TemporaryDirectory(prefix="openzl-laya-eval-") as temp:
    root = Path(temp)
    rng = random.Random(SEED)
    datasets = {
        "monotonic": [n for n in range(ELEMENTS)],
        "categorical": [rng.randrange(32) for _ in range(ELEMENTS)],
        "run_heavy": [n // 1024 % 17 for n in range(ELEMENTS)],
        "random": [rng.getrandbits(64) for _ in range(ELEMENTS)],
    }
    datasets["mixed"] = (
        datasets["monotonic"][: ELEMENTS // 2] + datasets["random"][ELEMENTS // 2 :]
    )
    compressors = export_candidates(root)
    for name, values in datasets.items():
        source = root / (name + ".bin")
        source.write_bytes(struct.pack("<" + "Q" * ELEMENTS, *values))
        entries = {
            candidate: measure(
                source, root / (name + candidate + ".zl"), ["--compressor", str(path)]
            )
            for candidate, path in zip(CANDIDATES, compressors)
        }
        report = root / (name + ".json")
        with worker(lambda r: {**r, "truncated": True}):
            entries["sample_benchmark"] = measure(
                source,
                root / (name + "sample.zl"),
                ["--profile", "le-u64", "--laya", "--laya-report", str(report)],
            )
        entries["sample_benchmark"]["routing"] = json.loads(report.read_text())
        results["datasets"][name] = entries
    # The worker loads and warms its fixed bucket within the startup allowance.
    cold_report = root / "cold.json"
    results["cold_start"] = measure(
        root / "monotonic.bin",
        root / "cold.zl",
        ["--profile", "le-u64", "--laya", "--laya-report", str(cold_report)],
    )
    results["cold_start"]["decision_timeout_ms"] = 2000
    results["cold_start"]["routing"] = json.loads(cold_report.read_text())
    # Warm comparisons, all with the default decision allowance.
    for name in datasets:
        source = root / (name + ".bin")
        report = root / (name + "laya.json")
        entries = results["datasets"][name]
        entries["laya"] = measure(
            source,
            root / (name + "laya.zl"),
            ["--profile", "le-u64", "--laya", "--laya-report", str(report)],
        )
        entries["laya"]["routing"] = json.loads(report.read_text())
        minimum = min(entries[c]["size"] for c in CANDIDATES)
        for entry in entries.values():
            entry["size_regret_bytes"] = entry["size"] - minimum
            entry["size_regret_fraction"] = entry["size"] / minimum - 1
    # Repeat the first dataset warm so cold/warm are paired on identical input.
    results["warm_monotonic"] = measure(
        root / "monotonic.bin",
        root / "warm.zl",
        ["--profile", "le-u64", "--laya", "--laya-report", str(root / "warm.json")],
    )
    results["warm_monotonic"]["routing"] = json.loads((root / "warm.json").read_text())
    status = json.loads(subprocess.check_output([worker_path, "status"], text=True))
    results["worker_rss_kib"] = status["resident_bytes"] / 1024
    results["hardware"] = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
    ).strip()
    results["machine"] = subprocess.check_output(["uname", "-m"], text=True).strip()
    results["os"] = subprocess.check_output(
        ["sw_vers", "-productVersion"], text=True
    ).strip()
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(output_path)
