# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Fresh-seed adaptive size validation; run with the real worker stopped.

Usage: python3 cli/tests/laya_adaptive_evaluate.py build/cli/zli output.json
Requires the CMake laya_fixture helper. Set OPENZL_LAYA_DISABLED_ZLI to verify
frames using a separate disabled build. Uses fake model rankings, not inference.
Seed and distributions were frozen before the first execution of this suite.
"""

import array
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from laya_integration_tests import (
    CANDIDATES,
    DISABLED_ZLI,
    ZLI,
    export_candidates,
    worker,
)

OUTPUT = Path(sys.argv[1])

N = 4194304
SEED = 223606
rng = random.Random(SEED)


def packed(values):
    data = array.array("Q", values)
    assert data.itemsize == 8
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def data_cases():
    yield "near_u64_max", packed((1 << 64) - 1 - n for n in range(N))
    yield "categorical_257", packed(rng.randrange(257) for _ in range(N))
    values = bytearray()
    while len(values) < N * 8:
        values.extend(
            rng.getrandbits(64).to_bytes(8, "little") * rng.randrange(1, 16385)
        )
    yield "irregular_full_width_runs", bytes(values[: N * 8])
    noise = rng.randbytes(N * 8)
    yield "random", noise
    yield "zeros_then_random", bytes(N * 4) + noise[N * 4 :]


def run(args):
    subprocess.run(args, check=True, capture_output=True, timeout=90)


results = {
    "seed": SEED,
    "binary_sha256": hashlib.sha256(Path(ZLI).read_bytes()).hexdigest(),
    "round_trips": 0,
    "datasets": {},
    "purpose": "Frozen follow-up size validation; no timing claims",
}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    paths = export_candidates(root)
    source, output, restored = [root / n for n in ("input", "output.zl", "restored")]
    report = root / "report.json"

    def measure(data, flags):
        run([ZLI, "compress", str(source), "-o", str(output), "--force", *flags])
        run([DISABLED_ZLI, "decompress", str(output), "-o", str(restored), "--force"])
        assert restored.read_bytes() == data
        results["round_trips"] += 1
        return output.stat().st_size

    for name, data in data_cases():
        source.write_bytes(data)
        entry = {"sha256": hashlib.sha256(data).hexdigest(), "candidates": {}}
        for candidate, path in zip(CANDIDATES, paths):
            entry["candidates"][candidate] = measure(data, ["--compressor", str(path)])
        for guarded in (False, True):
            with worker(lambda r: r):
                size = measure(
                    data,
                    ["--profile", "le-u64", "--laya", "--laya-report", str(report)]
                    + (["--laya-size-guard"] if guarded else []),
                )
            entry["guarded" if guarded else "adaptive"] = {
                "bytes": size,
                "report": json.loads(report.read_text()),
            }
            if guarded:
                assert size <= entry["candidates"]["numeric"]
        results["datasets"][name] = entry
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
        print(
            name,
            entry["adaptive"]["bytes"],
            entry["guarded"]["bytes"],
            min(entry["candidates"].values()),
            flush=True,
        )
