#!/usr/bin/env python3
"""Compare sample scales against exhaustive full-file compression.

Usage: python3 cli/tests/laya_sampling_evaluate.py /absolute/path/to/zli output.json
Append --large-only for the follow-up 128 MiB stress cases.
Run with the real worker stopped. Uses exported, ordinary compressor graphs;
no model inference or downloads. CLI subprocess timings are diagnostic, not
estimates of an in-process routing implementation. Every frame is round-tripped.
"""

import array
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from laya_integration_tests import CANDIDATES, ZLI, export_candidates

OUTPUT = Path(sys.argv[1])
MIB = 1024 * 1024
SCALES = [65536, MIB, 4 * MIB, 16 * MIB]
RESULTS = {
    "existing_seed": 271828,
    "additional_seed": 161803,
    "profile": "le-u64",
    "sample_scales": SCALES,
    "binary_sha256": hashlib.sha256(Path(ZLI).read_bytes()).hexdigest(),
    "round_trips": 0,
    "datasets": {},
    "suite": "large" if "--large-only" in sys.argv else "initial",
    "limitations": "Single pass; subprocess timings are not in-process routing costs. "
    "Existing datasets are regression evidence, not a fresh holdout. "
    "Additional cases and scales fixed before execution; synthetic data only.",
}


def run(args):
    start = time.perf_counter()
    subprocess.run(args, check=True, capture_output=True, timeout=70)
    return (time.perf_counter() - start) * 1000


def encode(values):
    data = array.array("Q", values)
    assert data.itemsize == 8
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def datasets():
    for mib in (4, 32):
        count = mib * MIB // 8
        rng = random.Random(271828)
        yield f"{mib}m_monotonic", encode(range(count))
        yield f"{mib}m_categorical", encode(rng.randrange(32) for _ in range(count))
        yield f"{mib}m_run_heavy", encode(n // 1024 % 17 for n in range(count))
        noise = encode(rng.getrandbits(64) for _ in range(count))
        yield f"{mib}m_random", noise
        yield f"{mib}m_mixed", encode(range(count // 2)) + noise[len(noise) // 2 :]
    count = 32 * MIB // 8
    yield "new_odd_runs", encode(n // 1023 % 31 for n in range(count))
    yield "new_long_runs", encode(n // 65536 % 19 for n in range(count))
    rng = random.Random(161803)
    irregular = bytearray()
    while len(irregular) < 32 * MIB:
        length = rng.randrange(1, 8193)
        value = rng.randrange(257)
        irregular.extend(value.to_bytes(8, "little") * length)
    yield "new_irregular_runs", bytes(irregular[: 32 * MIB])
    # A distribution change challenges representativeness, not just periodicity.
    yield (
        "new_runs_then_random",
        encode(n // 2048 % 23 for n in range(count // 2))
        + encode(rng.getrandbits(64) for _ in range(count // 2)),
    )


def windows(data, scale):
    length = min(scale, len(data))
    end = 0
    for offset in (0, (len(data) - length) // 16 * 8, len(data) - length):
        if offset >= end:
            yield data[offset : offset + length]
            end = offset + length


def large_datasets():
    # Frozen after the initial experiment: 16 MiB windows must now predict
    # unseen data, rather than covering the complete 4/32 MiB file.
    count = 128 * MIB // 8
    yield "large_long_runs", encode(n // 65536 % 19 for n in range(count))
    rng = random.Random(141421)
    data = bytearray()
    while len(data) < 128 * MIB:
        length = rng.randrange(1, 16385)
        value = rng.randrange(513)
        data.extend(value.to_bytes(8, "little") * length)
    del data[128 * MIB :]
    yield "large_irregular_runs", bytes(data)
    # Deliberately adversarial placement, not a representative workload.
    # The 16 MiB sampler sees only regular runs, hiding the irregular regions.
    regular = encode(n // 1024 % 17 for n in range(16 * MIB // 8))
    for offset in (0, 56 * MIB, 112 * MIB):
        data[offset : offset + len(regular)] = regular
    yield "large_hidden_irregular", bytes(data)


with tempfile.TemporaryDirectory(prefix="openzl-sampling-") as temp:
    root = Path(temp)
    source = root / "input.bin"
    compressed = root / "output.zl"
    restored = root / "restored.bin"
    compressors = dict(zip(CANDIDATES, export_candidates(root)))

    def measure(data, candidate):
        source.write_bytes(data)
        elapsed = run(
            [
                ZLI,
                "compress",
                str(source),
                "--compressor",
                str(compressors[candidate]),
                "-o",
                str(compressed),
                "--force",
            ]
        )
        run([ZLI, "decompress", str(compressed), "-o", str(restored), "--force"])
        assert restored.read_bytes() == data
        RESULTS["round_trips"] += 1
        return {"bytes": compressed.stat().st_size, "encode_wall_ms": elapsed}

    for name, data in large_datasets() if "--large-only" in sys.argv else datasets():
        entry = {"input_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        entry["full"] = {
            candidate: measure(data, candidate) for candidate in CANDIDATES
        }
        entry["scales"] = {}
        for scale in SCALES:
            slices = list(windows(data, scale))
            scores = {}
            for candidate in CANDIDATES:
                measurements = [measure(part, candidate) for part in slices]
                scores[candidate] = {
                    "bytes": sum(m["bytes"] for m in measurements),
                    "encode_wall_ms": sum(m["encode_wall_ms"] for m in measurements),
                }
            selected = min(CANDIDATES, key=lambda c: scores[c]["bytes"])
            entry["scales"][str(scale)] = {
                "sample_sizes": [len(part) for part in slices],
                "scores": scores,
                "selected": selected,
                "selected_full_bytes": entry["full"][selected]["bytes"],
            }
        RESULTS["datasets"][name] = entry
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(RESULTS, indent=2) + "\n")
        print(name, {k: v["selected"] for k, v in entry["scales"].items()}, flush=True)
