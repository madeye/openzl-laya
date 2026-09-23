# Laya routing benchmark results

Measured on 2026-09-22, Apple M4, 32 GiB RAM, macOS 26.5.2, release builds.
Implementation: `b13d68e`; original policy: `f7cf666`. A rerun of the same
suites on Linux on an NVIDIA GB10, comparing two PyTorch workers with the
shipped native C++/CUDA worker and covering rejected quantization variants,
is reported in [Linux rerun](#linux-rerun-nvidia-gb10) below.

Adaptive routing improves compression on the original synthetic suite, but
larger probes can substantially increase total encode time. Laya currently
orders all seven trials without eliminating any, so its inference adds cost
without changing the winning size when every candidate succeeds. No general
speed advantage is established.

See the [usage guide](laya.md), [investigation](laya-routing-investigation.md),
and committed [measurement summary](laya-adaptive-evaluation.json).

## Method and interpretation

- Inputs: `le-u64`, level 6, default 16 MiB chunks, 4 and 32 MiB files.
- Original policy: five repetitions per size; adaptive policy: three. These
  runs are not paired, and OS caches remain warm between operations.
- Seed 271828: monotonic, categorical (32 values), runs of 1,024 repeated
  integers cycling through 17 values, random u64, and half-monotonic/half-random.
  These ten conditions are regression evidence used during development.
- Each repetition measures all seven full-file candidates, sample-only routing,
  cold startup, warm Laya routing, and a warm repeat. Adaptive evaluation checked
  282 frames by exact decompression; all ten conditions matched the exhaustive
  minimum in all three repetitions. The size guard was **off** in these timings.
- Networking was denied with `sandbox-exec` after model preparation. FluidUse
  0.2.0 loaded pinned Laya e8/1024 assets locally using Core ML `.all`.
- Compression ratio is input bytes / output bytes; higher is better. Random
  outputs slightly expand from framing overhead. MB/s uses decimal megabytes.
- Wall time includes process startup, input/output and routing. Codec throughput
  excludes routing and is not interchangeable with end-to-end throughput.
  Medians are descriptive; three repetitions do not establish confidence intervals.

## Compressed sizes

Byte counts were identical across repetitions. The final column is the adaptive
compression ratio, rounded; exact byte counts are the authoritative comparison.

| Input | Dataset | Numeric bytes | Original Laya bytes | Adaptive bytes | Adaptive ratio |
|---|---|---:|---:|---:|---:|
| 4 MiB | monotonic | 300 | 300 | 85 | 49,344.75× |
| 4 MiB | categorical | 445,633 | 327,879 | 327,824 | 12.79× |
| 4 MiB | run heavy | 944 | 1,427 | 145 | 28,926.23× |
| 4 MiB | random | 4,194,330 | 4,194,330 | 4,194,330 | 1.00× |
| 4 MiB | mixed | 2,097,552 | 2,097,552 | 2,097,260 | 2.00× |
| 32 MiB | monotonic | 1,276 | 1,276 | 158 | 212,369.82× |
| 32 MiB | categorical | 3,564,563 | 2,628,066 | 2,627,956 | 12.77× |
| 32 MiB | run heavy | 6,486 | 13,211 | 278 | 120,699.40× |
| 32 MiB | random | 33,554,472 | 33,554,472 | 33,554,472 | 1.00× |
| 32 MiB | mixed | 16,777,874 | 16,777,874 | 16,777,315 | 2.00× |

The 32 MiB run-heavy progression is **6,486 bytes numeric → 13,211 bytes with
original Laya routing → 278 bytes with adaptive routing**. Its ratio progresses
from 5,173× to 2,540× to 120,699×. The original model preferred numeric, but its
low-confidence fallback picked tokenization on unrepresentative small samples.
Broader trials and chunk-sized probes select delta → FieldLZ instead.

## End-to-end encode time

Milliseconds, median across repetitions. Sample-only uses a local fake worker
to trigger deterministic-order adaptive benchmarking; it includes that small
local IPC cost but performs no model inference. It is an evaluation mode, not
a separate production CLI flag. Earlier original-policy timings are provided
as context, not a controlled paired speed comparison.

| Input | Dataset | Numeric ms | Original Laya ms | Adaptive ms | Adaptive sample-only ms |
|---|---|---:|---:|---:|---:|
| 4 MiB | monotonic | 33 | 166 | 292 | 184 |
| 4 MiB | categorical | 40 | 146 | 142 | 22 |
| 4 MiB | run heavy | 21 | 147 | 165 | 46 |
| 4 MiB | random | 32 | 167 | 155 | 37 |
| 4 MiB | mixed | 35 | 174 | 333 | 211 |
| 32 MiB | monotonic | 288 | 441 | 1,489 | 1,406 |
| 32 MiB | categorical | 255 | 291 | 253 | 130 |
| 32 MiB | run heavy | 132 | 250 | 409 | 299 |
| 32 MiB | random | 269 | 400 | 394 | 276 |
| 32 MiB | mixed | 255 | 421 | 339 | 197 |

For run-heavy 32 MiB data, the smaller output costs 409 ms versus the earlier
250 ms. Monotonic 32 MiB data takes about 1.49 seconds because the router tries
all candidates on large probes. Better size does not imply better encode time.
Timing ranges and binary hashes are retained in the measurement summary. Its
notes identify a later size-guard graph-lifetime fix; fresh validation used the
final binary. The default-routing timing run did not enable that guard.

## Codec throughput and worker cost

The following 32 MiB medians come from the adaptive run logs, in MB/s. Decode
throughput measures the selected frame's codec, not model or worker execution.

| Dataset | Numeric encode | Adaptive selected encode | Numeric decode | Adaptive selected decode |
|---|---:|---:|---:|---:|
| monotonic | 197.2 | 5,480.1 | 1,263.8 | 3,256.1 |
| categorical | 228.9 | 2,028.0 | 1,192.0 | 4,484.4 |
| run heavy | 1,353.5 | 5,322.4 | 3,343.5 | 4,113.2 |
| random | 260.4 | 256.1 | 8,886.3 | 8,719.3 |
| mixed | 251.5 | 462.2 | 1,579.6 | 4,949.4 |

In the adaptive runs, median startup was 13.82 s (4 MiB suite) and 13.59 s
(32 MiB suite), warm inference 110.66 and 111.34 ms, and worker resident memory
630.86 and 634.66 MiB. Cold means a newly started worker with hash checks, model
load and warmup, not a machine with cold filesystem caches. Earlier five-run
measurements observed 16.6–17.3 s startup and about 114 ms warm inference;
startup observations vary and are not an SLA. No warm adaptive response was
truncated. `.all` does not establish that every operation ran on the Neural Engine.

## Fresh-seed validation and the size guard

After freezing the adaptive thresholds, seed 223606 generated five additional
32 MiB distributions. This suite uses deterministic fake model rankings and
makes no latency claim. It checked 45 frames using a Laya-disabled decoder;
a repeat through the committed script reproduced identical inputs and sizes.

| Dataset | Numeric bytes | Adaptive bytes | Guarded bytes | Best of seven bytes |
|---|---:|---:|---:|---:|
| near u64 max | 1,276 | 158 | 158 | 158 |
| categorical 257 | 5,432,580 | 4,807,721 | 4,807,721 | 4,721,606 |
| irregular full width runs | 5,395 | 5,395 | 5,395 | 5,395 |
| random | 33,554,472 | 33,554,472 | 33,554,472 | 33,554,472 |
| zeros then random | 16,777,301 | 16,777,791 | 16,777,301 | 16,777,301 |

Adaptive sampling alone matched the best size in three of five cases; the guard
matched four and never exceeded numeric. The 257-category case still has 1.82%
size regret against the exhaustive winner. The guard guarantees comparison
against numeric, not optimality among all seven, and costs another full
compression and an output buffer when a non-numeric candidate is proposed.

## Linux rerun (NVIDIA GB10)

Measured on 2026-09-22 and 2026-09-23 on a DGX Spark class machine: NVIDIA
GB10 (aarch64, 121.7 GiB unified memory), Linux 6.11, CUDA 13.0, release
build of the Linux port. Method as above: seed 271828, `le-u64`, level 6,
16 MiB chunks, three repetitions per size, size guard off, all 282 frames
decompressed and compared. The
[machine-readable summary](laya-adaptive-evaluation-gb10.json) retains ranges,
hashes and per-candidate sizes. An unrelated vLLM server held tens of GiB of
the unified memory during the runs.

**Every byte count equalled the Apple M4 adaptive run**, for numeric, adaptive
routing, sample-only routing and the best of seven, on all ten conditions, and
the same candidate won each condition, with each of the three Linux workers
described below. Model confidence stayed between 0.39 and 0.60 on all
conditions, so every report carried the diagnostic `low_confidence` label. On
the 32 MiB run-heavy input the model still preferred numeric (about 54%) over
tokenize (about 33%), giving delta + FieldLZ 1.3%; the probes selected it
regardless.

Three workers were measured on the same `zli` binary. The first two ran the
checkpoint through PyTorch: eager fp32 weights under bfloat16 autocast, then
bfloat16 `Linear` weights with 64-token padding buckets and
`torch.compile` reduce-overhead CUDA graphs. The shipped worker is native
C++/CUDA (see the [usage guide](laya.md)): fp16 tensor-core GEMMs through
cuBLASLt with fp32 accumulation and per-shape autotuning, a fused WMMA
attention kernel, fused layer-norm/rotary/GLU kernels, and one CUDA graph per
padding bucket, with no Python or PyTorch runtime. Profiling the eager
PyTorch form showed about 2,000 kernel launches per answer and most GPU time
spent re-casting weights; the native forward pass at 320 padded tokens is
about 340 kernels in a single graph launch and 3.2 ms of device time, of
which about half is cuBLASLt GEMMs and a quarter attention. Int8 weight-only,
int8 dynamic-activation and fp8 dynamic quantization were measured on the
PyTorch worker (torchao 0.18) and rejected: none reduced latency in this
launch-bound regime, and they changed the candidate ordering on 2, 10 and 8
of 15 benchmark prompts. Against a CPU fp32 run of the upstream
implementation, the native worker's probabilities differ by at most 0.0022
on 27 prompts with identical orderings, and its tokenizer matches on all 102
fixture strings; the compiled PyTorch path differed by up to 0.022.

| Worker medians (4 MiB / 32 MiB suite) | PyTorch eager | PyTorch compiled | Native C++/CUDA |
|---|---:|---:|---:|
| Cold startup | 3.9 s / 3.9 s | 14.6 s / 12.1 s | 5.2 s / 5.1 s |
| First inference after startup | 52 ms / 63 ms | 5.4 ms / 6.1 ms | 3.9 ms / 3.9 ms |
| Warm inference (routing report) | 13.3 ms / 15.1 ms | 6.6 ms / 6.7 ms | 4.5 ms / 4.5 ms |
| Resident memory | 2,359 MiB | 1,606 MiB | 475 MiB |

The native worker's cold start is hash verification of the 640 MB weights
(about 2 s), weight load (about 1.5 s) and graph recording for five buckets;
`prepare` no longer installs anything. Its warm inference in the routing
reports includes tokenization and host-device transfers; a direct socket
round trip measured 3.5–4.5 ms and the device time 3.2–3.9 ms.

Milliseconds, median of three repetitions; codec throughput in MB/s from the
native run.

| Input | Dataset | Adaptive bytes | Numeric ms | Adaptive ms, PyTorch eager | Adaptive ms, PyTorch compiled | Adaptive ms, native | Adaptive sample-only ms, native |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 MiB | monotonic | 85 | 45 | 305 | 296 | 279 | 287 |
| 4 MiB | categorical | 327,824 | 54 | 32 | 24 | 24 | 18 |
| 4 MiB | run heavy | 145 | 17 | 77 | 75 | 68 | 63 |
| 4 MiB | random | 4,194,330 | 36 | 75 | 50 | 41 | 49 |
| 4 MiB | mixed | 2,097,260 | 49 | 334 | 318 | 285 | 304 |
| 32 MiB | monotonic | 158 | 333 | 2,191 | 2,130 | 2,019 | 2,050 |
| 32 MiB | categorical | 2,627,956 | 322 | 116 | 88 | 88 | 85 |
| 32 MiB | run heavy | 278 | 114 | 715 | 740 | 715 | 662 |
| 32 MiB | random | 33,554,472 | 285 | 313 | 310 | 313 | 296 |
| 32 MiB | mixed | 16,777,315 | 311 | 230 | 223 | 223 | 206 |

| Dataset (32 MiB) | Numeric encode | Adaptive selected encode | Numeric decode | Adaptive selected decode |
|---|---:|---:|---:|---:|
| monotonic | 121.9 | 897.0 | 778.2 | 804.5 |
| categorical | 126.0 | 1,760.6 | 804.8 | 1,564.9 |
| run heavy | 530.7 | 645.1 | 1,217.5 | 1,093.9 |
| random | 166.9 | 152.2 | 3,331.6 | 3,266.5 |
| mixed | 136.3 | 235.6 | 1,020.2 | 1,513.5 |

Model inference is now a small part of routing: the probes dominate, and the
GB10's CPU compresses more slowly than the M4 in this suite (numeric 32 MiB
encode about 122 versus 197 MB/s on monotonic data), so the probe-heavy
conditions cost more wall time here than on the M4 even though inference is
about 25 times faster than the M4 worker's. Routing on the 32 MiB monotonic
input took about 2.0 s of the 2.0 s total. Faster inference shows only where
probing is cheap, for example 4 MiB random (75 to 41 ms) and 32 MiB
categorical (116 to 88 ms). The timings are single-machine descriptive
medians with the same caveats as above. One PyTorch-era repetition had to be
discarded because memory pressure from the vLLM servers made CUDA
initialization fail and that worker fell back to the CPU; the native worker
has no CPU path and reports the CUDA error instead.

The fresh-seed suite (seed 223606, fake model rankings, decoded with a
Laya-disabled build, 45 frames) reproduced the Apple results exactly: adaptive
routing matched the best of seven on three of five cases, the guard on four, and
the guard never exceeded numeric. `categorical_257` kept its 1.82% regret
(4,807,721 versus 4,721,606 bytes) and `zeros_then_random` was again corrected
from 16,777,791 to 16,777,301 bytes by the guard.

## macOS C++ worker (replacing the Swift worker)

The macOS sections above were measured with the earlier Swift/FluidUse
worker. Its replacement is the shared C++ worker with a Core ML backend, which
loads the same pinned e8/1024 bundle with `.all`. On an Apple M4 (macOS 27,
the 26 fixture prompts of `cli/tests/laya_reference.json` within the
4096-byte context limit, sent over the real socket):

| Worker | Warm inference p50 | Startup (cached Core ML model) | Resident memory | Executable |
|---|---:|---:|---:|---:|
| Swift (FluidUse 0.2.0) | 110.7 ms | 13.8 s (median above) | 692 MB | 18 MB |
| C++ (Core ML backend) | 110.4 ms | 12.8–15.1 s | 637–641 MB | 0.5 MB |

Inference time is unchanged: both run the same Core ML graph, and the
prediction dominates. Startup is dominated by the first prediction compiling
device kernels (about 13 s). The first start of a newly built worker took
28.6 s, because Core ML caches the compiled model per executable. SHA-256
asset verification uses CommonCrypto (0.5 s for the weights, against 1.8 s for
the portable implementation). Against the PyTorch reference, the C++ worker
selects the reference's candidate on 26 of 26 prompts (max |dp| 0.021). The
Swift worker matched on 20 of 26 (max |dp| 0.576), because `JSONEncoder`
serialized statistics differently from the reference, for example `1.0` as
`1`.

## Reproduction and retained evidence

Build the CLI, worker and test-only candidate exporter:

```sh
cmake -S . -B build-laya -DCMAKE_BUILD_TYPE=Release \
  -DOPENZL_BUILD_CLI=ON -DOPENZL_BUILD_TESTS=ON -DOPENZL_ENABLE_LAYA=ON
cmake --build build-laya -j
build-laya/cli/openzl-laya-worker prepare
# Stop a running worker before each evaluator; they use isolated fake sockets.
build-laya/cli/openzl-laya-worker stop
python3 cli/tests/laya_evaluate.py build-laya/cli/zli evaluation-4m.json
# The preceding evaluator leaves a real worker running.
build-laya/cli/openzl-laya-worker stop
OPENZL_LAYA_DISABLED_ZLI=/path/to/disabled/zli \
  python3 cli/tests/laya_adaptive_evaluate.py build-laya/cli/zli fresh.json
python3 cli/tests/laya_sampling_evaluate.py build-laya/cli/zli sampling.json
python3 cli/tests/laya_sampling_evaluate.py build-laya/cli/zli large.json --large-only
```

`stop` reports an error if no worker is running; in that case proceed with the
evaluator. The evaluator defaults to 524288 elements (4 MiB); the recorded
32 MiB runs pass `--elements 4194304`. `OPENZL_LAYA_FIXTURE` can override the
CMake helper path. To repeat the offline condition on macOS, run the evaluator
under a sandbox profile containing
`(version 1) (allow default) (deny network-outbound (remote ip "*:*"))` after
preparation. On Linux the same commands apply; `prepare` also installs the
worker's PyTorch environment, and `OPENZL_LAYA_DEVICE=cpu` selects CPU
inference. Run evaluations without concurrent builds or other benchmarks.

Committed evidence: `laya-adaptive-evaluation.json` contains adaptive wall-time
ranges, byte counts, binary hashes and fresh-seed results from the Apple M4;
`laya-adaptive-evaluation-gb10.json` contains the same for the Linux rerun;
`laya-evaluation.json` retains the original single-run 4 MiB evaluation. Raw
repeated measurements and runner copies are workspace artifacts, not tracked
repository files:

- `build/benchmarks/laya-20260922/`: original five-run suites and methodology.
- `build/benchmarks/laya-improvements/`: four-scale sampling investigation.
- `build/benchmarks/laya-adaptive/`: adaptive three-run suites, fresh-seed runs,
  codec throughput, worker timings, and validation logs.
- `build-laya/benchmarks/laya-gb10/` (PyTorch eager), `laya-gb10-graphs/`
  (PyTorch compiled) and `laya-gb10-native/` (native C++/CUDA): Linux
  reruns; `run_benchmarks.py` there calls the evaluator three times per size
  and aggregates the medians.

Validation at implementation completion passed 2,647 CTest cases, 13 routing
integration tests, 8 Swift tests (since replaced by the C++ worker), 10 CLI format tests, C++/Python formatting,
Make/CMake enabled and disabled builds, and 22 general CLI tests on disabled
builds. The Linux port was validated with the real-worker lifecycle tests, the
13 routing integration tests, the Laya unit tests, the native parity check,
2,018 CTest cases, and Make/CMake enabled builds on the GB10 machine. The existing ACE format-version training test failed once during
concurrent compilation, then passed its isolated rerun and the final full suite.

These synthetic results do not establish production performance, calibrated
model confidence, or optimal sampling thresholds. The original regression
suite informed development, and the fresh suite is small. Reuse exported
compressors only where the workload is sufficiently similar; their quality on
new files is not guaranteed by the measurements above.
