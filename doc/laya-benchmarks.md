# Laya routing benchmark results

Measured on 2026-09-22, Apple M4, 32 GiB RAM, macOS 26.5.2, release builds.
Implementation: `b13d68e`; original policy: `f7cf666`.

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
evaluator. The standard evaluator fixes `ELEMENTS = 524288` (4 MiB). The recorded
32 MiB runs used a copy with only `ELEMENTS = 4194304` changed; that input size is
not a CLI option. `OPENZL_LAYA_FIXTURE` can override the CMake helper path. To
repeat the offline condition on macOS, run the evaluator under a sandbox profile
containing `(version 1) (allow default) (deny network-outbound (remote ip "*:*"))`
after preparation. Run evaluations without concurrent builds or other benchmarks.

Committed evidence: `laya-adaptive-evaluation.json` contains adaptive wall-time
ranges, byte counts, binary hashes and fresh-seed results; `laya-evaluation.json`
retains the original single-run 4 MiB evaluation. Raw repeated measurements and
runner copies are workspace artifacts, not tracked repository files:

- `build/benchmarks/laya-20260922/`: original five-run suites and methodology.
- `build/benchmarks/laya-improvements/`: four-scale sampling investigation.
- `build/benchmarks/laya-adaptive/`: adaptive three-run suites, fresh-seed runs,
  codec throughput, worker timings, and validation logs.

Validation at implementation completion passed 2,647 CTest cases, 13 routing
integration tests, 8 Swift tests, 10 CLI format tests, C++/Python formatting,
Make/CMake enabled and disabled builds, and 22 general CLI tests on disabled
builds. The existing ACE format-version training test failed once during
concurrent compilation, then passed its isolated rerun and the final full suite.

These synthetic results do not establish production performance, calibrated
model confidence, or optimal sampling thresholds. The original regression
suite informed development, and the fresh suite is small. Reuse exported
compressors only where the workload is sufficiently similar; their quality on
new files is not guaranteed by the measurements above.
