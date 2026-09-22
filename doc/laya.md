# Experimental local integer routing

Laya orders local trials of seven ordinary OpenZL compressors; measured sizes
select one compressor per file. The objective is compressed size. This is not policy learning or a guarantee of
minimum full-file size; confidence and sampling defaults are experimental.
No API key or hosted backend is used. Decoding needs no model or Swift runtime.

## Build and use

Apple Silicon and macOS 14+ are required for the optional worker. Swift 6 is
required to build FluidUse. Ordinary builds keep this feature disabled.

```sh
cmake -S . -B build -DOPENZL_BUILD_CLI=ON \
  -DOPENZL_ENABLE_LAYA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/cli/openzl-laya-worker prepare
build/cli/zli compress input.bin --profile le-u64 --laya -o output.zl \
  --laya-size-guard --laya-report report.json \
  --laya-save-compressor selected.compressor
build/cli/openzl-laya-worker status
build/cli/openzl-laya-worker stop
```

Make also supports `make OPENZL_ENABLE_LAYA=1 zli`, placing the worker next to
`zli`. `make OPENZL_ENABLE_LAYA=1 install-cli PREFIX=/path` installs both.
CMake install installs both executables when `OPENZL_INSTALL=ON`.
Rebuild when changing the feature flag; CMake tracks definitions and Make
tracks its compilation command configuration. FluidUse is pinned to 0.2.0;
`cli/laya/Package.resolved` pins its transitive dependencies.

`prepare` is the only command that downloads assets. It uses macOS system
`curl` and honors shell proxy settings. It verifies fixed SHA-256
hashes, stages the complete download, and publishes it in
`~/Library/Application Support/OpenZL/Laya/<revision>`. The pinned revision is
`7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc`. The tokenizer and e8 1024-token
bucket occupy approximately 487 MB. Startup verifies hashes, loads local assets, and warms the fixed prediction
bucket before reporting readiness. Missing/corrupt assets produce a preparation hint and local
sample benchmarking. Core ML is configured with `.all`; this does not mean
all operations execute on the Neural Engine.

## Options and selection

- `--laya-context TEXT`: optional domain context, at most 4096 UTF-8 bytes.
- `--laya-confidence NUMBER`: confidence threshold in [0, 1], default 0.8.
  Confidence is FluidUse's `1 - H(p)/log(7)`, not the winning probability.
  The threshold labels low confidence in diagnostics; it no longer bypasses trials.
- `--laya-timeout-ms INTEGER`: decision deadline including queue time, default
  2000, range 1–60000. Cold startup has a separate 60-second allowance.
- `--laya-report PATH`: JSON diagnostics, statistics, probabilities, action
  probability, truncation, fallback reason, sample output sizes, startup,
  inference, routing and compression times, and final bytes.
- `--laya-size-guard`: compress the full file with numeric as well, keeping the
  smaller output (numeric wins ties). This prevents size regression against
  numeric at the cost of another compression and output buffer. It does not
  guarantee the best of all seven candidates. Traces and exports match the
  retained output; `compression_ms` includes both compressions.
- `--laya-save-compressor PATH`: ordinary serialized compressor for reuse with
  `zli compress --compressor PATH`, including with Laya disabled.

All existing integer widths, signedness, byte orders, levels and chunk sizes
are supported. Custom compressors, dictionaries, training and profile
arguments cannot be combined with routing. Sidecars must have distinct paths;
existing outputs require `--force`.

Three non-overlapping, element-aligned windows of up to 64 KiB are sampled at
the beginning, middle and end. Only aggregate statistics, serialized in stable key order, and optional context
are sent to the local worker. Files smaller than 1 MiB keep the existing
numeric profile and do not start the worker. Empty/misaligned inputs retain
normal numeric-profile behavior.

The stable candidates are `numeric`, `fieldlz`, `range_fieldlz`, `range_zstd`,
`delta_fieldlz`, `tokenize` (sorted alphabet compressed with delta + FieldLZ;
indices with FieldLZ), and `zstd`. Endian conversion and segmentation are
shared with numeric profiles. Laya orders trials by probability; every candidate
is benchmarked regardless of confidence. Invalid/truncated responses, deadlines
and worker failures use the stable candidate order. Action probability remains
diagnostic. Candidates must round-trip every probe exactly; failures exclude
that candidate for the file. Numeric wins size ties, then stable candidate order.

Routing starts with at most three 64 KiB prefixes of the first, middle, and last
actual chunks (duplicate chunks are omitted). At >=128x sample compression or
mean sampled run length >=32 elements, probes grow to 1 MiB and then to the
smaller of the actual chunk size and 16 MiB. A changed winner between scales
also triggers growth. All surviving candidates are reconsidered at each scale.
These thresholds were frozen before the implementation benchmark, but are still
experimental. Files with chunks larger than the cap can hide important data;
`--laya-size-guard` is the full-output safeguard against numeric regression.

Reports retain `probe_stages` with offsets, sizes, per-candidate outputs, winners,
and elapsed time, plus `trial_order`, the probe cap, and optional `size_guard`
results. `sample_sizes` describes the original statistics windows;
`sample_output_bytes` describes the final benchmarking stage. The byte budget
bounds trial input, not wall time: up to seven candidates times three windows
at each of 64 KiB, 1 MiB and 16 MiB. Highly compressible files can therefore spend
substantially more time routing. Only aggregate statistics reach the worker,
including file/chunk sizes, level, window offsets and nonzero delta summaries.

The size guard makes the final file-level selection after measuring both full
outputs; compressor export is deferred until that selection. Without the guard,
sampling cannot guarantee no regression or the smallest full-file output.

## Worker protocol and lifecycle

The worker listens on `/tmp/openzl-laya-<uid>/worker.sock` inside an
owner-checked 0700 directory, with a 0600 socket and advisory process lock.
The lock owner alone removes stale sockets. `start` waits for readiness;
`status` reports the PID and resident memory; `stop` drains admitted work before exit (waiting up to 60 seconds). The worker
exits after ten idle minutes. At most eight connections are admitted; excess
connections close immediately and the CLI falls back locally. One dedicated
FIFO consumer serializes inference through one loaded LayaManager. Timed-out
clients close their own sockets; they do not kill the shared worker.

Each connection carries one request/reply, framed as a four-byte big-endian
length and UTF-8 JSON (maximum 65536 bytes). Version 1 includes `id`, `command`,
`candidates`, `statistics`, `context`, and `deadline_ms`. Responses echo the
ID/version/revision and carry candidate probabilities, confidence, action
probability, truncation and timing. The client validates identity, finite
normalized probabilities, selected maximum, and entropy confidence. The
worker's log is in the private runtime directory.

## Validation and evaluation protocol

```sh
swift test --package-path cli/laya
ctest --test-dir build --output-on-failure
# Stop the worker before the fake-socket integration tests:
build/cli/openzl-laya-worker stop
python3 cli/tests/laya_integration_tests.py build/cli/zli
```

Evaluation is frozen before execution: development seed 314159, evaluation
seed 271828, 4 MiB little-endian unsigned 64-bit columns, monotonic,
categorical, run-heavy, random and mixed distributions. Compare numeric,
Laya routing, sample benchmarking, and all seven full-file candidate sizes.
Report size regret relative to the exhaustive minimum, encode/decode times,
and total process wall time. These synthetic datasets are not representative
production evidence. They became regression inputs during development; a
separate fresh-seed suite evaluates the frozen adaptive thresholds. No improvement is
assumed. Dataset-wide learning, ACE search, CSV/Parquet routing and arbitrary
graph generation are deferred.

## Results and limitations

The [benchmark report](laya-benchmarks.md) records both policies, 4/32 MiB
compression ratios and wall times, codec throughput, worker startup and memory,
fresh-seed validation, reproduction steps, and retained evidence.

Adaptive routing reduced the 32 MiB run-heavy output from 13,211 to 278 bytes,
but increased median wall time from the earlier 250 to 409 ms. It matched the
best of seven on the ten original regression conditions; fresh-seed tests
still exposed misses. The optional size guard corrected a regression against
numeric, but cannot guarantee the exhaustive minimum. Laya currently orders
all trials without eliminating any, so no speed benefit is established.

See the [sampling investigation](laya-routing-investigation.md) for the design
rationale and the [machine-readable summary](laya-adaptive-evaluation.json) for
exact sizes, wall-time ranges, hashes and guard outcomes.
