# Experimental local integer routing

Laya orders local trials of seven ordinary OpenZL compressors; measured sizes
select one compressor per file. The objective is compressed size. This is not policy learning or a guarantee of
minimum full-file size; confidence and sampling defaults are experimental.
No API key or hosted backend is used. Decoding needs no model or worker.

## Build and use

Ordinary builds keep this feature disabled. There is one C++ worker
(`cli/laya/worker.cpp`): the socket server, asset handling, tokenizer and
prompt construction are shared, and a model backend implements
`cli/laya/model.h` per platform:

- macOS: Apple Silicon, macOS 14+ and the Xcode command-line tools are
  required. The backend (`cli/laya/coreml/model.mm`, a thin Objective-C++
  file) runs the pinned Core ML conversion of the model; no Swift toolchain or
  package dependency is involved.
- Linux: an NVIDIA GPU, the CUDA toolkit (13.0 was used; `nvcc` and
  cuBLASLt) and CMake 3.24+ are required. The backend (`cli/laya/cuda/`) has
  its own safetensors loader and kernels and runs the pinned upstream
  checkpoint without Python or PyTorch.

`prepare` only downloads the assets.

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
tracks its compilation command configuration.

`prepare` is the only command that downloads anything. It uses system `curl`
and honors shell proxy settings. It verifies fixed SHA-256 hashes, stages the
complete download, and publishes it atomically. Startup verifies hashes, loads
local assets, and warms the model before reporting readiness. Missing/corrupt
assets produce a preparation hint and local sample benchmarking.

On macOS the assets are the tokenizer and e8 1024-token bucket of
`FluidInference/laya-coreml` at revision
`7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc`, approximately 487 MB, in
`~/Library/Application Support/OpenZL/Laya/<revision>`. Core ML is configured
with `.all`; this does not mean all operations execute on the Neural Engine.
Inputs are encoded as FluidUse 0.2.0 (the previous Swift worker) encoded
them, and the input buffers are reused across predictions. Reports carry
`compute_units: all`, `precision: e8`, `bucket: 1024` and `backend: coreml`.
Load takes about a second once Core ML has cached its compiled model; the
first prediction compiles device kernels, so a cold start is dominated by
that warmup (about 13 s on an M4).

On Linux the assets are the safetensors weights, configuration and tokenizer of
`convaiinnovations/laya-multilingual` at revision
`052592a15d198d9ad47da779604259b10b47b7aa`, approximately 680 MB, in
`$XDG_CACHE_HOME/openzl/laya/<revision>` (default `~/.cache/openzl/laya`). The
Core ML conversion above was made from these same weights, and the two
tokenizers are byte-identical, but the two workers are separate numerical
implementations. The checkpoint stores fp16 weights; the Linux worker keeps
them exact by running every GEMM on tensor cores with fp16 inputs and fp32
accumulation through cuBLASLt (the library PyTorch uses for its GEMMs, with
each shape's fastest heuristic candidate timed at load), and keeps
embeddings, layer norms, the residual stream, softmax and the decision heads
in fp32. Attention is one fused tensor-core kernel per layer (WMMA, online
softmax in fp32); rotary embeddings, GLU with exact GELU, layer norms and
activations are fused kernels. Prompts are padded to a multiple of
`OPENZL_LAYA_BUCKET` tokens (default 64, minimum 32) and each padded length
replays one CUDA graph; the real token count lives in device memory so a
graph serves its whole bucket. Startup verifies the asset hashes, loads the
weights (about 1.5 s) and records the graphs of the usual prompt lengths
before readiness. Reports carry the actual `compute_units` (for example
`cuda:NVIDIA GB10`), `precision`, `padded_tokens`, `device_ms` and
`backend: native-cuda`.

The tokenizer is a port of the HuggingFace `tokenizers` byte-fallback BPE
(Metaspace pre-tokenizer, added-token matching with lstrip/rstrip, ranked
merges with fused unknowns) and the prompt is built like upstream's
`build_sequence`; the statistics are serialized byte-for-byte like Python's
`json.dumps(sort_keys=True, separators=(",", ":"))`, including its float
layout. `openzl-laya-worker check` verifies all of this against fixtures
produced by the upstream PyTorch implementation
(`cli/tests/laya_tokenizer_reference.json`, 102 strings, and
`cli/tests/laya_reference.json`, 27 prompts with fp32 CPU probabilities);
CTest runs it as `laya_native_check` on both platforms and skips it when
assets are absent. On the GB10 the fixtures match on every string and prompt,
with probabilities within 0.0022 of the fp32 reference and identical
candidate orderings. The Core ML e8 conversion (int8 embeddings, fp16 encoder)
is checked on the decision: on an M4 every prompt selects the reference's
candidate, with probabilities within 0.021; near-tied low-probability
candidates may swap order (3 of 27 prompts).

The previous Swift worker serialized statistics with Foundation's
`JSONEncoder`, which writes `1.0` as `1`, so its prompts differed from the
reference and from the Linux worker; on the 26 fixture prompts within the
context limit it selected a different candidate than the reference on 6. The
C++ worker uses the same Python-compatible serialization on both platforms. On
unified-memory systems (for example DGX Spark) CUDA context creation fails
while other processes hold the memory, in which case the worker exits with a
CUDA error and the CLI benchmarks locally.

Int8 weight-only, int8 dynamic and fp8 dynamic quantization were measured
on an earlier PyTorch-based worker and rejected: the forward pass is
launch-bound at batch one, so they did not reduce latency, and they changed
the candidate ordering on 2 to 10 of 15 prompts. The native worker keeps the
checkpoint's fp16 weights.

## Options and selection

- `--laya-context TEXT`: optional domain context, at most 4096 UTF-8 bytes.
- `--laya-confidence NUMBER`: confidence threshold in [0, 1], default 0.8.
  Confidence is `1 - H(p)/log(7)` (as in FluidUse), not the winning probability.
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

Make also supports Linux: `make OPENZL_ENABLE_LAYA=1 zli` compiles the
worker with `nvcc` (override with `NVCC=`) next to `zli`.

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
FIFO consumer serializes inference through one loaded model. Timed-out
clients close their own sockets; they do not kill the shared worker.

Each connection carries one request/reply, framed as a four-byte big-endian
length and UTF-8 JSON (maximum 65536 bytes). Version 1 includes `id`, `command`,
`candidates`, `statistics`, `context`, and `deadline_ms`. Responses echo the
ID/version/revision and carry candidate probabilities, confidence, action
probability, truncation, timing, and the compute units, precision and context
bucket actually used. The revision is the platform's pinned asset revision
(`cli/utils/laya.h`), so a worker built for the other platform is rejected. The client validates identity, finite
normalized probabilities, selected maximum, and entropy confidence. The
worker's log is in the private runtime directory.

## Validation and evaluation protocol

```sh
ctest --test-dir build --output-on-failure   # includes laya_native_check
# Real worker lifecycle (needs prepared assets):
python3 cli/tests/laya_worker_tests.py build/cli/openzl-laya-worker
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

A rerun on Linux on an NVIDIA GB10 reproduced every byte count of both
suites. The native C++/CUDA worker answers warm requests in about 3.5–4.5 ms
end to end (3.2–3.9 ms on the device), cold-starts in about 5 s and holds
about 475 MiB resident, but the slower CPU makes probe-heavy conditions cost
more wall time (32 MiB monotonic: about 2.1 s). Quantizing below fp16 was
measured and rejected.

See the [sampling investigation](laya-routing-investigation.md) for the design
rationale and the machine-readable summaries
([Apple M4](laya-adaptive-evaluation.json),
[NVIDIA GB10](laya-adaptive-evaluation-gb10.json)) for exact sizes, wall-time
ranges, hashes and guard outcomes.
