# Local Laya integer routing

This experimental option uses local Laya inference to order trials of seven
ordinary OpenZL compressors. Actual sample sizes select one compressor for
the file. No hosted service or API key is required, and decoding needs no model.

## Build and run

Apple Silicon, macOS 14+ and Swift 6 are required. The feature is off by default.

```sh
cmake -S . -B build-laya -DCMAKE_BUILD_TYPE=Release \
  -DOPENZL_BUILD_CLI=ON -DOPENZL_ENABLE_LAYA=ON
cmake --build build-laya -j
build-laya/cli/openzl-laya-worker prepare
build-laya/cli/zli compress input.bin --profile le-u64 --laya \
  --laya-size-guard --laya-report report.json \
  --laya-save-compressor selected.compressor -o output.zl
```

`prepare` downloads and verifies the pinned model assets. Compression loads
local assets only. Missing assets or failed inference trigger local benchmarking.
The persistent worker starts on demand and exits after ten idle minutes;
`openzl-laya-worker start`, `status`, and `stop` control it explicitly.

## Selection and reuse

- Integer widths, signedness, byte orders, levels and chunk sizes are supported.
  Custom compressors, inline training, dictionary bundles and non-integer
  profiles cannot be combined with routing.
- Files smaller than 1 MiB keep the numeric profile without starting the worker.
- All seven candidates are tested, even for confident model predictions.
  Probes begin at 64 KiB at chunk boundaries and can grow to 1 MiB and then
  one chunk, capped at 16 MiB, for compressible data, long runs or changed winners.
- `--laya-size-guard` compares the proposed full output against numeric and
  retains the smaller frame, preferring numeric on ties. This costs another
  compression when needed; it does not guarantee the best of all candidates.
- `--laya-confidence` is diagnostic; it no longer bypasses measured trials.
  `--laya-timeout-ms` bounds the worker decision, not the local benchmarking.
- Reports show model probabilities, trial order, probe sizes and timings,
  final selection and size-guard outcomes. Traces and exports match the retained
  frame, including when the guard changes the selection.

An exported compressor can be reused with a Laya-disabled CLI:

```sh
zli compress another.bin --compressor selected.compressor -o another.zl
zli decompress another.zl -o restored.bin
```

## Measured tradeoffs

On Apple M4, three repetitions of the original 4/32 MiB synthetic suite matched
the smallest of seven full-file candidates in all ten conditions. These are
development regression datasets, not evidence of general optimality. The 32 MiB
run-heavy file improved from 13,211 to 278 bytes, but median wall time increased
from the earlier 250 to 409 ms. Monotonic 32 MiB routing took about 1.49 seconds.

Fresh-seed tests still missed the best candidate on two of five cases. The
optional guard corrected one regression against numeric; it was never larger
than numeric on those five cases. A 257-category file remained 1.82% above the
exhaustive minimum. Laya currently orders all trials without saving any, so its
inference adds latency without changing the winning size when all trials succeed.

The repository's `doc/laya.md` is the full option/protocol guide.
`doc/laya-benchmarks.md` records ratios, byte counts, timing methodology,
reproduction commands and limitations; `doc/laya-adaptive-evaluation.json`
retains the numerical summary. The original and adaptive measurements were
not paired timing runs, and no general speed improvement is claimed.
