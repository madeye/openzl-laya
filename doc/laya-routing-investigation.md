# Routing improvement investigation (2026-09-22)

This is the historical investigation preceding adaptive routing. The resulting
implementation and its later measurements are documented in the
[usage guide](laya.md) and [benchmark report](laya-benchmarks.md).

The primary problem is the selection policy, not a demonstrated protocol or
candidate-mapping bug. On the original 32 MiB run-heavy input, Laya preferred
numeric (75.34%) over tokenize (19.12%). Its entropy confidence was 0.603, so
the local fallback compared only those two candidates. Tokenize won the small
samples, 363 versus 483 bytes, but lost on the full file, 13,211 versus 6,486
bytes. Delta + FieldLZ, the full-file winner at 278 bytes, had probability
0.35% and never entered that local comparison.

## Experiments

Added `cli/tests/laya_sampling_evaluate.py`. It exports the seven existing
compressors using deterministic fake-worker responses, then measures them
without model inference. All measurements use the unchanged release binary,
little-endian u64, default level and default 16 MiB chunks. Every measured
frame is decompressed and compared byte for byte. Numeric wins sample ties.

The initial suite fixes four window sizes (64 KiB, 1 MiB, 4 MiB, 16 MiB) and
14 cases before execution: the existing five distributions at 4 and 32 MiB,
plus four 32 MiB cases with odd-length runs, long runs, irregular runs, and
runs followed by random data. Existing seed: 271828; additional seed: 161803.
These existing datasets are regression evidence, not a new evaluation set.

A follow-up suite uses three 128 MiB files: long runs, irregular runs with
seed 141421, and deliberately placed regular regions surrounded by irregular
data. This checks partial sampling: three 16 MiB windows cover only 48 MiB.
The follow-up cases were specified after seeing the initial suite; they are
stress tests, not a preregistered independent validation of a final policy.

Both suites ran under the existing outbound-IP-denying sandbox, on Apple M4.
They completed **1,344 exact frame round trips**. Each condition ran once;
this experiment establishes sizes, not statistically reliable latency gains.
Subprocess timing includes a separate CLI launch for each sample/candidate
and must not be presented as the cost of an in-process routing implementation.

Raw results are retained in the workspace:

- `build/benchmarks/laya-improvements/sampling.json`
- `build/benchmarks/laya-improvements/large.json`

Each file records input hashes, binary hash, candidate sizes, selected sizes,
sample sizes, subprocess timings, seeds, and validated frame count.

Reproduce with the real worker stopped:

```sh
python3 cli/tests/laya_sampling_evaluate.py build-laya/cli/zli sampling.json
python3 cli/tests/laya_sampling_evaluate.py build-laya/cli/zli large.json --large-only
```

## Findings

Every entry below benchmarks **all seven** candidates at the specified scale.
It is not the original Laya top-two policy.

| Window size | Full-file winner found, initial 14 | Worse than numeric, initial 14 | Full-file winner found, larger 3 | Worse than numeric, larger 3 |
|---|---:|---:|---:|---:|
| 64 KiB | 8 | 5 | 0 | 3 |
| 1 MiB | 12 | 1 | 1 | 1 |
| 4 MiB | 12 | 1 | 2 | 0 |
| 16 MiB | 14 | 0 | 3 | 0 |

The initial 16 MiB result is not evidence of successful prediction: one or
two windows cover the entire 4/32 MiB file. Even the larger suite has only
three synthetic cases; its success does not guarantee unseen-file behavior.

Selected full-file sizes illustrate why a single larger default is inadequate:

| Dataset | Numeric | All candidates, 64 KiB | All candidates, 1 MiB | All candidates, 4 MiB | All candidates, 16 MiB | Exhaustive best |
|---|---:|---:|---:|---:|---:|---:|
| Original runs, 32 MiB | 6,486 | 13,109 | 278 | 278 | 278 | 278 |
| Long runs, 32 MiB | 494 | 2,845 | 350 | 350 | 281 | 281 |
| Irregular runs, 32 MiB | 4,333 | 6,595 | 6,134 | 6,134 | 4,333 | 4,333 |
| Long runs, 128 MiB | 1,940 | 11,387 | 1,364 | 1,364 | 1,097 | 1,097 |
| Irregular runs, 128 MiB | 9,230 | 21,808 | 27,987 | 8,938 | 8,938 | 8,938 |

The original run-heavy cycle occupies 136 KiB (17 values, each repeated 1,024
times, each eight bytes). A 64 KiB window contains only eight runs. The old
statistics report monotonicity 1, delta quantiles all zero, and a maximum of
15 rather than the full-file maximum 16. They conceal both periodic resets
and rare transitions. Larger windows expose more structure and change the
relative impact of per-frame setup and metadata costs.

## Recommended design

1. **Use measured sizes to decide, with broad candidate coverage.** Laya can
   order trials, but its top two must not permanently exclude other candidates.
   Always retain numeric as a comparison. Simply adding delta to the shortlist
   fixes some cases, but does not address scale-dependent rankings in general.

2. **Use multiple scales and actual chunk boundaries.** Begin with cheap
   probes; escalate when samples are extremely compressible, run lengths are
   censored by window boundaries, or rankings change with scale. Compare on
   real chunk-sized regions before accepting a change in these cases. Keep
   candidates eligible during escalation: the eventual winner can lose at
   every smaller scale. Budget caps and escalation thresholds still need a
   separately frozen evaluation; this experiment does not tune or validate
   them. Three full 16 MiB probes across seven candidates process 336 MiB of
   trial input, so blindly using this strategy on every file could be costly.

3. **Offer an explicit full-output guard when regression prevention matters.**
   Compress with the proposed graph and numeric using identical options;
   keep the smaller real output and export its compressor. This guarantees
   no size regression against numeric, but costs another full compression.
   It does not guarantee the best of all seven. Achieving that requires all
   seven full-file trials. Sample confidence alone cannot provide either
   guarantee for arbitrary inputs.

4. **Improve the information sent to Laya before changing its prompt.** Include
   full-file size, actual chunk size, compression level, window offsets, run
   boundary/censoring information, and nonzero-transition statistics. At the
   time of this investigation the worker saw only the original aggregate
   statistics and optional context. Entropy
   confidence measures how concentrated its choices are; we have not calibrated
   it as the probability of selecting the smallest OpenZL output. Raising the
   threshold alone just invokes the same faulty sampling policy more often.

5. **Require Laya to justify its latency.** Previous warm inference costs about
   114 ms, while sample-only routing was faster on all ten original conditions.
   For this seven-candidate problem, benchmark-only routing is an essential
   competitor. Reuse exported compressors for known homogeneous workloads to
   amortize routing. Keep Laya experimental until a frozen evaluation shows
   that it saves enough trials or improves size to offset its inference cost.

This investigation preceded implementation. Adaptive routing and the optional
full-output guard are now described in [laya.md](laya.md). The investigation
alone does not establish a new end-to-end speed improvement.

Implemented: broad candidate coverage, chunk-aligned adaptive probes, optional
full-output comparison against numeric, deferred export, and additional size,
level, offset and nonzero-delta statistics. Explicit run-censoring statistics,
calibrated model confidence and a demonstrated inference speed benefit remain
open. The [benchmark report](laya-benchmarks.md) records the fresh-seed misses.
