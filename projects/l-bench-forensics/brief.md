# The number in the slide deck

Two people benchmarked the same model on the same machine and came back with
throughputs that differ by a factor of five. Neither of them is lying. One
measured prefill and one measured decode, and the deck says "tokens per second"
over both.

That is the easy version. The hard version is the run where somebody changed the
micro-batch size *and* the context depth between measurements, quoted the
difference as evidence for the micro-batch, and shipped a configuration change
on it.

You are building `benchkit`, the thing that reads a pile of `llama-bench` output
and answers a narrower question than "which is faster": **what are these numbers
allowed to prove.**

## The measurements

`projects/_fixtures/llama_bench/` holds three real sweeps recorded on an Apple
M4 Max with the Metal backend. Twenty-four rows, each with forty fields of
configuration and, importantly, the individual repetition timings rather than
only their average.

- `sweep.json` — prefill at 128, 512 and 2048 tokens and decode at 32 and 128,
  each at two micro-batch sizes.
- `depth_q4k.json` — the same 13B dense model with the KV cache pre-filled to 0,
  512, 2048 and 8192 tokens.
- `depth_moe.json` — a 30B mixture-of-experts model, three depths.

Three repetitions per row. That is few enough that the average is not
trustworthy and the honest thing is to say so, which is one of the milestones.

## What the data actually contains

Some of it is what you would expect: prefill runs about ten times the token rate
of decode, and decode slows as the context grows.

Some of it is not. One depth row is *faster* than the same configuration with an
empty cache, and the spreads do not overlap, so it is not noise you can wave
away — it is a measurement artefact, and an audit that reports a tidy decay
curve over it is worse than useless. Your `audit` has to notice.

The two models are also not comparable in the way the deck will want them to be.
The mixture-of-experts model decodes faster per second of wall clock while
carrying more weight in the file, so the weight bytes implied per second of
decoding come out several times higher — which is a statement about how much of
the model each token touches, not about which model is better.

## What you build

`benchkit/parse.py` — read the files, classify each row as prefill or decode,
derive throughput from the recorded nanoseconds rather than copying the reported
figure, and separate configuration from results. That separation is what the
rest depends on: a field like `n_ubatch` describes the run, `avg_ts` describes
its outcome, and `test_time` describes neither.

`benchkit/stats.py` — median, quartiles, inter-quartile range, and `separable`,
which returns 1 only when two sample sets' inter-quartile ranges do not overlap.
Distance between medians is not evidence. This is the same rule the rest of this
bank grades performance by, and here you implement it.

`benchkit/compare.py` — the confound finder. Given any two rows, which
configuration keys disagree. Given an axis, which pairs isolate it and which
pairs vary it alongside something else.

`benchkit/decay.py` — throughput against context depth: the table, the loss
relative to an empty cache, whether each loss is separable from noise, and the
least-squares slope in tokens per second lost per 1024 tokens of context.

`benchkit/report.py` — the recommendation. A micro-batch chosen for prefill
throughput subject to a decode floor, with every option and its numbers kept, so
the choice can be argued with instead of trusted.

`tests/test_bench.py` — `audit(rows)` returns the reasons not to act on this
data. Empty list on clean data; the real fixture is not clean.

## Milestones

1. Parsing and derived quantities.
2. Statistics, including a `separable` that refuses far-apart medians with
   overlapping spreads.
3. Controlled versus confounded comparisons.
4. The decay table, its separability flags, and its slope.
5. The recommendation and the cross-model reading.
6. The audit: it must flag the real anomaly and stay silent on clean data.
