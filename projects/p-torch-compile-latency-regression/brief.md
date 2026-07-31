# p99 went up after compile

The classification service held p50 at 41 ms per batch. After the "enable
torch.compile" PR, the benchmark in the README shows 12 ms, but prod p99 went
up to 210 ms, and for the first few seconds after deploy the service
sometimes hangs. We rolled compile back — stable again, but slow.

We need to bring compile back, hold the line on p99, and leave a guard in the
repo so this doesn't happen again.

Measure the numbers on your own machine. We're comparing against the code we
started from, not an absolute value: someone else's hardware doesn't matter
here.

## What you write

**`bench/bench.py`** — `measure(fn, warmup, reps, timer, sync)`. The one that
exists times the whole loop with a single timer and divides by the number of
reps. Yours must return `median` and a spread measure, do warmup **outside**
the measured region, and call `sync` after each rep. `timer` and `sync` are
injected from outside — otherwise the rig can't be tested without waiting on
real time.

**`tools/inventory.py`** — `inventory(model, example) -> {"graph_count",
"graph_break_count", "op_count"}`.

**`tools/guards.py`** — `failing_guards(text) -> list[list[str]]`. Input is a
`TORCH_LOGS=recompiles` dump, output is one list of guard expressions per
recompilation event, in the order they appear.

**`service/`** — the service itself. There are three defects in here, and
the ticket names none of them.

**`tests/test_regression.py`** — your safety net.

## Milestones

1. A measurement rig you can trust. Checked with an injected clock: there's
   no real time in the check, so the result doesn't depend on the machine.
2. Graph inventory on the model the grader provides.
3. Attribution of recompilations to guards, on a fixed log.
4. `torch.compile(model, fullgraph=True)` doesn't throw, zero breaks, output
   matches eager.
5. Eight batch sizes produce no more than three unique graphs.
6. Speedup vs. the starting code, measured by interleaved A/B. The gate
   demands not just a ratio but **separability**: if the interquartile
   ranges overlap, there's no win, just noise.
7. We reintroduce a graph break in `forward` — your test must fail.

```
mlsys project start p-torch-compile-latency-regression
mlsys project grade p-torch-compile-latency-regression --milestone 1
```
