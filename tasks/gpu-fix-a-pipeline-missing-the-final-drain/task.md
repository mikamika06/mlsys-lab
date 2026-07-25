## Context

Software pipelining overlaps memory latency with compute: instead of
"fetch chunk, wait, compute on it, fetch the next chunk, wait, ...",
a **one-stage pipeline** fetches chunk `c+1` *while* computing on chunk
`c`, so the fetch for the next iteration is already in flight by the
time the current one finishes. The shape is a **prologue** (prefetch
the very first chunk before the loop starts), a **steady-state loop**
(each iteration prefetches the next chunk and consumes the one fetched
last iteration), and — the part that's easy to forget — an **epilogue**.

When the loop's counter runs out, it has prefetched one chunk more than
it has consumed: the very last chunk is sitting in the buffer, fetched,
ready, and never folded into the result. Nothing crashes. Nothing looks
wrong at a glance — the loop ran the right number of times, every index
was touched. The output is just quietly missing the contribution of
whatever was fetched last.

## Task

`solve.cu`'s `pipelined_sum` computes, for each thread `i`, the sum of
`num_chunks` values at `x[i*num_chunks .. i*num_chunks+num_chunks-1]`,
using this exact one-stage pipeline shape: prefetch chunk `0` into
`buf` before the loop; each iteration prefetches chunk `c` into `next`,
adds `buf` (last iteration's prefetch) into `acc`, then shifts
`buf = next`. The loop runs `c` from `1` to `num_chunks - 1`.

**Fix the missing drain**: after the loop exits, `buf` holds chunk
`num_chunks - 1` — the last chunk ever prefetched — which the loop body
never got a chance to consume. Add the one line that folds it into
`acc` before writing `out[i]`.

## Example

`num_chunks = 3`, values `[a, b, c]`. Prologue: `buf = a`. Iteration
`c=1`: `next = b`; `acc = 0 + a = a`; `buf = b`. Iteration `c=2`:
`next = c`; `acc = a + b`; `buf = c`. Loop ends with `acc = a+b` and
`buf = c` still unconsumed — the epilogue must add it:
`acc = (a+b) + c`.

## What the gate checks

The grader launches `pipelined_sum` over 64 threads, each summing 5
chunks of a fixed random input, and compares the output against each
row's true sum (`x.sum(axis=1)` in numpy). It requires

$$
\mathrm{max\_abs\_err} \le 10^{-9}
$$

On this fixture the undrained version is off by up to **2.8729** per
thread — exactly the magnitude of whatever value happened to be sitting
in `buf` when that thread's loop exited. Adding the one-line epilogue
drives the error to exactly `0`.
