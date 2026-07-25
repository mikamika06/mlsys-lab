## Context

Some values in a kernel are "hot" in a very specific way: every thread
wants them, and every thread wants the *exact same* address, every
single time. A per-element bias added once per output, a shared
normalization constant, a broadcast scale factor — these never vary
across the warp and never vary across iterations of a loop. That's
precisely the shape of data GPUs give a fast **broadcast path** for
(constant / read-only cache on real hardware): the whole warp gets it
in one fetch.

But a fast broadcast path doesn't help if the kernel throws it away —
re-reading the same broadcast address from global memory on every
iteration of a loop still costs one global-memory access step *per
iteration*, even though the address, and the value, never changed.
The fix has nothing to do with *where* the value lives in the memory
hierarchy; it's simpler than that: read it once, keep it in a register,
stop asking global memory a question you already know the answer to.

## Task

`solve.cu`'s `biased_sum` computes, for each thread `i`,
`out[i] = sum_{k=0}^{K-1} (x[k][i] + bias[0])` — but re-reads
`bias[0]` from global memory on every one of the `K` loop iterations.
**Fix it**: load `bias[0]` into a local variable once, before the loop
starts, and use that local variable inside the loop instead of
re-indexing `bias` every time.

## Example

`K = 20`: the buggy version issues 20 separate global-memory access
steps for `bias[0]` per thread (all landing on the identical address,
still 20 distinct steps in the access trace) — the fixed version issues
exactly 1.

## What the gate checks

The grader launches `biased_sum` over 64 threads (2 warps) summing 20
values each against a fixed bias, checks the output against
`x.sum(axis=0) + K*bias[0]` (numpy), and reads the real transaction
count from the simulator's coalescing model. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-9} \quad\text{and}\quad \mathrm{transactions} \le 60
$$

Both versions are numerically identical (`max_abs_err` effectively `0`
either way) — this is purely a traffic bug. Re-reading `bias[0]` every
iteration measures **84** transactions; loading it once measures
**46** — the redundant reads alone account for almost half of the
buggy version's total global-memory traffic.
