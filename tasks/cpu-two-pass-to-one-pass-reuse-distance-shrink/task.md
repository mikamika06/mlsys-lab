## Context

Computing a mean AND a variance is a classic "needs two statistics" job:
$\text{sum} = \sum x_i$ and $\text{sumsq} = \sum x_i^2$ (from which
$\text{mean} = \text{sum}/n$ and $\text{var} = \text{sumsq}/n -
\text{mean}^2$). The naive way to get both is one loop per statistic —
two full passes over the array. If the array is bigger than the cache
(the common case for anything that matters), the second pass cannot reuse
a single byte the first pass brought in: by the time the loop wraps back
around, the early elements have long since been evicted to make room for
the later ones. Reading `n` elements twice costs roughly `2n` worth of
cache misses.

The fix doesn't need two arrays or extra memory — `sum` and `sumsq` are
completely independent running accumulators. Track both **in the same
loop**, over the same single pass through `x`, and every element is read
from memory exactly once.

## Task

`sol.hpp` gives you a deterministic 64-line, 64-byte-line, fully-associative
LRU cache model (`reset_cache()` / `touch_byte(addr)` / `miss_count()`).
Implement:

```cpp
void compute_stats(const float* x, int n, float* out_sum, float* out_sumsq);
```

Compute `*out_sum = sum(x[0..n))` and `*out_sumsq = sum(x[i]^2 for i in
[0, n))` in a SINGLE forward pass over `x`. Call `touch_byte(&x[i])`
exactly once per element, in order — not once per statistic.

The driver (`main.cpp`, fixed) calls this with `n = 2048`
(`x[i] = (i % 13) * 0.3 + 1.0`) — 8192 bytes, twice the modeled cache's
4096-byte capacity — and prints the two statistics plus the total miss
count.

## Example

```
sum=5728.104004
sumsq=18600.919922
misses=128
```

2048 floats span exactly $8192 / 64 = 128$ distinct 64-byte lines — a
single pass touches each one exactly once, so `misses == 128` no matter
how much bigger than the cache the array is.

Reading `x` twice (once for `sum`, once for `sumsq`, in two separate
loops) computes the exact same two numbers, but touches `128` lines
twice each: by the time the second loop starts, the 64-line cache has
long since evicted everything the first loop brought in, so it re-misses
essentially all of them again:

```
sum=5728.104004
sumsq=18600.919922
misses=256
```

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed number to satisfy `max_abs_err <=
1e-3` against the same driver linked with the reference. Two separate
loops get both statistics numerically right but print `misses=256`
instead of `128` — a difference of `128`, which alone fails the gate.
