## Context

Summing an array into a single running total creates a **loop-carried
dependency chain**: `s += x[i]` can't start until the previous `+=`
finishes, so a reduction of $N$ elements has a critical path of length
$N$ no matter how many execution ports the CPU has. Splitting the work
across $K$ **independent** accumulators — accumulator $j$ takes every
$K$-th element — breaks that single long chain into $K$ chains of length
roughly $N/K$ that have no dependency on each other, so the hardware can
overlap them. The accumulators are combined into one total only at the
end.

This task makes the critical-path length *real and measurable*: every
value is a `Tracked{value, depth}` pair (declared in `sol.hpp`), and its
`operator+` computes `depth = 1 + max(a.depth, b.depth)` — exactly the
standard "longest path in a DAG of additions" recurrence. A single running
accumulator's depth grows by 1 on every add, ending at $N$; $K$
independent accumulators end near $N/K$ (plus a handful more to combine
the $K$ partials).

## Task

Implement

```cpp
Tracked reduce_with_accumulators(const Tracked* x, int n, int num_accumulators);
```

using `num_accumulators` independent accumulators: accumulator $j$
(`0 <= j < num_accumulators`) should sum every `x[i]` with
`i % num_accumulators == j`. Combine the partial sums into the final
total. The returned `.value` must be the exact sum of `x[0..n)`; the
returned `.depth` is whatever falls out of actually using
`operator+` correctly — you don't set it directly.

## Example

For `x = {1,2,3,4,5,6}` (depth 0 each) and `num_accumulators = 3`:
`partial[0] = x[0]+x[3]` (depth 1), `partial[1] = x[1]+x[4]` (depth 1),
`partial[2] = x[2]+x[5]` (depth 1); combining them sequentially gives a
final depth of 3 — versus depth 5 for one accumulator summing all six
serially.

## What the gate checks

`main.cpp` builds $N = 4096$ deterministic integer-valued doubles (so the
sum is exact regardless of accumulation order) and calls your function
with 4 accumulators. It prints the sum, then a `depth_ok` flag that's `1`
only if the resulting `.depth` stayed at or below 1200 — the real 4-way
split lands around 1027, while summing everything into one running
accumulator lands around 4096. A solution that ignores
`num_accumulators` and sums serially gets the **value** exactly right but
`depth_ok 0`, and fails the gate even though the printed sum matches — the
point of this task is the shape of the dependency chain, not just the
final number. The grader compiles your `.cpp` with the real local
`clang++`, runs it, and requires the printed output to match the
reference's exactly.
