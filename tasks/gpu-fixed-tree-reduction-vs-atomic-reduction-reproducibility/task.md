## Context

Floating-point addition isn't associative: `(a+b)+c` and `a+(b+c)` can
round to different bit patterns. A reduction's exact result therefore
depends on the *order* the additions happen in, not just the value set.

A **fixed-structure tree reduction** (the kind every hand-written CUDA
block reduction uses) combines shared-memory slot `tid` with slot
`tid + stride` at every step — a pairing determined entirely by thread
index and stride, never by which thread physically finished first. Run
it a thousand times on the same input and you get the exact same bits a
thousand times.

An **atomic-style reduction** (`atomicAdd(&sum, x[tid])` from every
thread) has no such fixed structure — the order the hardware happens to
grant the atomic to each thread is a real scheduling detail that can
vary between runs, even on the exact same input. Summing the exact same
values in a different order can land on a different bit pattern.

## Task

Implement both, in real CUDA-C:

```cuda
__global__ void tree_reduce_sum(float* out, const float* x, int n);
__global__ void ordered_reduce_sum(float* out, const float* x, const float* order, int n);
```

`tree_reduce_sum`: standard sequential-addressing tree reduction (`stride
= blockDim.x/2` down to `1`, halving each step; `tid < stride` active;
barrier every step) into `out[0]`.

`ordered_reduce_sum` models one specific atomic arrival sequence: from
thread 0 only, accumulate `x[order[0]], x[order[1]], ..., x[order[n-1]]`
**strictly in that order**, one addition at a time, into `out[0]`.

## Example

32 values: one huge one (`1e16`) plus the integers `1..31` (true sum
`1e16 + 496`). Summing them **in three different orders** — ascending
index, descending index, and a fixed shuffle — on the identical value
set produces **three different results**, `20.0` apart at the extremes:
proof that "the same logical contributions, different arrival order"
really does change the answer. The tree reduction's own combine order is
fixed by its stride structure, so it always lands on one specific,
predictable value for this input, no matter how many times you run it.

## What the gate checks

`max_abs_err <= 1e-6` across `tree_reduce_sum`'s output and
`ordered_reduce_sum`'s output for three different `order` arrays
(ascending, descending, one fixed shuffle), each compared against a numpy
oracle that replicates the exact same sequence of additions. Reducing
`ordered_reduce_sum` with anything other than a strict single-threaded
walk through `order` (e.g. sorting first, or using a tree instead of a
straight accumulate) changes at least one of the three order-dependent
sums and fails the match.
