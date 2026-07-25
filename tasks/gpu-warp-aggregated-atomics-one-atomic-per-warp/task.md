## Context

An `atomicAdd` to one shared counter serializes every thread that touches
it, but real hardware gives a warp a cheap way to cut its own contention
before it ever reaches that counter: **ballot** — every lane votes whether
it wants to increment — followed by electing one **leader** lane. Only
that leader issues an `atomicAdd`, and it adds the *count* of lanes that
voted yes, not `1`. A whole warp's worth of increments collapses into a
single atomic operation.

The size of the win depends entirely on how many lanes in the warp
actually wanted to increment. A warp where all 32 lanes vote yes turns 32
atomics into 1 — a 32x reduction. A warp where only 1 lane votes yes still
needs exactly 1 atomic either way — aggregation buys nothing there, because
there was only ever one contender. An entirely idle warp (0 votes) does
zero atomics regardless of strategy. The benefit tracks *occupancy
density* within a warp, not just the raw count of increments happening
somewhere in the kernel.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void warp_aggregated_atomic_counts(float* naive_out, float* warp_agg_out,
                                               const float* active_count, int m);
```

Each configuration `i` in `[0, m)` IS one warp, described by
`active_count[i]` — how many of its 32 lanes want to increment. For each
`i`: `naive_out[i] = active_count[i]` (one atomic per active lane), and
`warp_agg_out[i] = 1.0f` if `active_count[i] > 0`, else `0.0f` (one atomic
for the whole warp, but only if it has any active lane at all).

## Example

The grader runs 10 warps. A fully dense warp (`active_count = 32`) goes
from `naive_out = 32` to `warp_agg_out = 1` — the full 32x. A warp where
only 1 lane is active (`active_count = 1`) stays at `warp_agg_out = 1` too
— no reduction at all, since the naive count was already `1`. An idle warp
(`active_count = 0`) reports `0` under both strategies.

## What the gate checks

`check.py` parses `solve.cu` and runs `warp_aggregated_atomic_counts` on
the software GPU (`arena.cuda_sim.GPU`) with a 1-block, 10-thread launch.
It requires `max_abs_err == 0.0` against both outputs computed
independently. Reporting `warp_agg_out[i] = 1.0f / 32.0f` (an *average*
reduction factor, applied uniformly) matches the one fully-dense warp in
the fixture but is wrong on every other one — the reduction a warp gets is
either `1` or `0`, never a fraction, and it depends entirely on that
specific warp's own occupancy, not a fleet-wide average.
