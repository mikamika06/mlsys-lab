## Context

A single running accumulator is hostage to whatever order its values
arrive in. If a huge-magnitude spike shows up mid-stream, every value the
accumulator has gathered so far gets swallowed the instant it combines
with that spike (its bits fall below the spike's floating-point
granularity), even though a cancelling value later brings the
accumulator's magnitude back down — the small values lost along the way
never come back.

**Split-accumulate (blocked) reduction** processes the array in fixed-size
groups, each with its OWN fresh accumulator that starts back at `0` for
every group — a magnitude spike inside one group can only ever damage
that group's own local total, never leak into a sibling group's. If a
spike and its exact cancellation both happen to land in the same group,
that group's own total comes out exact, and grouping shields every other
group from ever seeing that spike at all.

## Task

Write a CUDA-C kernel (single thread):

```cpp
__global__ void split_accumulate(float* out, const float* values, int n, int block_size);
```

Process `values[0..n)` in consecutive groups of `block_size`: for each
group, accumulate a fresh `block_sum` (starting at `0`) over that group's
`block_size` elements, then add the finished `block_sum` into a running
`total`. Write the final `total` to `out[0]`.

## Example

The fixture is 30 ones, then a cancelling pair `[1e20, -1e20]`, then 32
more ones — 64 elements, true sum exactly `62.0`. With `block_size = 2`,
the pair falls in its own group (`block_sum = 1e20 + (-1e20) = 0.0`,
exact — nothing else is in that group to be swallowed), while every other
group is a pair of ones (`block_sum = 2.0`, exact). Combining all 32
group totals (all of them small: `2.0`, `2.0`, ..., `0.0`, ..., `2.0`)
never risks a magnitude mismatch, so the grand total comes out exactly
`62.0`.

A single naive running accumulator over the SAME 64 elements in the SAME
order fails: it reaches `30.0` after the first 30 ones (exact so far),
then adding `1e20` rounds the result to exactly `1e20` — `1e20`'s
floating-point granularity at that magnitude is far coarser than `30`,
so those 30 ones are gone, permanently. Adding `-1e20` next cancels the
spike back to `0`, but the 30 it swallowed on the way up never returns.
The remaining 32 ones bring it to `32.0` — the correct answer is `62.0`.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU over the fixed fixture, requiring `rel_err <=
1e-9` against the true sum of `62.0`. A single unbroken running
accumulator (no block reset) computes `32.0` — about 48% relative error —
and fails the gate; only resetting the accumulator at each `block_size`
boundary, as the reference does, isolates the magnitude spike and
recovers the exact answer. The empty starter fails identically.
