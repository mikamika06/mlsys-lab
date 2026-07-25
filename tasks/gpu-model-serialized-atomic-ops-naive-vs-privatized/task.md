## Context

An `atomicAdd` to a single, shared memory location is not free just
because it's correct: every thread that hits it has to be serialized
against every other thread hitting it, because only one update can
actually happen at a time. Building a histogram (or any other "every
thread contributes to one shared total" computation) the naive way —
every one of `n` threads does its own `atomicAdd` straight into the shared
bin — means `n` serialized atomic operations, no matter how many blocks or
SMs the work is spread across.

**Privatization** breaks that bottleneck: give every *block* its own
private copy of the bin, let every thread in that block update the private
copy with ordinary (uncontended, free) memory operations, and only flush —
one `atomicAdd`, from one thread — once per block, at the very end. The
total count of *global* atomic operations drops from `n` (one per thread)
to the number of blocks (one per block) — however many threads each block
has.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void modeled_atomic_counts(float* naive_out, float* privatized_out,
                                       const float* n, const float* block_size, int m);
```

For each configuration `i` in `[0, m)`, given `n[i]` total updates spread
across `block_size[i]`-thread blocks:

- `naive_out[i] = n[i]` — one global atomic per update.
- `privatized_out[i] = ceil(n[i] / block_size[i])` — one global atomic per
  block (round up: a partially-filled last block still needs its own
  flush).

## Example

The grader runs 5 configurations. For `n = 100000, block_size = 256`:
`naive_out = 100000`, `privatized_out = ceil(100000/256) = 391` — nearly
256x fewer global atomics, matching `block_size` almost exactly (each
flush "absorbs" roughly one block's worth of updates). A tiny
configuration, `n = 7, block_size = 32` (fewer updates than a single
block's threads), still needs `privatized_out = 1` — you can't flush zero
times even for one partially-full block.

## What the gate checks

`check.py` builds the 5 configurations, parses `solve.cu`, and runs
`modeled_atomic_counts` on the software GPU (`arena.cuda_sim.GPU`) with a
1-block, 5-thread launch. It requires `max_abs_err <= 1e-6` against both
outputs computed independently (`n[i]` and `numpy.ceil(n[i]/block_size[i])`).
Using plain `floor` instead of `ceil` for `privatized_out` matches the one
configuration where `n[i]` divides evenly by `block_size[i]` but
undercounts every other configuration by one block — silently claiming one
fewer atomic flush happened than the last, partially-full block actually
needs.
