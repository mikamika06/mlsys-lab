## Context

A grid-stride kernel doesn't need exactly `ceil(N / block_size)` blocks —
launch fewer, and each block just loops further to cover more elements.
That flexibility matters because a GPU can only keep `max_concurrent`
blocks resident at once; a grid of `num_blocks` runs in
`ceil(num_blocks / max_concurrent)` back-to-back **waves**, each wave
occupying every SM slot until the whole wave finishes before the next
one starts. If `num_blocks` isn't an exact multiple of `max_concurrent`,
the *last* wave only fills part of the machine — every slot the tail
wave doesn't use sits idle for that wave's entire duration, the same
waste a ragged final loop iteration causes on a CPU.

The fix doesn't require more blocks — it requires *fewer*, chosen so
every wave is completely full: drop down to the largest multiple of
`max_concurrent` that's still `<=` the natural block count.

## Task

Implement

```cuda
__global__ void optimal_grid_blocks(float* out, int idx, int N, int block_size, int max_concurrent);
```

Compute `total_blocks = ceil(N / block_size)`. If
`total_blocks > max_concurrent`, the answer is the largest multiple of
`max_concurrent` that is `<= total_blocks` — integer-divide
`total_blocks` by `max_concurrent`, then multiply back
(`(total_blocks / max_concurrent) * max_concurrent`). Otherwise (the
natural block count doesn't even fill one wave, so there's only one wave
no matter what), the answer is just `total_blocks`. Write the result to
`out[idx]`.

## Example

`N = 1000`, `block_size = 128`, `max_concurrent = 3`:
`total_blocks = ceil(1000/128) = 8`. `8 > 3`, so the answer is
`(8 / 3) * 3 = 2 * 3 = 6` — 6 blocks run as exactly 2 full waves of 3;
each block's grid-stride loop covers a bit more ground than it would
with 8 blocks, but no SM slot ever sits idle waiting on a half-empty
final wave.

## What the gate checks

`check.py` runs 4 fixed `(N, block_size, max_concurrent)` scenarios,
launching `optimal_grid_blocks` as a single thread each time (a real
kernel launch through the CUDA-C frontend and simulator), and compares
the written result against an independently-computed Python integer
reference (the same formula, computed fresh from each scenario's
numbers — never a hardcoded expected value) for every scenario. It
requires

$$
\mathrm{max\_abs\_err} = \max_i |\text{out}_i - \text{oracle}_i| \le 10^{-6}
$$

The four scenarios cover both branches: `(1000,128,3) \to 6`,
`(100000,256,40) \to 360`, `(777,64,5) \to 10`, and
`(50,32,8) \to 2` (this last one has only `2` natural blocks against a
capacity of `8` — already a single, unavoidably partial wave, so the
answer is just the natural count, not `0`). A stub that always writes
`total_blocks` unmodified gets the first three wrong (it never rounds
down to a full-wave multiple) and fails the gate.
