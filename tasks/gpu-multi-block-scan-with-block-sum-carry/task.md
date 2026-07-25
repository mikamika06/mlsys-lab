## Context

A prefix sum (scan) over one block is cheap — a single warp-synchronous
shuffle ladder. A scan over *multiple* blocks is harder, because there is
no way for one block to wait on another mid-kernel: no global barrier
exists across blocks in a single launch. Real multi-block scans solve
this with **three separate kernel launches**, each block only ever
talking to global memory, never to another block directly:

1. Every block scans its own slice locally, and records its own total.
2. A single small kernel scans the (few) per-block totals into each
   block's **carry**: the sum of every block that comes strictly
   *before* it — an **exclusive** scan, not an inclusive one, since a
   block must not add its own total to itself.
3. Every block adds its carry to every element it already locally
   scanned.

## Task

Implement

```cpp
__global__ void multi_block_scan(float* data, float* block_sums, int phase, int n_blocks);
```

launched with 128 elements = 4 blocks of 32 (one warp per block),
branching on `phase`:

- **`phase == 0`** (launched with grid=4, block=32): each block's 32
  threads run the intra-warp inclusive-scan shuffle ladder (`delta` =
  1, 2, 4, 8, 16, each step guarded by `lane >= delta`) over its own 32
  elements of `data`, writing the local scan back in place. Lane 31 (the
  block's final, largest value) writes it to
  `block_sums[blockIdx.x]`.
- **`phase == 1`** (launched with grid=1, block=4): each of the 4
  threads loads `block_sums[tid]`, runs the SAME shuffle-ladder pattern
  but only 2 steps (`delta` = 1, 2 — there are only 4 lanes), and then
  **subtracts its own original value** before writing back — turning the
  inclusive scan of totals into each block's exclusive carry.
- **`phase == 2`** (launched with grid=4, block=32): every thread reads
  `block_sums[blockIdx.x]` (now its block's carry) and adds it to its
  own `data[i]`.

## Example

Block sums (totals) `[3.0, -1.0, 2.0, 5.0]`: the inclusive scan of those
is `[3.0, 2.0, 4.0, 9.0]`. Subtracting each lane's own original value
gives the exclusive carries `[0.0, 3.0, 2.0, 4.0]` — block 0 gets carry
`0` (nothing precedes it), block 3 gets carry `4.0` (the sum of blocks
0-2's totals, `3.0 + -1.0 + 2.0`), **not** `9.0` (which would incorrectly
include block 3's own total).

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and launches
it three times, in order (`phase` 0, then 1, then 1's grid-of-1 result
feeding phase 2), over a fixed 128-element random input, comparing the
final array against numpy's own `cumsum`. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

Leaving `block_sums` as the plain **inclusive** scan of the block totals
in phase 1 (forgetting to subtract each lane's own value) measures
`max_abs_err ≈ 13.99` on this fixture — every block from the second one
onward gets its own total folded into its own carry on top of every
earlier block's, over-adding by exactly that block's own sum.
