## Context

A tree reduction over 256 elements in shared memory needs $\log_2 256 =
8$ steps, each halving the number of live partial sums. The naive way to
pick which threads are active without a divergent `%` branch is
`index = 2*stride*tid`, active while `index < blockDim.x` — every active
thread's slot is `2*stride` apart from the next active thread's, which
sounds fine until you check which *bank* that lands in.

At `stride=1`, 32 threads in a warp compute `index = 0, 2, 4, ..., 62`.
Modulo 32 (the bank count), that's `0, 2, ..., 30, 0, 2, ..., 30` — the
pattern repeats after 16 threads, so **two** threads hit every bank: a
2-way conflict. At `stride=2`, `index = 0, 4, 8, ..., 124` repeats every 8
threads — **four** threads per bank, a 4-way conflict. Every step, the
stride doubles and so does the conflict degree.

**Sequential addressing** — start `stride` at `blockDim.x/2` and halve it
each step, with thread `tid` active while `tid < stride` — sidesteps this
completely: at every stride `>= 32`, each active thread's pair of
addresses (`tid`, `tid+stride`) never lands in the same bank as any other
active thread's pair.

## Task

`block_reduce_sum` (in `solve.cu`) sums 256 floats into `out[0]` using the
interleaved-addressing scheme above — it computes the exactly correct
sum, but with the 2-way/4-way/8-way/... conflict growth. Fix the
*addressing*, not the arithmetic: rewrite the reduction to use sequential
addressing (`stride = blockDim.x/2` down to `1`, halving each step;
`tid < stride` active; `sdata[tid] += sdata[tid+stride]`), with a
`__syncthreads()` after the initial load and after every step.

## Example

Reducing 256 known values, both the interleaved and the sequential
version write the identical sum to `out[0]` — the bug is purely about
*how fast* that sum gets computed, not *what* it computes.

## What the gate checks

`max_abs_err <= 1e-9` (the sum itself must still be exactly right) *and*
`smem_waves <= 50`. The shipped interleaved-addressing kernel measures
`150` total shared-memory waves across all 8 steps on the fixed 256-element
input; correct sequential addressing measures `45`. Any fix that still
uses a stride-multiplied index (even without the modulo branch) keeps the
same doubling conflict pattern and stays well above the gate.
