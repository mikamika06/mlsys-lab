## Context

A warp coalesces its memory traffic into few 128-byte transactions only
when consecutive lanes touch consecutive addresses. That's a property of
*which thread reads which address*, not of the data itself — so relaying a
matrix out of column-major storage (`src[c*n+r]`) into row-major storage
(`dst[r*n+c]`) has a structural problem: `src` wants `r` to vary fastest
across a warp (so consecutive lanes land in the same column-major run),
while `dst` wants `c` to vary fastest (so consecutive lanes land in the
same row-major run). No single thread-to-`(r, c)` mapping satisfies both
at once — pick the mapping that coalesces the write, and the read becomes
a scattered "column-major gather," one transaction per lane instead of one
per warp.

The fix is the same trick a full shared-memory transpose uses, just for
one dimension of the disagreement: stage through `__shared__`. Read `src`
with the mapping *it* wants (coalesced), write the staged tile with
whatever indexing keeps the data correct, then — after a barrier — read
the tile back and write `dst` with the mapping *it* wants (also
coalesced). Both global-memory directions end up coalesced; only the
`__shared__` traffic in between pays for the mismatch, and that's over
100x cheaper per byte than global memory to begin with.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void relayout_col_to_row(float* dst, const float* src, int n);
```

`src` is an `n x n` matrix in column-major order (`src[c*n+r]` is
`matrix[r][c]`); write `dst` in row-major order (`dst[r*n+c]` is
`matrix[r][c]`). One block, `n*n` threads. Use `__shared__` staging so that
the global read from `src` AND the global write to `dst` are each
coalesced — pick a mapping where `r` varies fastest across the warp for
the read step, and `c` varies fastest for the write step, with a
`__syncthreads()` between them.

## Example

For `n = 16` (256 threads, 8 warps), the grader reports:

```
staged (reference): transactions = 16
naive (r fastest for the write, no staging): transactions = 72
```

Both produce byte-identical output (`max_abs_err = 0.0` either way — this
is pure data movement, nothing numerical to lose) — but the naive version,
which picks the mapping that coalesces `dst` and reads `src` with the
resulting column-major stride, needs 4.5x more global-memory transactions
for the exact same relayout.

## What the gate checks

`check.py` builds a random `16x16` matrix, parses `solve.cu`, and runs
`relayout_col_to_row` on the software GPU (`arena.cuda_sim.GPU`) with a
1-block, 256-thread launch. It requires `max_abs_err <= 1e-9` **and**
`transactions <= 20`. A kernel that gets every element into the right
place but reads `src` directly with the row-major-friendly mapping (no
`__shared__` staging at all) passes correctness outright — and still fails
the gate on `transactions = 72`.
