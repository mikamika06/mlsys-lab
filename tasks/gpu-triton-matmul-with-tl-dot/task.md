## Context

Block-level GPU programming (Triton's `tl.dot` model included) launches
a grid of fixed-size tiles over the output matrix — but the matrix's
true dimensions almost never divide evenly by the tile size. The last
tile along each axis inevitably **overhangs**: some of its threads
correspond to `(row, col)` pairs that don't exist in the real matrix at
all. Reading `A`/`B` or writing `C` for those threads would touch memory
past the end of the array — the fix is a boundary **mask**: compute the
`(row, col)` a thread would own, and only actually read/write if both
are still within the true `M`/`N` bounds.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void block_matmul_masked(float* C, const float* A, const float* B, int M, int N, int K);
```

Fixed for this task: `M=N=K=5` against a `4x4`-thread block tile (`BLOCK
= 4`) — grid is `4` blocks of `16` threads (a 2x2 arrangement of tiles,
flattened: `blockRow = blockIdx.x/2`, `blockCol = blockIdx.x%2`;
`tx = threadIdx.x%4`, `ty = threadIdx.x/4`; `row = blockRow*4+ty`,
`col = blockCol*4+tx`). Only if `row < M && col < N`: accumulate
`A[row*K+k] * B[k*N+col]` over `k` in `[0,K)`, then
`C[row*N+col] = acc`.

## Example

Block `3` (`blockRow=1, blockCol=1`) covers rows `4..7` and columns
`4..7` — but the matrix only has rows/columns `0..4`. Thread
`(tx=1,ty=1)` in that block computes `row=5, col=5`: both are `>= 5`
(`M=N=5`), so it must do nothing at all — no read, no write. Thread
`(tx=0,ty=0)` in the same block computes `row=4, col=4`: both in range,
so it computes the real `C[4][4]`.

## What the gate checks

`max_abs_err <= 1e-6` against `A @ B` (numpy) for the fixed `5x5x5`
problem. Missing the mask (or masking only one of `row`/`col`) either
crashes on an out-of-range global memory access or silently overwrites a
valid neighboring `C` element with garbage from the overhanging thread.
