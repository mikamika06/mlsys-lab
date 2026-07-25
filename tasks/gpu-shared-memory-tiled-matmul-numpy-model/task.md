## Context

A naive matmul thread reads its whole row of `A` and column of `B`
straight from global memory — for an output tile of `TILE x TILE`
threads, the same `TILE` rows of `A` and `TILE` columns of `B` each get
re-read from global memory `TILE` separate times (once per thread that
needs them). **Tiling through shared memory** fixes that: the block
cooperatively loads one `TILE x TILE` chunk of `A` and one of `B` into
shared memory ONCE, every thread in the block reuses those same shared
values for its own partial dot product, and only then does the block
move on to the next `K`-chunk.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void tiled_matmul(float* C, const float* A, const float* B, int M, int N, int K);
```

Fixed for this task: `M=N=K=8`, `TILE=4`, launched as 4 blocks of 16
threads (a 2x2 arrangement of `4x4` output tiles). Each thread:
`tx = threadIdx.x % 4`, `ty = threadIdx.x / 4`,
`blockRow = blockIdx.x / 2`, `blockCol = blockIdx.x % 2`,
`row = blockRow*4 + ty`, `col = blockCol*4 + tx`. For each of `K/4 = 2`
`K`-sub-tiles `t`: load `As[ty*4+tx] = A[row*K + t*4+tx]` and
`Bs[ty*4+tx] = B[(t*4+ty)*N + col]` into `__shared__ float As[16]`,
`Bs[16]`; `__syncthreads()`; accumulate `As[ty*4+k] * Bs[k*4+tx]` for `k`
in `[0,4)`; `__syncthreads()` again before the next sub-tile. Finally
`C[row*N + col] = acc`.

## Example

Thread `(tx=1, ty=2)` in block `0` (`row=2, col=1`) computes
`C[2][1] = sum_{k=0}^{7} A[2][k]*B[k][1]`, split into two `TILE`-deep
partial sums (`k=0..3` from sub-tile `0`, `k=4..7` from sub-tile `1`),
each computed entirely out of that sub-tile's shared-memory staging —
`A[2][0..3]` and `B[0..3][1]` are each read from global memory exactly
once, no matter how many of the block's 16 threads need them.

## What the gate checks

`max_abs_err <= 1e-6` against `A @ B` (numpy), **and**
`transactions <= 15` on the simulator's measured global-memory traffic
(reference measures `10`). A version that recomputes results correctly
but re-reads `A`/`B` straight from global memory inside the inner `k`
loop (skipping the shared-memory staging) still gets the values right
but blows well past the transaction budget.
