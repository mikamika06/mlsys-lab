## Context

NumPy (or PyTorch) broadcasting an $(R, C)$ matrix plus a length-$C$
vector hides the index arithmetic entirely: `mat + vec` "just works". A
GPU kernel has no such thing — every thread computes ONE flat output
index and has to derive, by hand, which row and column that corresponds
to, and which element of the shorter vector to add.

For a flat index $i$ into a row-major $(R, C)$ matrix, the row and column
are:

$$\text{row} = \left\lfloor \frac{i}{C} \right\rfloor \qquad \text{col} = i \bmod C$$

`out[i] = mat[i] + vec[col]` only needs `col`, since `mat[i]` and `out[i]`
are already addressed by the flat index directly.

## Task

Write a CUDA-C kernel:

```cpp
__global__ void broadcast_add(float* out, const float* mat, const float* vec, int rows, int cols);
```

One thread per output element: `i = blockIdx.x * blockDim.x + threadIdx.x`.
Guard `i < rows * cols`, derive `col = i % cols`, and write
`out[i] = mat[i] + vec[col]`.

## Example

For a $4 \times 64$ matrix (launched as 4 blocks of 64 threads, one block
per row) and a length-64 vector, thread `i` in block `b` has
`col = i % 64`. Since each block covers exactly one full row, `col` sweeps
`0..63` — the exact same consecutive run `threadIdx.x` sweeps — so a warp
of 32 consecutive threads touches 32 consecutive elements of `mat`, `out`,
AND `vec` at once: every access coalesces into a single 128-byte
transaction.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it
thread-by-thread on the software GPU over a fixed random $4 \times 64$
fixture, checking the result against `mat + vec` broadcast in numpy
(`max_abs_err <= 1e-9`) and the simulator's observed transaction count
(`transactions <= 24` — exactly what the reference's row/col derivation
produces: 8 warps $\times$ 3 coalesced accesses each). Swapping the index
math (e.g. deriving `col` from `i / rows` instead of `i % cols`, or
reading `vec` at a scattered per-thread offset) breaks the coalescing,
correctness, or both. The empty starter touches no memory at all and
fails the correctness gate.
