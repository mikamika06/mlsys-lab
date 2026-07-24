## Context

In an untiled matmul every thread streams a whole row of A and a whole
column of B, so each element of A is re-fetched `n` times across the block.
Tiling fixes that: the block cooperatively loads a `16 x 16` piece of A and
of B into `__shared__` memory, everyone reads it from there, and global
memory sees each element once per tile row instead of once per thread.

The simulator counts real 128-byte global transactions per warp, so the win
is measured, not asserted.

This CUDA-C frontend only supports **1D launches** (`threadIdx`/`blockIdx`
only carry a real `.x`; `.y` always reads `0`). The natural 2D tiling — a
`2 x 2` grid of `16 x 16`-thread blocks — is flattened into a 1D launch of
`grid = 4` blocks of `block = 256` threads, and the kernel recovers its 2D
tile coordinates itself:

```
tx = threadIdx.x % 16;              ty = threadIdx.x / 16;
bx = blockIdx.x % tiles_per_row;    by = blockIdx.x / tiles_per_row;
row = by * 16 + ty;                 col = bx * 16 + tx;
```

## Task

Implement the kernel (declared with this exact signature):

```cuda
__global__ void tiled_matmul(float* a, float* b, float* c, int n, int tiles_per_row);
```

`a` is A (offset 0), `b` is B, `c` is C, all `n x n` row-major in one flat
buffer; `tiles_per_row = n / 16`. Recover `row`/`col` as shown above, then
for each `k0 = 0, 16, 32, ...` below `n`:

1. Stage `a[row][k0 + tx]` and `b[k0 + ty][col]` into two `__shared__
   float[16 * 16]` tiles.
2. `__syncthreads()` — the whole block's tiles must be loaded before anyone
   reads them.
3. Accumulate 16 products from the shared tiles into a running `float acc`.
4. `__syncthreads()` again — nobody may overwrite a tile a sibling thread is
   still reading from.

After the loop, write `acc` to `c[row][col]`.

## Example

For the fixed 32x32, seed-0 test case, a correct tiled kernel measures
`max_abs_err` on the order of `1e-15` against `A @ B` (floating-point noise
only) and a `transaction_ratio` around `0.10` — roughly a 10x cut in global
transactions versus the untiled baseline the grader measures internally. An
empty kernel body (the starter) never writes `c`, so every output element
stays `0`: `max_abs_err` is large (order `10`), failing that gate outright
regardless of the (trivially low, since nothing was touched) transaction
count.

## What the gate checks

- `max_abs_err <= 1e-9` against `A @ B` computed with numpy.
- `transaction_ratio <= 0.55`: measured global transactions (from the real
  CUDA-C kernel, executed thread-by-thread on `arena.cuda_sim.GPU`) divided
  by an untiled baseline the grader measures with its own internal 2D
  kernel on the same simulator.
