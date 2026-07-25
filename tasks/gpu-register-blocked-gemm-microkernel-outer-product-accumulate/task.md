## Context

The simplest GEMM kernel gives one thread one output element: for that
element's whole `K`-length dot product, the thread reads one value of
`A` and one of `B` per step — 2 loads feeding 1 FMA. **Register
blocking** (thread coarsening) gives each thread a small `TM x TN` TILE
of the output instead, held entirely in its own registers, and at each
step of the `K` loop it loads `TM` values of `A` and `TN` values of `B`
— `TM + TN` loads — and forms their **outer product**: every
`A`-value times every `B`-value, `TM * TN` FMAs from those same
`TM + TN` loads. For a `2x2` tile, that's `4` loads producing `4` FMAs,
instead of the `8` loads four separate scalar-output threads would need
for the same `4` FMAs — each loaded value gets reused `TN` (or `TM`)
times instead of once.

## Task

Implement

```cuda
__global__ void gemm_regblock(float* C, const float* A, const float* B, int M, int N, int K);
```

Each thread owns a `2x2` tile of `C`. With `tilesPerRow = N / 2`:
`tileCol = tid % tilesPerRow`, `tileRow = tid / tilesPerRow`,
`row0 = tileRow*2`, `col0 = tileCol*2`. Keep 4 running accumulators
(`acc00, acc01, acc10, acc11`, plain local floats, all starting at `0`)
— these never touch shared or global memory until the very end. For
`k = 0 .. K-1`: load `a0 = A[row0*K+k]`, `a1 = A[(row0+1)*K+k]`,
`b0 = B[k*N+col0]`, `b1 = B[k*N+col0+1]`, then accumulate the outer
product:

$$
\text{acc}_{ij} \mathrel{+}= a_i \cdot b_j \quad \text{for } i,j \in \{0,1\}
$$

Finally write all 4 accumulators to their `C` positions.

## Example

`row0=0, col0=0, K=1`, `A` row `0` is `[3]`, `A` row `1` is `[5]`, `B`
column `0` is `[2]`, `B` column `1` is `[7]`: `a0=3, a1=5, b0=2, b1=7`.
`acc00 = 3*2 = 6`, `acc01 = 3*7 = 21`, `acc10 = 5*2 = 10`,
`acc11 = 5*7 = 35` — the full `2x2` outer product from one shared pair
of loads per row/column.

## What the gate checks

`check.py` seeds a fixed random `8x8 @ 8x8` matmul, launches
`gemm_regblock` with `16` threads (`(M/2)*(N/2)`, one per `2x2` tile),
and compares the result against a numpy `A @ B` oracle. It requires

$$
\mathrm{max\_abs\_err} = \max |{\text{C} - \text{A@B}}| \le 10^{-6}
$$

(It also reports `size_ratio` — the same computation's transaction count
against a fixed 64-thread scalar baseline — for context only, not gated:
this simulator counts memory transactions per program *step*, and
register blocking's real saving is fewer bytes moved per FLOP across
*threads*, which doesn't reduce the per-step segment count the way
shared-memory tiling does. The graded work here is getting the
outer-product accumulation itself right.) A stub that leaves every
accumulator at its default and writes nothing produces `max_abs_err`
in the double digits and fails immediately.
