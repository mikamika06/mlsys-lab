## Context

A highly optimized GEMM (General Matrix Multiply) achieves its speed largely
by **cache blocking** (tiling): instead of a naive triple-nested loop that
streams through the whole of `A`, `B`, and `C` for every output element,
work is split into small `block_size x block_size` tiles that fit in L1/L2
cache. Each tile's worth of data is reused many times while it's hot, so the
same total amount of arithmetic causes far fewer cache misses.

The mathematical operation being computed — regardless of loop order or
blocking — is:

$$
C \;\leftarrow\; \alpha \, (A \, B) \;+\; \beta \, C
$$

with `A` an $M \times K$ matrix, `B` a $K \times N$ matrix, and `C` an
$M \times N$ matrix, all row-major and contiguous in memory.

## Task

Implement

```cpp
void blocked_gemm(const float* A, const float* B, float* C,
                   int M, int N, int K, int block_size,
                   float alpha, float beta);
```

using a **blocked** triple loop over tiles of size `block_size` in each of
the $M$, $N$, $K$ dimensions:

1. Scale `C` by `beta` up front (`C[i*N+j] *= beta` for every element).
2. For each tile `(ii, jj, kk)` of the $M \times N \times K$ iteration space
   (stepping by `block_size` in each dimension), accumulate
   `alpha * A[i][k] * B[k][j]` into `C[i][j]` for every `(i, j, k)` inside
   that tile.

`M`, `N`, and `K` are **not required to be multiples of `block_size`** — the
last tile along each dimension is simply shorter; clamp each tile's upper
bound to the matrix dimension instead of assuming a full-size tile
everywhere.

## Example

For `block_size = 4` and `M = 10`, the row/tile boundaries along $M$ are
`[0,4)`, `[4,8)`, `[8,10)` — the last tile has only 2 rows, not 4. A kernel
that always loops `i` from `ii` to `ii + block_size` without clamping to `M`
reads and writes past the end of the last tile.

## What the gate checks

The driver multiplies a fixed $10 \times 7$ by $7 \times 9$ pair (deliberately
not multiples of `block_size = 4`, so a mishandled edge tile shows up as
wrong values specifically in the last rows/columns) with `alpha = 1.5`,
`beta = 0.5`, and prints the resulting $10 \times 9$ `C`. The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{max\_abs\_err} = \max_{i,j} \bigl| \hat C_{ij} - C_{ij} \bigr| \;\le\; 10^{-3}
$$

against the reference (a tolerance, not exact match, since summing the same
terms in a different tile order changes float rounding slightly but not the
mathematical result).
