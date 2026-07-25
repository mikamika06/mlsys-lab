## Context

The simplest matmul kernel gives one thread per output element: at
every step `k`, it loads exactly one value from `A` and one from `B`,
multiplies, accumulates. That's 2 loads per multiply-add — but two
*different* output elements in the same row share the exact same `A`
value at step `k`, and two output elements in the same column share the
exact same `B` value. One-thread-per-output throws that sharing away:
every thread reloads it independently.

**Thread coarsening** has each thread own a small `C x C` tile of
outputs instead of one. For a `2x2` tile, at every `k` the thread loads
2 values from `A` (its 2 output rows) and 2 from `B` (its 2 output
columns) — 4 loads total — and computes all 4 products by reusing each
of those loaded values across the 2 outputs that need it: the `A` value
for row 0 feeds *both* of row 0's outputs; the `B` value for column 0
feeds *both* of column 0's outputs. 4 loads produce 4 results instead
of 4 independent single-output threads needing 8.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void coarsened_matmul(const float* A, const float* B, float* C, int N);
```

`N` is even. `tiles_per_row = N / 2`; thread `idx` (linear id) owns
output rows `row0 = (idx / tiles_per_row) * 2`, `row1 = row0 + 1` and
columns `col0 = (idx % tiles_per_row) * 2`, `col1 = col0 + 1`. For each
`k` from `0` to `N-1`, load `a0 = A[row0][k]`, `a1 = A[row1][k]`,
`b0 = B[k][col0]`, `b1 = B[k][col1]` and accumulate:
`acc00 += a0*b0`, `acc01 += a0*b1`, `acc10 += a1*b0`, `acc11 += a1*b1`.
After the loop, write the 4 accumulators to `C[row0][col0]`,
`C[row0][col1]`, `C[row1][col0]`, `C[row1][col1]`.

## Example

`N=4`, `idx=0`: `tiles_per_row=2`, `row0=0, row1=1, col0=0, col1=1` —
this thread computes `C`'s entire top-left `2x2` block. `idx=1`:
`row0=0, row1=1, col0=2, col1=3` — the top-right `2x2` block.

## What the gate checks

The grader launches `coarsened_matmul` on fixed `8x8` random matrices
and compares the result against an exact `A @ B` (numpy). It requires

$$
\mathrm{rel\_err} \le 10^{-9}
$$

Every element of `C` still has to come out exactly right — coarsening
changes how many times each input value gets loaded, not the math
itself, so a correct implementation matches `A @ B` to within ordinary
floating-point precision.
