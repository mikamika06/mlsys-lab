## Context

When calling optimized BLAS routines like `cblas_dgemm` from C++, you often want
to multiply submatrices without copying the data into contiguous buffers. BLAS
supports this via the `lda`, `ldb`, `ldc` (leading dimension) parameters,
which specify the element stride between consecutive rows (row-major layout)
in the original memory buffer, and a starting element offset for each
submatrix.

Consider a large 1D buffer `W` representing a 2D matrix in row-major order.
We want $C = A \times B$, where $A$ ($M \times K$), $B$ ($K \times N$) and $C$
($M \times N$) are all submatrices living inside the same `W`. The real
compiler's own layout for the descriptor struct that would carry these
parameters to a native kernel:

```cpp
struct GemmDescriptor {
    int m, n, k;
    int lda, ldb, ldc;
    long offset_a, offset_b, offset_c;
};
```

Six `int`s (24 bytes) are already a multiple of the 8-byte alignment `long`
needs, so no padding is inserted before the three `long`s: `sizeof(GemmDescriptor) == 48`.

## Task

Implement `submatrix_gemm_params` (declared in `sol.hpp`):

```cpp
GemmParams submatrix_gemm_params(int M, int N, int K, int W_cols,
                                  int a_row, int a_col, int b_row, int b_col,
                                  int c_row, int c_col);
```

Given the parent matrix's column count `W_cols` and each submatrix's
top-left corner (`a_row, a_col`), (`b_row, b_col`), (`c_row, c_col`), compute:

- `lda`, `ldb`, `ldc` — the row stride (in elements) for each submatrix.
  Because $A$, $B$, $C$ all live inside the SAME parent buffer `W`, every
  leading dimension equals `W_cols`, regardless of the submatrix's own shape.
- `offset_a`, `offset_b`, `offset_c` — the flat element index of each
  submatrix's top-left corner: `offset = row * W_cols + col`.

The fixed driver (`main.cpp`) then performs the real GEMM `C = A * B` in
place inside a deterministic `W` buffer, reading and writing **only** through
your `lda`/`ldb`/`ldc` + offsets (never the raw `(row, col)` pairs) — wrong
parameters read or write the wrong elements.

## Example

```cpp
submatrix_gemm_params(3, 4, 5, 20, 1, 2, 5, 6, 10, 11)
// -> lda=20 ldb=20 ldc=20 offset_a=22 offset_b=106 offset_c=211
```

For the driver's fixed scenario (`W` is $20\times20$, same M/N/K/corners as
above) the correct run prints:

```
sizeof=48
20 20 20 22 106 211
checksum=3030.680000
```

A starter that returns all-zero parameters collapses `A`, `B` and `C` onto
the same few elements of `W` — including writing `C` back on top of the very
values `A` is still reading, so later dot products consume already-corrupted
data and the checksum blows up to an astronomically wrong number instead of
`3030.68`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, extracts every printed number, and requires `max_abs_err <= 1e-6`
against the same driver linked with `ref.cpp`. `sizeof(GemmDescriptor)` is
identical for both (it's a fixed struct, not something you compute), so the
gate lives or dies on the six `lda/ldb/ldc/offset_a/offset_b/offset_c`
integers and the resulting GEMM checksum.
