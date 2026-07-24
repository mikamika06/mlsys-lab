## Context

When calling BLAS functions like `cblas_dgemm` from a C++ extension, memory
layout is a common source of bugs. Row-major-only GEMM kernels (like the
`row_major_gemm` provided here — a real, self-contained stand-in for
`cblas_dgemm(order=CblasRowMajor, ...)`) take a `TransX` flag per matrix
plus a leading dimension, instead of a global row/column-major order flag.

Suppose you want to compute $C = A B$ where:
- $C$ is $M \times N$, stored **row-major**.
- $B$ is $K \times N$, stored **row-major**.
- $A$ is $M \times K$, but its memory is stored **column-major**.

You can't pass a per-call "order" — only per-matrix transpose flags and
leading dimensions. The trick: a column-major $M \times K$ matrix has the
exact same byte layout as a row-major $K \times M$ matrix, namely $A^T$. So
you tell the kernel "this pointer holds a $K \times M$ row-major matrix,
transpose it for me" — `TransA = CblasTrans`, `lda = M` (the row length of
that $K \times M$ view) — and the kernel's own transpose undoes the
column-major storage back into the correct $A$, with no copying and no
manual transpose loop.

## Task

Fix `gemm_with_col_major_a` in `solve.cpp`:

```cpp
void gemm_with_col_major_a(int M, int N, int K,
                            const double* A_colmajor, const double* B_rowmajor,
                            double* C_rowmajor);
```

Call `row_major_gemm` (declared in `sol.hpp`, implemented in `main.cpp`)
with the transpose flag and leading dimension that correctly interpret
`A_colmajor`'s column-major memory:

$$\texttt{TransA} = \texttt{CblasTrans}, \qquad \texttt{lda} = M$$

`B_rowmajor` is already plain row-major, so `TransB = CblasNoTrans`,
`ldb = N`; the output `ldc = N`.

## Example

For $M=3, K=5$: `A_colmajor[i + j*3]` holds logical element $(i, j)$. Reading
that same buffer row-major with a row length of $3$ and transposing gives
back exactly the $3 \times 5$ matrix $A$ — no copy, no explicit transpose
loop.

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across three `(M, N, K)`
fixtures. The starter calls `row_major_gemm` with `TransA = CblasNoTrans`
and `lda = K` — the naive-looking but wrong call that treats
column-major memory as if it were already row-major — producing a garbled
result that differs from the reference on every fixture.
