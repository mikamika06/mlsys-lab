## Context

BLAS (Basic Linear Algebra Subprograms) is split into levels by how much
work each routine does relative to how much data it touches:

- **Level 1**: vector-vector operations such as `axpy` ($y \gets \alpha x + y$),
  $O(N)$ data, $O(N)$ FLOPs.
- **Level 2**: matrix-vector operations such as `gemv`
  ($y \gets \alpha A x + \beta y$), $O(MN)$ data, $O(MN)$ FLOPs.

Real CBLAS routines (as shipped in Apple's Accelerate framework, and every
other BLAS implementation) take two details seriously:

- **Increments**: `cblas_saxpy(n, alpha, x, incx, y, incy)` does not assume
  `x`/`y` are contiguous. It reads/writes `x[i * incx]` and `y[i * incy]`
  for `i` in `[0, n)`, so a caller can operate on a strided slice of a
  larger buffer without copying it out first.
- **Layout**: `cblas_sgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)`
  needs to know whether `A` is row-major or column-major, and its leading
  dimension `lda` (the stride, in elements, between consecutive rows for
  row-major storage).

This task links the real routines via `dlopen`/`dlsym` at startup (so it
builds with a plain `clang++`, no extra linker flags) and hands you
ready-to-call function pointers.

## Task

Implement

```cpp
void compute_axpy_gemv(int n, float alpha_axpy, const float* x_axpy, int incx, float* y_axpy, int incy,
                        int m, int gemv_n, float alpha_gemv, const float* A, const float* x_gemv,
                        float beta_gemv, float* y_gemv);
```

using the two provided routines (`p_cblas_saxpy`, `p_cblas_sgemv`), never a
hand-written loop:

1. **AXPY**: one call to `p_cblas_saxpy(n, alpha_axpy, x_axpy, incx, y_axpy, incy)`.
   This updates `y_axpy` in place: `y_axpy[i*incy] = alpha_axpy * x_axpy[i*incx] + y_axpy[i*incy]`.
2. **GEMV**: one call to `p_cblas_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, m, gemv_n, alpha_gemv, A, gemv_n, x_gemv, 1, beta_gemv, y_gemv, 1)`.
   This updates `y_gemv` in place: `y_gemv = alpha_gemv * A * x_gemv + beta_gemv * y_gemv`,
   where `A` is `m x gemv_n`, row-major, with leading dimension `gemv_n`.

## Example

```
alpha_axpy = 2.5, x_axpy (stride 2) = {1, 2, 3}, y_axpy = {10, 20, 30}
-> y_axpy = 2.5*{1,2,3} + {10,20,30} = {12.5, 25.0, 37.5}

A (3x4 row-major) = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
x_gemv = {1, 0, -1, 2}, alpha_gemv = 1.5, beta_gemv = 0.5, y_gemv = {100, 200, 300}
row 0: 1*1 + 2*0 + 3*(-1) + 4*2 = 6  -> 1.5*6 + 0.5*100 = 59
```

## What the gate checks

The driver calls `compute_axpy_gemv` once with fixed strided AXPY inputs and
a fixed 3x4 row-major GEMV, and prints the resulting `y_axpy` and `y_gemv`
buffers. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs
it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference — both call the identical real Accelerate BLAS
routines, so a correct implementation reproduces the reference's output
bit-for-bit. Getting `incx`, `lda`, or the row-major/no-transpose flags
wrong changes which elements get read or how the dot products are formed,
and the printed numbers stop matching.
