## Context

When calling a C BLAS library like Accelerate or OpenBLAS from a C++
extension, you must correctly map your high-level tensor dimensions and
strides into a low-level GEMM call. The classic signature is:

```c
void cblas_sgemm(const int Order, const int TransA, const int TransB,
                  const int M, const int N, const int K,
                  const float alpha, const float *A, const int lda,
                  const float *B, const int ldb,
                  const float beta, float *C, const int ldc);
```

With `TransA = CblasTrans`, BLAS treats the physical buffer `A` as a
`K x M` row-major matrix with row stride `lda`, and computes as if using
its transpose `A^T` (`M x K`). This task implements exactly that piece of
arithmetic directly, in real C++: mapping strided, transposed row-major
buffers into the product $C = \alpha A^T B + \beta C$.

## Task

Implement, in `solve.cpp`,

```cpp
void sgemm_at_b(int M, int N, int K, float alpha, float beta,
                 const std::vector<float>& flatA, int lda,
                 const std::vector<float>& flatB, int ldb,
                 std::vector<float>& flatC, int ldc);
```

- `flatA` physically holds a `K x M` row-major matrix with row stride
  `lda` (each physical row has `lda` elements, only the first `M` are
  meaningful). Row `k` starts at `flatA[k * lda]`, so
  `A^T[m][k] = flatA[k * lda + m]`.
- `flatB` physically holds a `K x N` row-major matrix with row stride
  `ldb`: `B[k][n] = flatB[k * ldb + n]`.
- `flatC` physically holds an `M x N` row-major matrix with row stride
  `ldc`, both read (for the `beta * C` term) and written:
  `C[m][n] = flatC[m * ldc + n]`.

Mutate `flatC` in place so that, for every `m` in `[0,M)` and `n` in
`[0,N)`,

$$C[m][n] \leftarrow \alpha \sum_{k=0}^{K-1} A^T[m][k]\, B[k][n] + \beta\, C[m][n]$$

using the **old** value of `C[m][n]` on the right-hand side. Any element of
`flatC` outside that `M x N` logical window (row-stride padding beyond
column `N`) must be left untouched — `ldc` is strictly larger than `N` in
the test, so getting the indexing wrong and touching padding will show up.

## Example

For a physical `A` of shape `K x M` (row stride `lda > M`), `A^T[0][0]`
lives at `flatA[0]`, `A^T[0][1]` at `flatA[lda]` (not `flatA[1]` — that
would be the *next column* of the same physical row, i.e. `A^T[1][0]`).
Getting `lda`/`ldb`/`ldc` confused with the logical dimensions `M`/`N`/`K`
is the exact class of bug this task targets.

## What the gate checks

The fixed driver (`main.cpp`) builds deterministic `flatA`, `flatB`,
`flatC` buffers (row strides strictly larger than the logical dimensions),
calls `sgemm_at_b`, and prints every element of the full physical `flatC`
buffer. The gate is `max_abs_err <= 1e-4` between the printed floats and
the reference's: wrong transpose indexing, swapped `lda`/`ldb`/`ldc`, or
writing into stride padding all move enough values to blow the tolerance.
