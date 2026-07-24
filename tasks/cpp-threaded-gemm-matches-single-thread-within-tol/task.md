## Context

General matrix-multiply (GEMM) is the workhorse of numerical and ML code:
given row-major matrices $A \in \mathbb{R}^{M \times K}$ and
$B \in \mathbb{R}^{K \times N}$, compute

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \, B_{kj}, \qquad C \in \mathbb{R}^{M \times N}.$$

A common first optimization is to parallelize the outer row loop: hand each
thread a contiguous block of rows of $C$ to compute. Because every output
element $C_{ij}$ depends on exactly one row of $A$ and one column of $B$,
row blocking keeps each element's whole computation inside a single thread.
Done correctly there are no data races and no cross-thread accumulation, so
the answer is **deterministic and independent of the thread count** — a
threaded run must reproduce the single-thread result.

## Task

Implement the contract in `sol.hpp`:

```cpp
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, int num_threads);
```

Compute $C = A B$ in row-major layout (`A[i*K+k]`, `B[k*N+j]`, `C[i*N+j]`).
`C` is preinitialized to `0` by the caller. You may split the `M` rows into
up to `num_threads` contiguous blocks and run each block on its own
`std::thread`, but the result must be the same for **any** `num_threads >= 1`.

## Example

For $A = \begin{bmatrix} 1 & 2 \\ 0 & 3 \end{bmatrix}$,
$B = \begin{bmatrix} 4 & 5 \\ 6 & 7 \end{bmatrix}$:

$$C = A B = \begin{bmatrix} 16 & 19 \\ 18 & 21 \end{bmatrix}.$$

The driver builds fixed integer-valued $16 \times 24$ and $24 \times 16$
inputs, calls `gemm` with `num_threads` = 1, 2, and 4, and for each run prints
the full $C$ matrix followed by its checksum. A correct implementation prints
identical numbers for every thread count.

## What the gate checks

The driver `main.cpp` is compiled with your `gemm` and, separately, with the
reference `gemm`, both under `clang++ -O2 -std=c++20`. The printed numbers are
compared element-wise. The gate passes when

$$\text{max\_abs\_err} = \max_{p} \lvert y^{\text{ref}}_p - y^{\text{you}}_p \rvert \le 10^{-6}$$

across every printed value (all `C` entries and checksums, for all three
thread counts). The inputs are integer-valued so exact GEMM is representable
in `float`; a correct implementation matches the reference exactly, while a
result that varies with `num_threads` (a race) or is wrong fails the gate.
