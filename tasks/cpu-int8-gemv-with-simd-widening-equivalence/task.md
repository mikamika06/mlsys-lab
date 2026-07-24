## Context

Single‑Instruction‑Multiple‑Data (SIMD) units in modern CPUs can multiply and
accumulate several 8‑bit elements in parallel while widening to 32‑bit integer
lanes before summation. In mathematical form, for a matrix
$A \in \mathbb{Z}_8^{m \times n}$ and a vector $x \in \mathbb{Z}_8^n$ the
widening GEMV computes

$$
y_i = \sum_{j=1}^{n} A_{ij} \, x_j , \quad i = 1,\dots,m,
$$

with each product $A_{ij} x_j$ promoted to a 32‑bit integer before accumulation.
This mirrors the SIMD *dot‑product widening* primitive in NEON/AVX. The final
result $y \in \mathbb{Z}_{32}^m$ must be identical to the mathematical 32‑bit
sum over 8‑bit inputs — not affected by wraparound or float rounding.

## Task

Implement `int8_gemv(A, x)`:

```python
import numpy as np

def int8_gemv(A: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ...
```

It receives

* `A`: a 2‑D NumPy array of shape $(m, n)$ with dtype `int8`
* `x`: a 1‑D NumPy array of length $n$ with dtype `int8`

Return a tuple `(y, access)` where

* `y` is an `int32` vector of shape $(m,)` containing the correctly widened GEMV
  result;
* `access` is a **list of integer byte addresses** the kernel would have touched
  when scanning `A` and `x` in linear memory order. Assume that both arrays are
  contiguous in row‑major order and use `itemsize=1` for each `int8` element.

The logical byte address of element `[i, j]` in `A` is
$$
\mathrm{addr}_A(i,j) = \mathrm{base}_A + i n + j,
$$
and of element `[j]` in `x` is
$$
\mathrm{addr}_x(j) = \mathrm{base}_x + j + m n,
$$
where `base_A = 0`. The separation of bases guarantees unique addresses across
`A` and `x`.

You must record every load access in this deterministic model in sequence,
reflecting the traversal order of your algorithm.

## Example

```python
import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int8)
x = np.array([1, -1, 2], dtype=np.int8)

y, acc = int8_gemv(A, x)
print(y)
# [1*1 + 2*(-1) + 3*2, 4*1 + 5*(-1) + 6*2] -> [5, 11]
print(acc[:10])  # first few byte addresses touched
```

## What the gate checks

Two gates:

1. **Value correctness** — the 32‑bit output must be byte‑identical to the
   mathematical reference. The gate measures $\max |y_{\mathrm{your}} - y_{\mathrm{ref}}|$ ,
   which must be exactly $0$.
2. **Cache friendliness** — the access trace is fed to the deterministic cache simulator
   (`arena.cachesim`), pinned to a $4$‑way set associative L1 with 64‑byte lines and 64 sets.
   The resulting *miss rate* must not exceed the reference rate produced by a cache‑friendly
   row‑major traversal. Passing this gate means `miss_rate_ok == 1.0`.
