## Context

Matrix multiplication computes

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}, \qquad A \in \mathbb{R}^{N \times K},\; B \in \mathbb{R}^{K \times M} .$$

The result does not depend on the order the three nested loops are written in,
but the memory access pattern does. For row-major arrays, the **ikj** order

$$
\text{for } i: \quad \text{for } k: \quad \text{for } j: \quad C_{i,\cdot} \mathrel{+}= A_{ik}\, B_{k,\cdot}
$$

turns the inner loop into a row-streaming update: $A_{ik}$ is a scalar that is
loop-invariant across $j$, and both $B_{k,\cdot}$ and $C_{i,\cdot}$ are
accessed contiguously, one full row at a time. This is the "saxpy" view of
matmul — each step of the $(i,k)$ pair adds a scaled row of $B$ into a row of
$C$. Contrast this with the naive **ijk** order, which for every fixed $(i,j)$
walks down an entire column of $B$ with a large stride, touching a new cache
line on almost every step.

## Task

Implement `matmul_ikj(A, B)`:

```python
def matmul_ikj(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

`A` has shape $(N, K)$ and `B` has shape $(K, M)$, both `float64`. Return the
$(N, M)$ product $C = AB$ as a `float64` array.

Compute the product with three **explicit nested Python `for` loops in i, k,
j order**, accumulating into `C[i, j]` (or a whole row `C[i, :]`) as you go.
Do not call `np.dot`, `np.matmul`, the `@` operator, or any other single-call
BLAS routine — the point of the exercise is the loop order itself, not the
final numeric answer.

## Example

```python
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
matmul_ikj(A, B)
# [[19. 22.]
#  [43. 50.]]
```

## What the gate checks

Two gates. The result must match a NumPy reference (`A @ B` in float64) with
$\mathrm{max\_abs\_err} \le 10^{-6}$. A Python-level line tracer also counts
how many source lines execute inside your function across all test cases
(`op_count`); the gate requires $\mathrm{op\_count} \ge 200$. A vectorized
one-liner such as `A @ B` runs almost entirely in C and emits only a handful
of Python line events, so it fails this gate even though its output is
correct — only a genuine three-deep Python loop passes both.
