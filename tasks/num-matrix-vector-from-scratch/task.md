## Context

For a matrix $A \in \mathbb{R}^{N \times M}$ and a vector $x \in \mathbb{R}^{M}$,
the matrix-vector product $y = Ax$ has entries

$$
y_i = \sum_{j=0}^{M-1} A_{ij}\, x_j, \qquad i = 0, \dots, N-1 .
$$

Each output entry is a dot product between one row of $A$ and $x$. Computing
this with a library call such as `np.dot` or `A @ x` hides the two nested
loops (over $i$ and over $j$) that actually do the work — the goal here is to
write those loops explicitly.

## Task

Implement `matvec_from_scratch(A, x)`:

```python
def matvec_from_scratch(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    ...
```

`A` has shape $(N, M)$ and `x` has shape $(M,)$, both `float64`. Return the
length-$N$ product $y = Ax$ as a `float64` array.

Compute it with two **explicit nested Python `for` loops** — an outer loop
over rows $i$ and an inner loop over columns $j$ accumulating
$y_i \mathrel{+}= A_{ij} x_j$. Do not call `np.dot`, `np.matmul`, the `@`
operator, `np.sum` over a row, or any other single-call reduction — the loops
must do the summation themselves.

## Example

```python
import numpy as np
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x = np.array([1.0, 0.0, -1.0])
matvec_from_scratch(A, x)
# [-2. -2.]
```

## What the gate checks

Two gates. The result must match a NumPy reference (`A @ x` in float64) with
$\mathrm{max\_abs\_err} \le 10^{-6}$. A Python-level line tracer also counts
how many source lines execute inside your function across all test cases
(`op_count`); the gate requires $\mathrm{op\_count} \ge 100$. A vectorized
call such as `A @ x` runs almost entirely in C and emits only a handful of
Python line events, so it fails this gate even though its output is correct
— only a genuine double Python loop passes both.
