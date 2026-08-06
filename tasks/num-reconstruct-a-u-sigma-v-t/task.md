## Context

The singular value decomposition of any real matrix $A \in \mathbb{R}^{m\times n}$ is
$$
A = U \Sigma V^\top,
$$
where $U \in \mathbb{R}^{m\times k}$ and $V \in \mathbb{R}^{n\times k}$ have
orthonormal columns and $\Sigma = \operatorname{diag}(\sigma_1,\dots,\sigma_k)$
holds the (non-negative, descending) singular values, with $k=\min(m,n)$ for
the *economy* (reduced) decomposition.

`linalg.svd(A, full_matrices=False)` returns `U` of shape $(m,k)$, a
1-D array `s` of length $k$ (just the diagonal of $\Sigma$, not the full
matrix), and `Vt` of shape $(k,n)$ — already $V^\top$, not $V$. A common bug
is forgetting that `s` needs to become a diagonal matrix before the matrix
product, or reconstructing with the wrong shape when $A$ is rectangular
($m \neq n$).

## Task

Implement `reconstruct_from_svd`:

```python
def reconstruct_from_svd(A: list[list[float]]) -> list[list[float]]:
    ...
```

`A` is a real 2-D array of shape $(m, n)$ — not necessarily square. Compute
its (economy) SVD and return
$$
\hat A = U \operatorname{diag}(\sigma) V^\top,
$$
an array with **exactly the same shape as `A`**, reconstructed to within
floating-point precision.

## Example

```python
A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # shape (3, 2)
Ahat = reconstruct_from_svd(A)
Ahat.shape        # (3, 2)
max(abs(x - y) for row1, row2 in zip(Ahat, A) for x, y in zip(row1, row2)) # ~1e-15
```

## What the gate checks

`max_abs_err` — the grader reconstructs five matrices with your function
(square, tall $m>n$, wide $m<n$, and a $2\times2$ hand-picked case) and takes
the worst-case elementwise absolute error between your output and the
**original matrix `A` itself** (not another library's decomposition — the SVD
reconstruction identity is exact up to floating-point round-off). A shape
mismatch, a missing `diag`, or dropping/transposing `Vt` incorrectly produces
a large error. Gate: `< 1e-8`.
