## Context

For $A \in \mathbb{R}^{m\times n}$ with SVD $A = U\Sigma V^\top$, the Gram
matrix
$$
G = A^\top A = V \Sigma^\top \Sigma V^\top = V \operatorname{diag}(\sigma_1^2,\dots,\sigma_n^2) V^\top
$$
is symmetric positive semi-definite, and its eigendecomposition directly gives
the squared singular values of $A$: the eigenvalues of $G$ are exactly
$\sigma_i^2$ (with $V$ the same right singular vectors). When $A$ is
rectangular, $G$ is $n\times n$ but only $k=\min(m,n)$ of its eigenvalues are
non-zero (up to floating-point noise) — the rest correspond to the null space
of $A$.

## Task

Implement `svd_singular_values`:

```python
def svd_singular_values(A: np.ndarray) -> np.ndarray:
    ...
```

Given `A` of shape $(m,n)$, compute $G = A^\top A$, find its eigenvalues with
`numpy.linalg.eigh` / `eigvalsh` (symmetric eigensolver — never
`numpy.linalg.svd` or `numpy.linalg.eig`), clip any tiny negative
floating-point noise to zero, take the square root, and return the largest
$k=\min(m,n)$ values **sorted descending**. This must match
`numpy.linalg.svd(A, compute_uv=False)` sorted the same way.

## Example

```python
import numpy as np
A = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]])   # shape (3, 2)
svd_singular_values(A)
# -> array([3., 2.])
```

## What the gate checks

`rel_err` — the grader draws several matrices (square, tall, wide, seeded
randomly) and compares your descending-sorted singular values to
`np.linalg.svd(A, compute_uv=False)` sorted the same way, via the global
relative-L2-norm scorer; the worst case across all matrices must satisfy
`rel_err < 1e-6`. Your function is also run under a call tracer that fails the
gate immediately if `numpy.linalg.svd` is ever invoked — the singular values
must come from the eigendecomposition of $A^\top A$, not a shortcut call to
the library SVD.
