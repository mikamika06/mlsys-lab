## Context

The singular value decomposition (SVD) of a real matrix $A \in \mathbb{R}^{n \times n}$ is

$$A = U \Sigma V^\top,$$

where $U$ and $V$ are orthogonal matrices and $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_n)$ with $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_n \ge 0$.

A classical approach avoids computing both $U$ and $V$ simultaneously. The *one-sided Jacobi SVD* works only on $A$ itself, iteratively applying Givens rotations to its columns. The key observation is that the Gram matrix $G = A^\top A$ has eigenvalues $\sigma_k^2$; diagonalizing $G$ by Jacobi rotations simultaneously yields the singular values.

At each step, the algorithm picks the largest off-diagonal element $G_{ij}$ and applies a rotation angle $\theta$ chosen so that the new $G_{ij}' = 0$. The rotation angle is

$$\theta = \frac{1}{2}\arctan\!\left(\frac{2\,G_{ij}}{G_{ii} - G_{jj}}\right),$$

and the corresponding rotation matrix $J(i,j,\theta)$ simultaneously:

1. zeros out $G_{ij}$ and $G_{ji}$ in the Gram matrix, and
2. rotates columns $i$ and $j$ of $A$, preserving the left singular vectors implicitly.

After convergence, the diagonal of $G$ holds $\sigma_k^2$ and the accumulated rotation matrix $V$ holds the right singular vectors.

## Task

Implement `one_sided_jacobi_svd(A)`:

```python
def one_sided_jacobi_svd(A: np.ndarray, tol: float = 1e-12, max_iter: int = 1000) -> np.ndarray:
    """
    Compute singular values of A via the one-sided Jacobi SVD algorithm.

    Parameters
    ----------
    A : np.ndarray of shape (n, n), square real matrix.
    tol : float, convergence tolerance on off-diagonal Frobenius norm.
    max_iter : int, maximum number of Jacobi sweeps.

    Returns
    -------
    singular_values : np.ndarray of shape (n,), sorted in descending order.
    """
```

Use the standard Jacobi eigenvalue formulas. At each iteration, find the pair $(i, j)$ with the largest $|G_{ij}|$, compute

$$\tau = \frac{G_{jj} - G_{ii}}{2\,G_{ij}}, \quad
t = \frac{\operatorname{sign}(\tau)}{|\tau| + \sqrt{1 + \tau^2}}, \quad
c = \frac{1}{\sqrt{1+t^2}}, \quad s = t\cdot c,$$

and apply the rotation to columns $i,j$ of $A$ and to $G$. The singular values are $\sigma_k = \sqrt{G_{kk}}$ after convergence.

## Example

```python
import numpy as np
A = np.array([[3., 1.],
              [1., 3.]])
sigma = one_sided_jacobi_svd(A)
# sigma ≈ [3.16227766, 2.82842712]   (sorted descending)
# reference: np.linalg.svd(A, compute_uv=False) gives the same values
```

## What the gate checks

The gate computes the reference singular values via `numpy.linalg.svd(A, compute_uv=False)` and sorts both the reference and the candidate in descending order. The relative error

$$\text{rel\_err} = \frac{\lVert \sigma_{\text{cand}} - \sigma_{\text{ref}} \rVert}{\lVert \sigma_{\text{ref}} \rVert + \epsilon}$$

must satisfy $\text{rel\_err} < 10^{-8}$. The gate uses four fixed test matrices of sizes $3 \times 3$ through $6 \times 6$, plus one random $6 \times 6$ matrix seeded deterministically, covering both well-conditioned and moderately ill-conditioned cases.
