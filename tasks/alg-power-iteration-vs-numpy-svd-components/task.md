## Context

Principal Component Analysis (PCA) reduces dimensionality by projecting data onto directions of maximal variance. For a centered data matrix $X \in \mathbb{R}^{n\times d}$ the covariance matrix is $C = X^\top X$. The eigenvectors of $C$ are the principal components; they also appear as the right singular vectors in the Singular Value Decomposition (SVD) $X = U\Sigma V^\top$. NumPy’s `np.linalg.svd` computes these vectors exactly, but it is expensive for very large $d$. A classic alternative is power iteration: repeatedly multiply a random vector by $C$ and renormalise; after enough iterations the vector aligns with the dominant eigenvector. Repeating this process while deflating previously found directions yields an orthonormal basis of the top‑$k$ components.

## Task

Implement `pca_power_iteration(X, k)`:

```python
def pca_power_iteration(X: np.ndarray, k: int) -> np.ndarray:
    ...
```

The function receives a 2‑D NumPy array $X$ of shape $(n,d)$ and an integer $k \le d$. It must return a float64 array of shape $(k,d)$ whose rows are the first $k$ principal components (right singular vectors) of $X$, ordered from largest to smallest eigenvalue. The implementation should use deterministic random seeds so that repeated calls produce identical results.

## Example

```python
import numpy as np
X = np.array([[0, 1], [1, 0], [2, 3]], dtype=float)
components = pca_power_iteration(X, 2)
print(components)
# [[-0.70710678 -0.70710678]
#  [-0.4472136   0.89442719]]
```

The returned vectors are orthonormal and match the first two rows of `np.linalg.svd(X, full_matrices=False)[2][:2]` up to sign.

## What the gate checks

Two aspects are verified:

1. **Numerical accuracy** – The mean per‑component relative error between your result and NumPy’s SVD components (after aligning signs) must be at most $10^{-4}$, measured by `arena.scorers.channel_rel_err`.

2. **Correctness of the algorithm** – Your implementation must actually perform power iteration with deflation; a naive or incorrect approach will produce components that fail the error test.

The grader generates several random matrices and compares your output against NumPy’s SVD reference for each case.
