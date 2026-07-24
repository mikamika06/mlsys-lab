## Context

Principal Component Analysis (PCA) reduces dimensionality by projecting data onto orthogonal directions that capture most variance. For a centered data matrix $X \in \mathbb{R}^{n\times d}$, the covariance matrix is $C = X^\top X / n$. The eigen‑decomposition of $C$ yields principal axes. An equivalent and numerically stable route uses Singular Value Decomposition (SVD). If

$$
X = U\,\Sigma\,V^\top,
$$

then the columns of $V$ are the orthonormal principal directions, $\Sigma$ contains singular values proportional to variances, and $U$ holds the coordinates in component space. Projecting onto the first $k$ components gives a low‑rank representation:

$$
X_{\text{proj}} = X\,V_k^\top,
$$

where $V_k \in \mathbb{R}^{d\times k}$ contains the leading columns of $V$.

## Task

Implement `pca_svd(X, k)` that returns the projection of every row of a 2‑D NumPy array `X` onto its first `k` principal components using SVD. The function must:

1. Subtract the column mean from `X`.
2. Compute the economy‑size SVD of the centered matrix.
3. Return the projected data as an `(n, k)` array of type `float64`.

The signature is

```python
def pca_svd(X: np.ndarray, k: int) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np
X = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
proj = pca_svd(X, 1)
print(proj)
# [[-2.82842712]
#  [ 0.        ]
#  [ 2.82842712]]
```

## What the gate checks

The grader computes a reference projection with NumPy’s `linalg.svd` on the centered data and compares it to your output using the scorer `channel_rel_err`. Because each principal component is defined only up to sign, the grader first aligns signs of corresponding columns before computing the error. Your implementation must achieve

$$
\text{channel\_rel\_err} \le 10^{-6}.
$$

A correct solution will produce a matrix that differs from the reference by at most $10^{-6}$ relative per‑component error.
