## Context

Principal component analysis (PCA) seeks a low‑dimensional linear subspace that captures as much variance of the data as possible. For an $n\times d$ data matrix $\mathbf X$, we first centre it by subtracting its column means, obtaining $\tilde{\mathbf X}$. The covariance matrix is
$$\mathbf C = \frac{1}{\,n-1\,}\,\tilde{\mathbf X}^{\mathsf T}\tilde{\mathbf X}.$$

The eigenvectors of $\mathbf C$ corresponding to the $k$ largest eigenvalues form an orthonormal basis for the best rank‑$k$ approximation in the least‑squares sense. If we denote these eigenvectors by $\mathbf V_k \in \mathbb R^{d\times k}$, then the projection of a centred sample $\tilde{\mathbf x}_i$ onto this subspace is
$$\mathbf y_i = \mathbf V_k^{\mathsf T}\,\tilde{\mathbf x}_i,$$
and the reconstruction back to the original space is
$$\hat{\mathbf x}_i = \mathbf V_k\,\mathbf y_i.$$

The full reconstruction of $\mathbf X$ can be written compactly as
$$\hat{\mathbf X} = \tilde{\mathbf X}\,\mathbf V_k\,\mathbf V_k^{\mathsf T} + \bar{\mathbf x},$$
where $\bar{\mathbf x}$ is the vector of column means.

## Task

Implement `rank_k_project_reconstruct(X, k)`:

```python
def rank_k_project_reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    ...
```

It takes a 2‑D NumPy array of shape $(n,d)$ and an integer $k$ ($1\le k \le d$). The function must return the reconstruction $\hat{\mathbf X}$ as a float64 NumPy array of shape $(n,d)$. You may use any NumPy linear‑algebra routine (e.g. `np.linalg.svd`) but you **must** centre the data before computing the components and add the mean back after reconstruction.

## Example

```python
import numpy as np
X = np.array([[0, 1], [2, 3], [4, 5]], dtype=float)
k = 1
X_hat = rank_k_project_reconstruct(X, k)
print(np.round(X_hat, 3))
# [[-0.707 -0.707]
#  [ 0.   0.   ]
#  [ 0.707  0.707]]
```

## What the gate checks

The grader computes a reference reconstruction using NumPy’s SVD and compares it to your output with the mean‑squared error scorer from `arena.scorers`. The metric `mse` must satisfy
$$\mathrm{mse} \le 10^{-4}.$$
Additionally, the returned array must have dtype `float64` and the same shape as the input.
