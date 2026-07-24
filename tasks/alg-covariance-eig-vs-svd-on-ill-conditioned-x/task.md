## Context

Principal Component Analysis (PCA) reduces the dimensionality of a dataset by projecting onto directions that capture the most variance.  
Given data $X \in \mathbb{R}^{n\times d}$, one common approach is to compute the covariance matrix  

$$C = \frac{1}{n-1}\,(X-\bar X)^{\!\top}(X-\bar X),$$

and then solve the eigenvalue problem $C\,v=\lambda v$.  
The eigenvectors $v$ corresponding to the largest eigenvalues form an orthonormal basis for the principal subspace.

An alternative, numerically more stable route is to perform a singular value decomposition (SVD) on the centered data:

$$X-\bar X = U\,\Sigma\,V^{\!\top},$$

where $V$ contains the right singular vectors.  The columns of $V$ are identical to the eigenvectors of $C$, but the SVD is robust even when $C$ is ill‑conditioned or nearly rank deficient.

Ill‑conditioning arises when some columns of $X$ are almost linear combinations of others, causing $C$ to have very small eigenvalues and magnifying rounding errors in an eigen decomposition.  In such cases a naive covariance‑eigen approach can produce wildly inaccurate components, whereas the SVD remains accurate.

## Task

Implement the function

```python
def cov_eig_vs_svd_pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `X` is a 2‑D NumPy array of shape $(n,d)$ with dtype `float64`.  
* `k` satisfies $1 \le k \le \min(n,d)$.  
* The function must return a tuple `(eig_vecs, svd_vecs)` where:
  * `eig_vecs` is an array of shape $(k,d)$ containing the first $k$ eigenvectors of the covariance matrix (ordered by decreasing eigenvalue).  
  * `svd_vecs` is an array of shape $(k,d)$ containing the first $k$ right singular vectors from a full SVD on centered data.  

Both arrays must be of dtype `float64`.  No explicit Python loops are required, but they are allowed if you wish.

## Example

```python
import numpy as np
X = np.array([[0., 0.], [1., 0.], [0., 2.]])
k = 2
eig_vecs, svd_vecs = cov_eig_vs_svd_pca(X, k)
print(eig_vecs.shape)   # (2, 2)
print(svd_vecs.shape)   # (2, 2)
```

The two returned arrays should be very close to each other; the first row of `eig_vecs` should match the first row of `svd_vecs` up to a sign.

## What the gate checks

The grader computes a reference solution by performing an SVD on the centered data and taking its first $k$ right singular vectors.  
It then compares the candidate’s covariance‑eigen vectors against this reference using the metric  

$$\mathrm{max\_abs\_err} = \max_{i}\;\lVert \text{eig\_vecs}_i - \text{ref\_vecs}_i\rVert_{\infty}.$$

The gate requires $\mathrm{max\_abs\_err} \le 10^{-6}$.  
If the shapes or dtypes differ, the candidate fails automatically.  The second array returned by the function is not graded but must be present and of correct shape.
