## Context

For a real symmetric matrix $A \in \mathbb{R}^{n\times n}$ the eigenvalue decomposition is
$$
A = V\,\Lambda\,V^\top,
$$
where $\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)$ contains the eigenvalues and $V$ is an orthogonal matrix whose columns are the corresponding eigenvectors.  
In many applications only the leading $k$ eigenpairs are required, for example in Principal Component Analysis (PCA) or spectral clustering.

A common way to extract these top‑$k$ components is **deflation**: after computing one dominant eigenpair $(\lambda_i,v_i)$ we remove its contribution from the matrix and repeat.  The power iteration method can be used to find each successive eigenvector, but for this task you may use any NumPy routine that respects the contract below.

## Task

Implement a function with the following signature:

```python
def topk_deflation(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `A` is a real symmetric matrix of shape `(n, n)` and type `float64`.
* `k` satisfies `1 <= k < n`.

The function must return:

1. An array `eigvals` of shape `(k,)` containing the $k$ largest eigenvalues in **descending** order.
2. A matrix `eigvecs` of shape `(n, k)` whose columns are the corresponding orthonormal eigenvectors.

Both outputs should be of type `float64`.  The implementation may use any NumPy linear‑algebra routine; loops are allowed but not required.

## Example

```python
import numpy as np

A = np.array([[2., 1.],
              [1., 3.]])
eigvals, eigvecs = topk_deflation(A, 2)

print(eigvals)
# [4.23606798 0.76393202]

print(eigvecs)
# [[-0.52573111 -0.85065081]
#  [-0.85065081  0.52573111]]
```

## What the gate checks

The grader computes a reference solution using `numpy.linalg.eigh` on a set of random symmetric matrices.  
Three metrics are reported:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `rel_err_eig` | $\displaystyle \max_i \frac{|\lambda^{\text{cand}}_i-\lambda^{\text{ref}}_i|}
{\max_j |\lambda^{\text{ref}}_j|}$ | $1\times10^{-4}$ |
| `vec_align_err` | For each candidate eigenvector the maximum relative error after aligning its sign with the closest reference vector:  
$\displaystyle \max_i \frac{\lVert v^{\text{cand}}_i-\operatorname{sign}(v^{\text{cand}}_i^\top r_j)\,r_j\rVert}
{\lVert r_j\rVert}$ | $1\times10^{-4}$ |
| `orthogonality_err` | Frobenius norm of the deviation from orthonormality:  
$\displaystyle \lVert V_{\text{cand}}^\top V_{\text{cand}}-I_k\rVert_F$ | $1\times10^{-6}$ |

All three metrics must be **less than or equal** to their thresholds for the solution to pass.  The reference is recomputed on each run, so hard‑coded expected values are not used.
