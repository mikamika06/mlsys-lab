## Context

Classical Multidimensional Scaling (MDS) reconstructs coordinates from a matrix of pairwise squared Euclidean distances.  
Given $D^2 \in \mathbb{R}^{n\times n}$ with entries  

$$
D_{ij}^2 = \lVert x_i - x_j\rVert^2,
$$

the algorithm proceeds by double‑centering to obtain a Gram matrix $B$ and then eigen‑decomposing it.

$$
J = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^{\top}, \qquad
B = -\tfrac12 J D^2 J.
$$

The top–$k$ eigenvalues $\lambda_1,\dots,\lambda_k$ and corresponding eigenvectors $v_1,\dots,v_k$ give the embedding:

$$
X = [\, v_1 \sqrt{\lambda_1} \;\; \cdots \;\; v_k \sqrt{\lambda_k}\,].
$$

The resulting coordinates are defined up to an orthogonal transformation; any rotation or reflection yields the same distance matrix.

## Task

Implement `mds_from_distances(D2: np.ndarray, k: int) -> np.ndarray` that takes a 2‑D NumPy array of shape $(n,n)$ containing squared Euclidean distances and returns an $n\times k$ array of coordinates in $\mathbb{R}^k$.  
The implementation must use only NumPy operations; no explicit Python loops.  
The output should be `float64`.

## Example

```python
import numpy as np
X_true = np.array([[0, 0], [1, 0], [0, 2]])
# compute squared distances
D2 = np.sum((X_true[:,None,:] - X_true[None,:,:])**2, axis=2)
X_est = mds_from_distances(D2, 2)
print(X_est)   # close to a rotated version of X_true
```

## What the gate checks

Two gates. The relative error $\mathrm{rel\_err}$ between the pairwise squared distances computed from your embedding and the input matrix must satisfy  
$\displaystyle \mathrm{rel\_err}\le 10^{-6}$.  
No explicit loops are allowed; a vectorized solution is required.
