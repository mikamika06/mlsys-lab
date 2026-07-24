## Context

The singular value decomposition (SVD) of a real matrix $A \in \mathbb{R}^{n\times d}$ writes

$$A = U\,\Sigma\,V^\top,$$

where $U$ and $V$ are orthogonal and $\Sigma=\operatorname{diag}(\sigma_1,\dots,\sigma_k)$ with $k=\min(n,d)$. The singular values $\sigma_i \ge 0$ quantify the energy of each principal component. If we square them, $\sigma_i^2$, they represent the variance captured by that component in a PCA setting.

The cumulative proportion of total variance explained after keeping the first $i$ components is

$$
p_i = \frac{\sum_{j=1}^{i}\sigma_j^{\,2}}{\sum_{j=1}^{k}\sigma_j^{\,2}},
\qquad i=1,\dots,k.
$$

This sequence starts at a value between 0 and 1 and monotonically increases to 1.

## Task

Implement `read_singular_values_variance_explained(A)`:

```python
def read_singular_values_variance_explained(A: np.ndarray) -> np.ndarray:
    ...
```

It receives a two‑dimensional NumPy array $A$ of shape $(n,d)$ and returns a one‑dimensional `float64` array containing the cumulative variance explained ratios $p_1,\dots,p_k$. The function must use only NumPy (no explicit Python loops) and should be robust to degenerate matrices.

## Example

```python
import numpy as np
A = np.array([[1, 0], [0, 1]])
# singular values are both 1.0
# variance explained per component: [0.5, 0.5]
# cumulative: [0.5, 1.0]
p = read_singular_values_variance_explained(A)
print(p)   # [0.5 1. ]
```

## What the gate checks

The grader computes a reference implementation using `numpy.linalg.svd` and compares your output with it via the relative L2 error

$$
\mathrm{rel\_err} = \frac{\lVert p_{\text{cand}}-p_{\text{ref}}\rVert}{\lVert p_{\text{ref}}\rVert}.
$$

The gate requires $\mathrm{rel\_err}\le 10^{-9}$.
