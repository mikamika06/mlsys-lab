## Context

Principal component analysis (PCA) finds directions of maximum variance by using the singular value decomposition (SVD). For a centered data matrix $X \in \mathbb{R}^{n \times d}$, the SVD is

$$
X = U \Sigma V^\top ,
$$

where the rows of $V^\top$ are ordered by decreasing singular value. The first $k$ principal directions are the first $k$ rows of $V^\top$.

The projection of the samples into the first $k$ principal components is

$$
Z = X V_k^\top ,
$$

where $V_k^\top$ contains the first $k$ rows of $V^\top$. A common debugging mistake is to swap $U$ and $V^\top$, transpose the wrong factor, or keep components in ascending singular-value order. These errors produce projections with the wrong channels.

## Task

Implement `pca_projection(X, k)`:

```python
def pca_projection(X: np.ndarray, k: int) -> np.ndarray:
    ...
```

The function receives a 2-D floating point array and an integer $k$. Return the projection of every row of $X$ onto the first $k$ principal components.

Use NumPy SVD. The returned array must have shape $(n, k)$ and contain the projected coordinates in descending explained-variance order.

## Example

```python
import numpy as np

X = np.array([
    [2.0, 0.0],
    [0.0, 1.0],
    [-2.0, 0.0],
])

Z = pca_projection(X, 1)
# Z has shape (3, 1)
# The first column is the coordinate along the highest-variance direction.
```

## What the gate checks

The gate computes the reference projection using NumPy's SVD implementation:

$$
Z_{\mathrm{ref}} = X V_k^\top .
$$

It compares the returned projection against this oracle using mean per-channel relative error. The metric `channel_rel_err` must satisfy

$$
\frac{1}{k}\sum_i
\frac{\lVert Z_i-\hat{Z}_i\rVert}{\lVert Z_i\rVert + 10^{-12}}
\le 10^{-6}.
$$

Solutions that swap the SVD factors, transpose the projection matrix incorrectly, or use the wrong component ordering will fail.
