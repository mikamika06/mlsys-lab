## Context

In Principal Component Analysis (PCA) we seek the directions of maximal variance in a dataset. For a centered data matrix $X \in \mathbb{R}^{n\times d}$, the covariance matrix is
$$C = \frac{1}{\,n-1\,}\; X^\top X.$$
The leading eigenvector of $C$ points along the first principal component.

Power iteration is a simple iterative scheme to approximate the dominant eigenpair. Starting from an arbitrary non‑zero vector $v_0$, we repeatedly apply
$$v_{k+1} \;=\; \frac{\,C\, v_k\,}{\lVert C\, v_k\rVert},$$
normalising after each multiplication so that $\lVert v_k\rVert=1$. Two subtle but crucial steps are:

* **Mean‑centering** – the covariance matrix is defined on centred data. Using raw $X$ yields a biased operator.
* **Per‑step renormalisation** – without normalising, the vector will grow or shrink exponentially and lose numerical stability.

Missing either step leads to an incorrect eigenvector.

## Task

Implement `leading_eigenvector(X: np.ndarray, num_iter: int = 1000) -> np.ndarray` that returns a unit‑norm vector approximating the leading eigenvector of the covariance matrix of $X$. Your implementation must:

1. Centre $X$ by subtracting its column means.
2. Compute the covariance operator implicitly (no explicit $C$).
3. Perform power iteration for `num_iter` steps, normalising after each multiplication.
4. Return a NumPy array of dtype `float64`.

The function should work for any 2‑D array with shape $(n,d)$ where $n>1$ and $d>0$.

## Example

```python
import numpy as np
X = np.array([[0, 0], [1, 0], [0, 2]], dtype=float)
v = leading_eigenvector(X, num_iter=500)
print(v)          # e.g. array([0.70710678, 0.70710678])
```

## What the gate checks

The grader computes a reference eigenvector using NumPy’s SVD on centred data and compares it to your output with the scorer `max_abs_err`. Because an eigenvector is defined up to sign, the grader aligns the signs before computing the error. Your solution must achieve
$$\mathrm{max\_abs\_err} \;\le\; 10^{-4}.$$
