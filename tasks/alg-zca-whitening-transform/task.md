## Context

Whitening is a linear transformation that decorrelates data and scales each feature to unit variance.  
Given a zero‑mean dataset $X \in \mathbb{R}^{n\times d}$, the covariance matrix is  

$$\Sigma = \frac{1}{\,n-1\,}\, X^\top X.$$

A *ZCA* (Zero‑Phase Component Analysis) whitening transform finds a matrix $W$ such that

$$Y = X W,$$

and the transformed data $Y$ satisfies $\operatorname{Cov}(Y)=I_d$.  
The ZCA solution keeps the data as close as possible to its original orientation by using the eigenvectors of $\Sigma$:

$$
\Sigma = U \Lambda U^\top,\qquad
W_{\text{ZCA}} = U\,\Lambda^{-1/2}\,U^\top.
$$

When $X$ is not centered, we first subtract its mean.

## Task

Implement the function `zca_whitening`:

```python
def zca_whitening(X: np.ndarray) -> np.ndarray:
    ...
```

The function receives a 2‑D NumPy array of shape $(n,d)$ and returns a new array of the same shape containing the ZCA‑whitened data.  
Use only NumPy; no explicit Python loops are required.

## Example

```python
import numpy as np
X = np.array([[1, 0], [0, 1], [-1, 0]])
Y = zca_whitening(X)
# Y has zero mean and Cov(Y) ≈ I_2
```

## What the gate checks

The grader computes the covariance matrix of the returned data and compares it to the identity matrix.  
It reports the maximum absolute entrywise difference:

$$\text{max_abs_err} = \max_{i,j}\,|\,\operatorname{Cov}(Y)_{ij}-I_{ij}\,|.$$

The solution must satisfy $\text{max_abs_err}\le 10^{-6}$ on a set of random test cases.  
Additionally the output shape must match the input shape.
