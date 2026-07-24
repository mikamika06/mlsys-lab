## Context

An $n \times n$ matrix $Q$ is orthogonal when $Q^\top Q = I_n$, meaning every pair of column vectors has dot product $0$ and each column has unit length. The orthogonality residual quantifies how far a given matrix deviates from this ideal:

$$\varepsilon(Q) \;=\; \lVert Q^\top Q - I_n \rVert_{\max}
   \;=\; \max_{i,j}\, \bigl| (Q^\top Q)_{ij} - \delta_{ij} \bigr|$$

where $\delta_{ij}$ is the Kronecker delta. For a perfectly orthogonal matrix — such as one produced by Householder reflections in a QR decomposition — $\varepsilon(Q) = 0$ up to floating-point rounding ($\sim 10^{-15}$ in `float64`). Matrices that are only approximately orthogonal (e.g., after finite-precision Gram-Schmidt) will have a small but nonzero residual.

## Task

Implement `orthogonality_error(Q)`:

```python
import numpy as np

def orthogonality_error(Q: np.ndarray) -> float:
    """Return the max-abs element of (Q.T @ Q - I)."""
    ...
```

Given an $n \times n$ NumPy array `Q`, form the $n \times n$ matrix $R = Q^\top Q - I_n$ and return $\max_{i,j} |R_{ij}|$ as a plain Python `float`. The identity matrix should match `Q.shape[0]` exactly.

## Example

```python
import numpy as np

# Householder reflection about the plane spanned by (1, 1, 0) in R^3
v = np.array([1.0, 1.0, 0.0])
v = v / np.linalg.norm(v)
Q = np.eye(3) - 2.0 * np.outer(v, v)
print(orthogonality_error(Q))   # ≈ 0.0
```

A non-orthogonal matrix gives a larger residual:

```python
M = np.array([[1.0, 2.0], [3.0, 4.0]])
print(orthogonality_error(M))   # > 0
```

## What the gate checks

The single gate `max_abs_err` is the maximum absolute difference between the learner's returned scalar and the NumPy-computed reference $\lVert Q^\top Q - I \rVert_{\max}$ across seven test cases:

- identity matrix, Householder reflections (3×3 and a product of 10 in 10×10), 2D rotation, random orthogonal matrix from `np.linalg.qr`, a slightly perturbed orthogonal matrix, and a non-orthogonal $2 \times 2$ matrix.

The oracle computes the expected answer itself via `np.max(np.abs(Q.T @ Q - np.eye(n)))` — no hardcoded values. The gate requires the learner's answer to agree within $10^{-10}$.
