## Context

GPTQ-style post-training quantization needs the inverse of the (damped)
layer Hessian $H = X^\top X + \lambda I$, where $X$ is the calibration
activation matrix and $\lambda$ a small damping term added to guarantee
$H$ is symmetric positive-definite (SPD). Production implementations never
call a generic dense inverse: since $H$ is SPD, they factor it with a
**Cholesky decomposition** $H = L L^\top$ (lower-triangular $L$), and get the
inverse from $L$ alone:

$$
H = L L^\top \quad\Longrightarrow\quad H^{-1} = (L L^\top)^{-1} = L^{-\top} L^{-1} = (L^{-1})^\top (L^{-1})
$$

This is both cheaper than a general inverse (Cholesky is $\sim\!2\times$
faster than LU for SPD matrices) and numerically more stable, because it
never forms $H^{-1}$ directly through a generic elimination path — it only
ever inverts (or solves against) the well-conditioned triangular factor $L$.

## Task

Implement `cholesky_inverse`:

```python
def cholesky_inverse(H: np.ndarray) -> np.ndarray:
    ...
```

* `H` — a 2-D NumPy array of shape $(n, n)$, symmetric positive-definite
  (already damped, so it is always invertible).

Compute the inverse **via the Cholesky factor**: first find the
lower-triangular $L$ with $H = L L^\top$, then obtain $L^{-1}$ and form
$H^{-1} = (L^{-1})^\top (L^{-1})$. Return $H^{-1}$ as an $(n,n)$ array.

## Example

```python
import numpy as np
X = np.array([[1.0, 0.5], [0.5, 2.0], [1.5, -1.0]])
H = X.T @ X + 0.1 * np.eye(2)   # damped Hessian, SPD

H_inv = cholesky_inverse(H)
np.allclose(H_inv @ H, np.eye(2))   # -> True
```

## What the gate checks

A single **rel_err** gate builds several random damped Hessians
$H = X^\top X + \lambda I$ of different sizes, computes the reference
inverse through the same Cholesky-factor route ($H^{-1} = L^{-\top}
L^{-1}$), and compares it to your returned matrix with the Frobenius-norm
relative error, requiring `< 1e-6`. Any shape mismatch or exception fails
the gate.
