## Context

For a linear layer with weight matrix $W \in \mathbb{R}^{d\times p}$ and input data $X \in \mathbb{R}^{n\times d}$, the Hessian of the squared‑error loss plus an $\ell_2$ regulariser is

$$
H = 2\, X X^\top + \lambda I_n,
$$

where $\lambda>0$ is the damping coefficient and $I_n$ is the $n\times n$ identity matrix.  
The inverse of this Hessian, $H^{-1}$, is required in many optimisation algorithms (e.g., quasi‑Newton methods).  A stable way to compute it is via a Cholesky factorisation:

$$
L = \operatorname{chol}(H),\qquad H^{-1} = L^{-\top}\!L^{-1}.
$$

## Task

Implement the function

```python
def hessian_and_inverse(X: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    ...
```

It must:

* Accept a 2‑D NumPy array `X` of shape `(n, d)` and a scalar damping `lam`.
* Return a tuple `(H, H_inv)` where:
  * `H` is the Hessian matrix $2\, X X^\top + \lambda I_n$.
  * `H_inv` is its inverse computed via Cholesky or any numerically stable method.
* Use only vectorised NumPy operations; no explicit Python loops.
* Ensure that both returned matrices are of dtype `float64`.

## Example

```python
import numpy as np
X = np.array([[1, 0], [0, 1]], dtype=np.float64)
lam = 0.5
H, H_inv = hessian_and_inverse(X, lam)
print(H)
# [[2.5 0. ]
#  [0.  2.5]]
print(H_inv)
# [[0.4 0. ]
#  [0.  0.4]]
```

## What the gate checks

The grader computes a reference Hessian and its inverse using NumPy:

```python
H_ref   = 2 * X @ X.T + lam * np.eye(n, dtype=np.float64)
H_inv_ref = np.linalg.inv(H_ref)          # or via Cholesky
```

It then evaluates the maximum absolute error

$$
\max\!\bigl(\,\|H - H_{\text{ref}}\|_\infty,\;\|H^{-1} - H_{\text{inv,ref}}\|_\infty\,\bigr)
$$

and requires this value to be at most $10^{-8}$.
