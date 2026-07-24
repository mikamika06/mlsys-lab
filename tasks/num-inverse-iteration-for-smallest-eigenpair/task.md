## Context

Plain power iteration, $x \leftarrow A x / \lVert A x \rVert$, converges to
the eigenvector of the **largest**-magnitude eigenvalue of $A$. Applying
the same idea to $A^{-1}$ instead converges to the dominant eigenvector of
$A^{-1}$ — which is the eigenvector of $A$ whose eigenvalue has the
**smallest** magnitude, since if $A v = \lambda v$ then
$A^{-1} v = \lambda^{-1} v$, and the smallest $|\lambda|$ becomes the
largest $|\lambda^{-1}|$.

Rather than forming $A^{-1}$ explicitly, each step **solves a linear
system**:

$$
A y_k = x_{k-1}, \qquad x_k = \frac{y_k}{\lVert y_k \rVert_2}.
$$

After enough steps, $x_k$ has converged (up to sign) to the eigenvector
$v_{\min}$ of smallest $|\lambda|$, and the corresponding eigenvalue is
recovered from the Rayleigh quotient

$$
\lambda_{\min} \approx x_k^\top A\, x_k .
$$

## Task

Implement `inverse_iteration`:

```python
def inverse_iteration(A: np.ndarray, num_iters: int = 100, x0: np.ndarray | None = None):
    ...
```

* `A` — a square, invertible matrix (the test matrices are symmetric
  positive-definite, so all eigenvalues are real and positive).
* `num_iters` — number of inverse-iteration steps to run.
* `x0` — optional starting vector; if `None`, start from the uniform
  vector $\mathbf{1}/\sqrt{n}$.

Each step must **solve** $A y = x$ for $y$ (e.g. with `np.linalg.solve`)
— do **not** multiply by $A$, and do **not** explicitly invert $A$.
Renormalize after every step. Return `(eigval, eigvec)` where `eigval` is
the Rayleigh quotient of the final iterate and `eigvec` is unit-norm.

## Example

```python
import numpy as np
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
val, vec = inverse_iteration(A, num_iters=50)
print(val)                 # -> smallest eigenvalue of A, ~1.382
print(A @ vec - val * vec) # -> ~0 (residual)
```

## What the gate checks

For several random well-conditioned symmetric positive-definite matrices
(sizes 4–8, constructed so the smallest eigenvalue is clearly separated
from the second-smallest), the grader compares your output against
`np.linalg.eigh(A)`:

* **rel_err** — mean relative error between your returned eigenvalue and
  the true smallest eigenvalue from `np.linalg.eigh`. Must be below
  `1e-6`.
* **vec_err** — mean of `1 - |cos(θ)|` between your eigenvector and the
  reference eigenvector (sign is irrelevant, only the direction matters).
  Must be below `1e-6`.
