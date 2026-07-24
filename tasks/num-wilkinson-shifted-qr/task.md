## Context

For a real symmetric matrix $A \in \mathbb{R}^{n \times n}$, the eigendecomposition is

$$
A = Q \Lambda Q^\top ,
$$

where $Q$ is orthogonal and $\Lambda$ contains the eigenvalues. The QR algorithm repeatedly applies

$$
A_k = Q_kR_k, \qquad A_{k+1}=R_kQ_k .
$$

A practical symmetric QR implementation uses a Wilkinson shift to accelerate convergence. For the trailing block

$$
\begin{bmatrix}
a & b\\
b & d
\end{bmatrix},
$$

the shift is selected as the eigenvalue estimate closer to $d$:

$$
\mu =
d-\frac{b^2}{a-d+\operatorname{sign}(a-d)\sqrt{(a-d)^2+4b^2}} .
$$

The shifted iteration applies QR factorization to $A_k-\mu I$ and updates

$$
A_{k+1}=RQ+\mu I .
$$

When the last subdiagonal element becomes sufficiently small, the trailing eigenvalue is deflated and removed from the active matrix.

## Task

Implement `wilkinson_eigvals(A, max_iter=200)`:

```python
def wilkinson_eigvals(A: np.ndarray, max_iter: int = 200) -> np.ndarray:
    ...
```

The input is a real symmetric square NumPy array. Return all eigenvalues as a one-dimensional `float64` NumPy array sorted in ascending order.

Use QR iterations with Wilkinson shifts and deflation. You may use `np.linalg.qr`, but do not call NumPy eigensolvers such as `np.linalg.eig` or `np.linalg.eigh`.

## Example

```python
import numpy as np

A = np.array([
    [2.0, 1.0],
    [1.0, 2.0],
])

vals = wilkinson_eigvals(A)

# approximately:
# [1.0, 3.0]
```

## What the gate checks

The gate computes the reference eigenvalues using NumPy's symmetric eigensolver and compares them to the implementation output.

The relative error is

$$
\mathrm{rel\_err}
=
\frac{\lVert \hat{\lambda}-\lambda\rVert_2}
{\lVert \lambda\rVert_2+10^{-12}} .
$$

The implementation must satisfy $\mathrm{rel\_err}\leq 10^{-8}$.

The gate uses a fixed iteration budget where unshifted QR is insufficient on the tested matrices. A Wilkinson shift with deflation is required for convergence.
