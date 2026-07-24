## Context

The least squares problem finds the vector $x$ that minimizes the residual between a matrix-vector product and a target:

$$
\min_x \lVert Ax-b\rVert_2^2 ,
$$

where $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^{m}$.

For a full-rank matrix, QR decomposition factors the matrix as

$$
A = QR ,
$$

where $Q$ has orthonormal columns and $R$ is upper triangular. The least squares objective becomes

$$
\lVert QRx-b\rVert_2^2 = \lVert Rx-Q^Tb\rVert_2^2 .
$$

The solution is obtained by first computing

$$
y = Q^Tb
$$

and then solving the triangular system

$$
Rx=y .
$$

Using QR avoids the numerical instability caused by explicitly forming the normal equations.

## Task

Implement `least_squares_qr(A, b)`.

The function receives a 2-D NumPy array `A` with shape $(m,n)$ and a 1-D NumPy array `b` with shape $(m,)$. Assume $m \ge n$ and that the columns of `A` are linearly independent.

Return a 1-D NumPy array containing the least squares solution $x$ with shape $(n,)$.

Your implementation must use QR decomposition followed by triangular back substitution. Do not call `np.linalg.lstsq`, `np.linalg.solve`, or equivalent direct solver routines.

## Example

```python
import numpy as np

A = np.array([[1., 0.], [0., 1.], [1., 1.]])
b = np.array([1., 2., 2.])

x = least_squares_qr(A, b)
```

The returned vector minimizes

$$
\lVert Ax-b\rVert_2^2 .
$$

## What the gate checks

The grader computes an independent reference answer with NumPy's least squares oracle `np.linalg.lstsq`.

The relative error is

$$
\mathrm{rel\_err} =
\frac{\lVert x_{\mathrm{candidate}}-x_{\mathrm{reference}}\rVert_2}
{\lVert x_{\mathrm{reference}}\rVert_2+10^{-12}} .
$$

The gate requires $\mathrm{rel\_err}<10^{-8}$ across deterministic full-rank matrices, including ill-conditioned matrices where normal equations lose precision.
