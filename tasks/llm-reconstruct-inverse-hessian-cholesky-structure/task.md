## Context

In many machine learning settings the Hessian matrix of a quadratic loss takes the form

$$H = X\,X^\top + \lambda I,$$

where $X\in\mathbb{R}^{n\times d}$ contains data rows, $\lambda>0$ is a regularisation constant and $I$ is the identity.  
The inverse $H^{-1}$ is required for second‑order optimisation or uncertainty estimation.  Direct inversion of an $n\times n$ matrix costs $O(n^3)$ operations and can be numerically unstable if $H$ is ill‑conditioned.

A stable way to compute $H^{-1}$ is via the Cholesky factorisation

$$H = LL^\top,$$

with $L$ lower triangular.  The inverse can then be obtained by solving two triangular systems:

\[
L\,X = I \quad\text{and}\quad L^\top H^{-1} = X .
\]

This avoids explicit inversion and exploits the structure of $L$.

## Task

Implement `reconstruct_inverse_hessian` that takes a 2‑D NumPy array `A` (shape $(n,d)$) and a scalar regulariser `lambda_reg`, builds the matrix

$$H = A\,A^\top + \lambda_{\text{reg}} I_n,$$

computes its Cholesky factorisation, and returns the inverse $H^{-1}$ as an `(n,n)` array of type `float64`.

```python
def reconstruct_inverse_hessian(A: np.ndarray, lambda_reg: float) -> np.ndarray:
    ...
```

The returned matrix must be exactly the inverse of $H$ up to a relative error of $10^{-4}$.

## Example

```python
import numpy as np
A = np.array([[1., 0.], [0., 2.]])
inv_H = reconstruct_inverse_hessian(A, 0.5)
print(inv_H)
# [[ 0.66666667 -0.16666667]
#  [-0.16666667  0.33333333]]
```

## What the gate checks

The grader computes a reference inverse using NumPy’s Cholesky routine and compares it to your output with the relative error metric

$$\mathrm{rel\_err} = \frac{\lVert H^{-1}_{\text{cand}} - H^{-1}_{\text{ref}}\rVert}{\lVert H^{-1}_{\text{ref}}\rVert}.$$

The solution must satisfy $\mathrm{rel\_err}\le 10^{-4}$.
