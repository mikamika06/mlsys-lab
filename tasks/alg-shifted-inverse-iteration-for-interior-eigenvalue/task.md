## Context

Power iteration finds the dominant eigenvalue of a matrix by repeatedly applying the
matrix to a vector. For a symmetric matrix $A$, the iteration

$$
x_{k+1} = \frac{A x_k}{\lVert A x_k\rVert}
$$

converges to the eigenvector associated with the largest magnitude eigenvalue.

Shifted inverse iteration targets an eigenvalue near a chosen shift $\sigma$. Instead
of multiplying by $A$, it applies the inverse of the shifted matrix:

$$
y_{k+1} = (A-\sigma I)^{-1}x_k,
$$

$$
x_{k+1} = \frac{y_{k+1}}{\lVert y_{k+1}\rVert}.
$$

For a symmetric matrix, the eigenvalue closest to $\sigma$ dominates the inverse
iteration because the inverse scales eigencomponents by

$$
\frac{1}{\lambda_i-\sigma}.
$$

The eigenvalue can be estimated from the converged vector using the Rayleigh quotient:

$$
\hat{\lambda} = \frac{x^\top A x}{x^\top x}.
$$

## Task

Implement `shifted_inverse_iteration(A, sigma, iterations)`.

The function takes a real symmetric NumPy array $A$ of shape $(n,n)$, a scalar shift
$\sigma$, and an integer iteration count. It must return a tuple
`(eigenvalue, eigenvector)`.

The returned `eigenvalue` should be a scalar float estimating the eigenvalue of $A$
closest to $\sigma$. The returned `eigenvector` should be a 1-D NumPy array with
unit Euclidean norm.

Use the shifted inverse iteration method. Solve the linear system
$(A-\sigma I)y=x$ at each iteration rather than explicitly forming the inverse.

## Example

```python
import numpy as np

A = np.array([[4.0, 0.0], [0.0, 10.0]])
value, vector = shifted_inverse_iteration(A, 9.0, 12)

# value is close to 10.0
# vector has norm close to 1.0
```

## What the gate checks

The gate computes the reference eigenvalue using NumPy's symmetric eigenvalue
solver and compares it with the value returned by the implementation. The reported
relative error

$$
\mathrm{rel\_err} =
\frac{|\hat{\lambda}-\lambda_{\mathrm{ref}}|}
{|\lambda_{\mathrm{ref}}|+10^{-12}}
$$

must be at most $10^{-5}$.

A method that performs ordinary power iteration will fail because it converges to the
largest-magnitude eigenvalue instead of the eigenvalue near the provided shift.
