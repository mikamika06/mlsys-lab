## Context

Power iteration finds the dominant eigenvalue of a matrix by repeatedly applying the
matrix to a vector and normalizing. For a symmetric matrix $A$, if the largest
eigenvalue is separated from the rest, the iteration

$$
x_{k+1} = \frac{A x_k}{\lVert A x_k \rVert}
$$

converges to the eigenvector associated with the largest eigenvalue.

To find the second eigenvalue, first estimate the dominant eigenpair
$(\lambda_1, v_1)$ and remove its contribution using deflation:

$$
B = A - \lambda_1 v_1 v_1^\top .
$$

The dominant eigenvalue of $B$ is the second largest eigenvalue of $A$. Running
power iteration on $B$ gives an estimate of $\lambda_2$.

## Task

Implement `second_eigenvalue(A)`:

```python
def second_eigenvalue(A: np.ndarray) -> float:
    ...
```

The input is a real symmetric NumPy array. Return an estimate of the second
largest eigenvalue. Use deflation: estimate the largest eigenpair, construct the
deflated matrix, then run power iteration on the deflated matrix.

The returned value should be a Python float. Use NumPy operations for vector and
matrix calculations. The function may use a fixed iteration count because the
grader uses well-conditioned symmetric matrices.

## Example

```python
import numpy as np

A = np.array([
    [5.0, 0.0],
    [0.0, 2.0],
])

value = second_eigenvalue(A)
# value is close to 2.0
```

## What the gate checks

The gate compares the returned second eigenvalue with a NumPy oracle computed
from `numpy.linalg.eigvalsh`.

The relative error

$$
\mathrm{rel\_err} =
\frac{|\hat{\lambda}_2 - \lambda_2|}
{|\lambda_2| + 10^{-12}}
$$

must satisfy $\mathrm{rel\_err} < 10^{-5}$.
