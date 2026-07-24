## Context

Power iteration is a simple algorithm for estimating the dominant eigenvector of a real symmetric matrix $A \in \mathbb{R}^{n\times n}$.  
Starting from an arbitrary non‑zero vector $x_0$, each step computes
$$
x_{t+1} = \frac{A\,x_t}{\lVert A\,x_t\rVert}\,.
$$
If the eigenvalues of $A$ are ordered by magnitude as
$\lambda_1 > |\lambda_2| \geqslant \dots \geqslant |\lambda_n|$, then the error in direction decays geometrically:
$$
\lVert x_t - v_1\rVert = O\!\left(\Bigl|\frac{\lambda_2}{\lambda_1}\Bigr|^t\right),
$$
where $v_1$ is a unit eigenvector associated with $\lambda_1$.  
The ratio
$$
r \;=\;\Bigl|\frac{\lambda_2}{\lambda_1}\Bigr|
$$
is called the *convergence factor*.

## Task

Implement `predict_error_curve(A, T)`:

```python
def predict_error_curve(A: np.ndarray, T: int) -> np.ndarray:
    ...
```

`A` is a real symmetric matrix of shape `(n, n)`.  
`T` is the number of iterations to report.  

The function must return a 1‑D NumPy array of length `T`, containing the predicted relative error at each iteration $t$ (with $t=0$ corresponding to the initial vector before any power iteration).  
The prediction assumes that the initial vector has a non‑zero component along the dominant eigenvector; the constant factor is ignored, so the curve starts at $1$ and decays geometrically with ratio $r$.

## Example

```python
import numpy as np
from predict_error_curve import predict_error_curve

# random symmetric matrix
rng = np.random.default_rng(0)
A_raw = rng.standard_normal((5, 5))
A = (A_raw + A_raw.T) / 2.0

T = 10
curve = predict_error_curve(A, T)
print(curve)
# e.g. [1.         0.73205081 0.53589858 0.39223232 0.28730154 0.21025655
#        0.15400044 0.11299933 0.08274892 0.06069261]
```

## What the gate checks

The grader computes a reference curve using the same algorithm on several random symmetric matrices.  
Your output is compared to that reference with the global relative L2 error
$$
\mathrm{rel\_err} = \frac{\lVert \text{curve}_{\text{ref}} - \text{curve}_{\text{your}}\rVert}
                        {\lVert \text{curve}_{\text{ref}}\rVert + 10^{-12}} .
$$
The gate requires $\mathrm{rel\_err} \leq 5\times10^{-2}$.
