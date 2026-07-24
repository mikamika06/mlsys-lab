## Context

Power iteration approximates the dominant eigenvector of a matrix by repeatedly applying the matrix and normalizing:

$$
x_{k+1} = \frac{A x_k}{\lVert A x_k \rVert}.
$$

For a matrix with eigenvalues ordered by magnitude,

$$
|\lambda_1| > |\lambda_2| \geq |\lambda_3| \geq \dots ,
$$

the direction error of power iteration decreases geometrically. The asymptotic convergence factor is determined by the spectral gap:

$$
\rho = \left|\frac{\lambda_2}{\lambda_1}\right|.
$$

A smaller value of $\rho$ means the dominant eigenvector is reached more quickly. The convergence factor can be estimated by measuring how the angle error changes between iterations.

## Task

Implement `estimate_convergence_rate(A)`.

The function receives a square NumPy array and returns a Python `float` containing the measured geometric decay ratio of power iteration.

Use a deterministic initial vector. Run power iteration to obtain an approximation of the dominant eigenvector, then run another power iteration sequence and measure the angle error relative to the dominant direction:

$$
e_k = \sqrt{1 - (x_k^\top v)^2}.
$$

Estimate the convergence rate from successive error ratios

$$
\frac{e_{k+1}}{e_k}
$$

after the initial transient iterations. Return a value close to

$$
\left|\frac{\lambda_2}{\lambda_1}\right|.
$$

Do not use `numpy.linalg.eig` or other eigensolvers.

## Example

```python
import numpy as np

A = np.array([[10.0, 0.0], [0.0, 8.0]])

rate = estimate_convergence_rate(A)
# close to 0.8
```

## What the gate checks

The grader computes the reference spectral ratio using NumPy's eigenvalue decomposition. It compares the returned value against the oracle value

$$
\rho = \left|\frac{\lambda_2}{\lambda_1}\right|.
$$

The relative error

$$
\mathrm{rel\_err} = \frac{|r-\rho|}{|\rho| + 10^{-12}}
$$

must be less than $0.05$.
