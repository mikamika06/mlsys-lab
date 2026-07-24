## Context

For a linear system $Ax=b$, the forward error measures the change in a solution
after the input has been perturbed. Let $x$ be the solution of the original
system and $\hat{x}$ be the solution of a perturbed system.

The relative forward error is

$$
\frac{\lVert \hat{x}-x\rVert}{\lVert x\rVert}.
$$

A backward error describes the relative size of the input perturbation. For a
right-hand-side perturbation $\Delta b$, the relative backward error is

$$
\frac{\lVert \Delta b\rVert}{\lVert b\rVert}.
$$

The condition number $\kappa(A)$ describes how much perturbations can be
amplified. The standard estimate is

$$
\frac{\lVert \hat{x}-x\rVert}{\lVert x\rVert}
\leq
\kappa(A)
\frac{\lVert \Delta b\rVert}{\lVert b\rVert}.
$$

A well-conditioned matrix has a smaller amplification factor, while an
ill-conditioned matrix can turn small backward errors into larger forward
errors.

## Task

Implement `forward_error_bound(A, b, delta_b)`:

```python
def forward_error_bound(A: np.ndarray, b: np.ndarray, delta_b: np.ndarray) -> tuple[float, float]:
    ...
```

The function must:

1. Solve $Ax=b$ to obtain $x$.
2. Solve $A\hat{x}=b+\Delta b$ to obtain $\hat{x}$.
3. Return:
   - `forward_error`: the measured relative forward error
     $\frac{\lVert \hat{x}-x\rVert}{\lVert x\rVert}$.
   - `bound`: the estimate
     $\kappa(A)\frac{\lVert \Delta b\rVert}{\lVert b\rVert}$.

Use NumPy linear algebra operations. Return Python floats.

## Example

```python
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
delta_b = np.array([1e-8, -2e-8])

forward_error, bound = forward_error_bound(A, b, delta_b)

# forward_error is the measured relative change in the solution.
# bound is the conditioning-based estimate.
```

## What the gate checks

The gate recomputes the expected values using NumPy as the reference
implementation.

The returned `forward_error` and `bound` must match the NumPy calculation with
relative error at most $10^{-10}$.

The gate also verifies that the numerical relationship

$$
\text{forward\_error} \leq \text{bound}
$$

holds for the tested systems.
