## Context

Solving a linear system $Ax=b$ produces a solution $x$ that can change when the
right hand side $b$ changes. The sensitivity of the solution measures the
relative movement of the solution after a perturbation.

For a perturbation $\Delta b$, define

$$
x = A^{-1}b
$$

and

$$
x' = A^{-1}(b+\Delta b).
$$

The measured relative solution change is

$$
s =
\frac{\lVert x' - x \rVert_2}
{\lVert x \rVert_2}.
$$

This value is an observed amplification of the input change. It is connected to
the condition number of $A$, because ill-conditioned systems can amplify small
changes in the input.

## Task

Implement `solution_sensitivity(A, b, delta)`:

```python
def solution_sensitivity(A, b, delta):
    ...
```

The inputs are NumPy arrays. `A` is a square matrix, `b` is a vector, and
`delta` is the perturbation added to `b`.

Return a Python `float` containing

$$
\frac{\lVert \mathrm{solve}(A,b+\Delta b)-\mathrm{solve}(A,b)\rVert_2}
{\lVert \mathrm{solve}(A,b)\rVert_2}.
$$

Use `np.linalg.solve` or an equivalent direct solver. Do not compute the
inverse matrix explicitly.

## Example

```python
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([4.0, 5.0])
delta = np.array([0.01, -0.02])

s = solution_sensitivity(A, b, delta)
```

The result is the relative change in the solution vector.

## What the gate checks

The gate builds several test systems and computes the reference result using
NumPy's linear solver.

The relative error

$$
\mathrm{rel\_err} =
\frac{|s_{\mathrm{candidate}}-s_{\mathrm{reference}}|}
{|s_{\mathrm{reference}}|+10^{-12}}
$$

must be less than $0.05$.
