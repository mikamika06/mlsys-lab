## Context

Gradient implementations are often written from symbolic derivatives. A small sign
or coefficient mistake can silently produce incorrect optimization behavior. A
common debugging technique is gradient checking: compare an analytic gradient to a
numerical approximation computed with finite differences.

For a scalar function $f:\mathbb{R}^d \rightarrow \mathbb{R}$, the partial
derivative with respect to coordinate $i$ can be approximated by the central
difference formula

$$
\frac{\partial f}{\partial x_i} \approx
\frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon},
$$

where $e_i$ is the unit vector for coordinate $i$ and $\epsilon$ is a small
step size.

The function used in this task is

$$
f(x) = \sum_i (i+1)x_i^3 + 2x_i^2 - 5x_i .
$$

The correct derivative for coordinate $i$ is

$$
\frac{\partial f}{\partial x_i} = 3(i+1)x_i^2 + 4x_i - 5 .
$$

## Task

Implement `fixed_gradient(x)`.

The function receives a one-dimensional NumPy array and returns a one-dimensional
NumPy array containing the analytic gradient of $f(x)$. The output must have the
same shape as the input and use floating point values.

The original analytic gradient contains a bug. Fix the derivative formula so it
matches a central finite-difference gradient check.

## Example

```python
import numpy as np

x = np.array([1.0, -2.0, 0.5])
g = fixed_gradient(x)

# f(x) = sum_i (i+1)x_i^3 + 2x_i^2 - 5x_i
# g should be approximately:
# [ 2.  3.  5.75]
```

## What the gate checks

The gate computes the reference gradient using central finite differences on the
same function definition. It compares `fixed_gradient` against that oracle using
the maximum absolute error

$$
\max_i |g_i - \hat{g}_i|.
$$

The error must be less than $10^{-5}$. This catches incorrect analytic gradients
that look plausible but do not match the actual derivative.
