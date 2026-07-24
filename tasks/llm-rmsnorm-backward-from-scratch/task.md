## Context

RMSNorm normalizes a vector using only its root mean square magnitude. For an
input vector $x \in \mathbb{R}^d$, the forward operation is

$$
r = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},
$$

$$
y_i = \frac{x_i}{r}.
$$

Given an upstream gradient $g = \frac{\partial L}{\partial y}$, the backward
pass must compute the gradient with respect to the input:

$$
\frac{\partial L}{\partial x}.
$$

The derivative follows from the dependency of every output element on the shared
normalization factor $r$. A correct implementation must account for this
coupling instead of treating each element independently.

## Task

Implement `rmsnorm_backward(x, grad_y, eps=1e-5)`:

```python
def rmsnorm_backward(
    x: np.ndarray,
    grad_y: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    ...
```

The function receives one-dimensional NumPy arrays `x` and `grad_y` with the
same shape. `x` is the input to RMSNorm and `grad_y` is the gradient of a scalar
loss with respect to the normalized output. Return `dx`, the gradient of the
loss with respect to `x`.

Use NumPy operations and return a `float64` array.

## Example

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
grad_y = np.array([0.5, -1.0, 2.0])

dx = rmsnorm_backward(x, grad_y)
# dx contains the input gradient for the RMSNorm operation
```

## What the gate checks

The gate builds a numerical oracle using central finite differences of the RMSNorm
forward function. For each test case it compares the submitted `dx` against this
finite-difference gradient using the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert dx_{\mathrm{candidate}}-dx_{\mathrm{oracle}}\rVert_2}
{\lVert dx_{\mathrm{oracle}}\rVert_2 + 10^{-12}} .
$$

The submitted implementation passes when $\mathrm{rel\_err} < 10^{-4}$.
