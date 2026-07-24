## Context

Finite differences estimate a derivative using function evaluations near a point.

The forward difference approximation is

$$
f'(x) \approx \frac{f(x+h)-f(x)}{h}.
$$

Its truncation error is first order:

$$
\frac{f(x+h)-f(x)}{h}=f'(x)+O(h).
$$

The central difference approximation uses values on both sides:

$$
f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}.
$$

The leading error terms cancel, giving second-order accuracy:

$$
\frac{f(x+h)-f(x-h)}{2h}=f'(x)+O(h^2).
$$

A log-log plot of error against step size reveals the convergence order. If the error behaves like $C h^p$, then the slope of the line fitted to $\log(h)$ and $\log(\mathrm{error})$ is approximately $p$.

## Task

Implement `finite_difference_error_orders(f, x, hs)`.

The function receives a scalar function `f`, a point `x`, and a one-dimensional NumPy array of positive step sizes `hs`. It must return a tuple:

```python
(forward_errors, central_errors)
```

where both outputs are NumPy arrays with the same shape as `hs`.

For each $h$ in `hs`, compute the absolute derivative estimation error of:

$$
\frac{f(x+h)-f(x)}{h}
$$

and

$$
\frac{f(x+h)-f(x-h)}{2h}.
$$

The derivative value should be estimated internally using a sufficiently small central finite difference. Do not use symbolic differentiation or external libraries.

Return floating point arrays.

## Example

```python
import numpy as np

def f(x):
    return np.sin(x)

forward, central = finite_difference_error_orders(
    f,
    0.7,
    np.array([1e-1, 1e-2, 1e-3, 1e-4])
)

# forward and central contain decreasing errors.
# A log-log fit gives a slope near 1 for forward
# and near 2 for central.
```

## What the gate checks

The gate computes a reference result using the same finite difference oracle method and compares the returned error curves.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y-\hat{y}\rVert_2}{\lVert y\rVert_2+\epsilon}
$$

between the submitted error curves and the reference curves must satisfy the required threshold.

The checked cases also verify that the resulting curves exhibit the expected numerical behavior: the forward difference has an error order near $1$ and the central difference has an error order near $2$.
