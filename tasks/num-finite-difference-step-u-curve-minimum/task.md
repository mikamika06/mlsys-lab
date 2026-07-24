## Context

Finite differences approximate derivatives by evaluating a function at nearby points. The central difference formula for a scalar function is

$$
f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}.
$$

The approximation error depends on the step size $h$. A large $h$ increases truncation error because the local linear approximation is less accurate. A very small $h$ can cause catastrophic cancellation because $f(x+h)$ and $f(x-h)$ become nearly equal and floating point subtraction loses significant digits.

For a fixed point $x$, the practical error often follows a U-shaped curve over a geometric sweep of step sizes. The best step is the value minimizing the measured error against a known analytic derivative:

$$
h^* = \operatorname*{argmin}_{h \in H}
\left|
\frac{f(x+h)-f(x-h)}{2h} - f'(x)
\right|.
$$

## Task

Implement `optimal_fd_step(f, df, x)`:

```python
def optimal_fd_step(f, df, x):
    ...
```

The function receives a callable `f`, its analytic derivative `df`, and a scalar evaluation point `x`. Return the positive step size $h$ from the fixed geometric grid

$$
H = \{10^k \mid k=-16,-15,\dots,-1\}.
$$

For each candidate step, compute the central finite-difference derivative estimate, measure its absolute error against `df(x)`, and return the step with the smallest error.

Do not use randomness. The returned value must be one of the grid values.

## Example

```python
import math

def f(x):
    return math.sin(x)

def df(x):
    return math.cos(x)

h = optimal_fd_step(f, df, 1.0)
# h is one of:
# 1e-16, 1e-15, ..., 1e-1
```

## What the gate checks

The gate computes the oracle answer by performing the same finite-difference sweep independently and compares the returned step to that oracle minimum. The relative error

$$
\mathrm{rel\_err} =
\frac{|h_{\mathrm{candidate}}-h_{\mathrm{oracle}}|}
{|h_{\mathrm{oracle}}|+\epsilon}
$$

must be less than $0.05$.

The cases include smooth functions where very small steps lose precision due to cancellation. A solution that always chooses a fixed step or ignores the measured finite-difference error will fail.
