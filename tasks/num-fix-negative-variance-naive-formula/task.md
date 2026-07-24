## Context

The variance of a sequence $x_1, x_2, \dots, x_n$ is often written using the
moment identity

$$
\mathrm{Var}(x) = E[x^2] - E[x]^2 .
$$

Although this formula is mathematically correct, evaluating the two terms
separately can lose precision when the values have a large offset and small
spread. The subtraction can amplify floating point error and may produce a
negative variance for data that should have non-negative variance.

Welford's online algorithm updates the mean and the accumulated squared
deviation without subtracting two large nearly equal values. For each new value
$x_k$, update

$$
\delta = x_k - m_{k-1},
$$

$$
m_k = m_{k-1} + \frac{\delta}{k},
$$

$$
M_k = M_{k-1} + \delta (x_k - m_k).
$$

The population variance is then

$$
\mathrm{Var}(x) = \frac{M_n}{n}.
$$

When all values share a very large offset, subtracting a constant from every
element before the online update can further reduce floating point loss because
variance is translation invariant:

$$
\mathrm{Var}(x) = \mathrm{Var}(x-c).
$$

## Task

Implement `stable_variance(x)`:

```python
def stable_variance(x: np.ndarray) -> float:
    ...
```

The function receives a one-dimensional NumPy array and returns the population
variance as a Python `float`. Use a numerically stable online variance
algorithm. Do not compute the result as `mean(x**2) - mean(x)**2`.

The input may contain values with a large constant offset and a small amount of
variation.

## Example

```python
import numpy as np

x = np.array([1e12, 1e12 + 1, 1e12 - 1], dtype=np.float64)
v = stable_variance(x)

# v is approximately 0.6666666666666666
```

## What the gate checks

The gate compares the returned value against NumPy's float64 variance reference.
The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert v_{\mathrm{candidate}} - v_{\mathrm{reference}} \rVert}
{\lVert v_{\mathrm{reference}} \rVert + 10^{-12}}
$$

must satisfy $\mathrm{rel\_err} \le 10^{-10}$ on shifted and ordinary fixtures.
A formula based on $E[x^2]-E[x]^2$ fails because the shifted fixture loses the
small variance during floating point subtraction.
