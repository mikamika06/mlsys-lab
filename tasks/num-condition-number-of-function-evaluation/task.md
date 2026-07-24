## Context

The condition number of evaluating a scalar function $f$ at an input $x$ measures how much a small relative perturbation in $x$ can change the output. For differentiable functions, the relative condition number is

$$
\kappa_f(x) = \left| \frac{x f'(x)}{f(x)} \right|.
$$

A large value means that small input errors can be amplified by the function evaluation.

For the natural logarithm,

$$
f(x) = \log(x),
$$

the derivative is

$$
f'(x) = \frac{1}{x}.
$$

Therefore the condition number simplifies to

$$
\kappa_{\log}(x) = \left| \frac{x(1/x)}{\log(x)} \right|
= \left| \frac{1}{\log(x)} \right|.
$$

This value becomes large when $x$ is close to $1$, because $\log(1)=0$ and the output is sensitive to small input changes.

## Task

Implement `log_condition_number(x)`:

```python
def log_condition_number(x: np.ndarray) -> np.ndarray:
    ...
```

The function receives a NumPy array of positive floating point values and returns a NumPy array of the relative condition number of evaluating `log(x)` at every element.

The returned array must have dtype `float64`. Use NumPy operations and avoid Python loops.

## Example

```python
import numpy as np

x = np.array([0.5, 1.01, 2.0])
k = log_condition_number(x)

# Equivalent to:
# np.abs(1.0 / np.log(x))
```

## What the gate checks

The gate computes the reference values using NumPy's implementation of the logarithm and the mathematical derivative of $\log(x)$. The returned values are compared using global relative error:

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}} - y_{\mathrm{reference}} \rVert_2}
{\lVert y_{\mathrm{reference}} \rVert_2 + 10^{-12}} .
$$

The error must satisfy $\mathrm{rel\_err} \le 10^{-6}$. The check includes values near $1$ where incorrect conditioning formulas produce noticeably different results.
