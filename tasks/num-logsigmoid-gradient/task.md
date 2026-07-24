## Context

The log-sigmoid function is used in binary classification and probabilistic models. It is defined as

$$
\operatorname{logsigmoid}(x) = \log(\sigma(x)),
$$

where

$$
\sigma(x) = \frac{1}{1 + e^{-x}}.
$$

A direct implementation first computes $\sigma(x)$ and then takes a logarithm. This can be numerically unstable because the sigmoid value may underflow to zero for large negative inputs.

A stable identity is

$$
\operatorname{logsigmoid}(x) = -\log(1 + e^{-x}),
$$

which can be implemented with stable primitives such as `logaddexp`.

The derivative is

$$
\frac{d}{dx}\operatorname{logsigmoid}(x) = \sigma(-x) = \frac{1}{1+e^x}.
$$

## Task

Implement `logsigmoid_with_grad(x)`:

```python
def logsigmoid_with_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The function receives a NumPy array of any shape and returns:

- `value`: the stable elementwise log-sigmoid values.
- `grad`: the derivative of the log-sigmoid at each element.

Both returned arrays must have the same shape as `x` and use `float64`.

The implementation must remain stable for very large positive and negative inputs.

## Example

```python
import numpy as np

x = np.array([-2.0, 0.0, 2.0])

value, grad = logsigmoid_with_grad(x)

# value is approximately:
# [-2.12692801 -0.69314718 -0.12692801]

# grad is approximately:
# [0.88079708 0.5        0.11920292]
```

## What the gate checks

The gate creates a reference value implementation using NumPy stable operations and creates a gradient oracle using central finite differences:

$$
g(x) \approx \frac{f(x+h)-f(x-h)}{2h}.
$$

The value error is checked with

$$
\max_i |v_i-v_i^{ref}| \le 10^{-10}.
$$

The gradient error is checked with

$$
\max_i |g_i-g_i^{ref}| \le 10^{-9}.
$$

The cases include extreme inputs where a naive sigmoid followed by `log` loses precision.
