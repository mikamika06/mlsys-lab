## Context

The softplus function is a smooth approximation to ReLU and is defined as

$$
\operatorname{softplus}(x) = \log(1 + \exp(x)).
$$

A direct implementation is numerically unstable. For large positive values, $\exp(x)$ can overflow before the logarithm is applied. For large negative values, adding $1$ to a very small exponential can lose precision.

A stable reformulation separates the positive and negative cases using the identity

$$
\operatorname{softplus}(x) = \max(x,0) + \log(1+\exp(-|x|)).
$$

The term $\log(1+\exp(y))$ should be computed with `np.log1p` to preserve accuracy when $y$ is close to zero.

## Task

Implement `stable_softplus(x)`:

```python
def stable_softplus(x: np.ndarray) -> np.ndarray:
    ...
```

The function takes a NumPy array of any shape and returns a `float64` NumPy array of the same shape containing the softplus value of each element. The implementation should be vectorized and must correctly handle inputs with magnitude up to $10^3$.

Do not use Python loops.

## Example

```python
import numpy as np

x = np.array([-1000.0, 0.0, 1000.0])
y = stable_softplus(x)

# y is approximately:
# [0.0, 0.69314718, 1000.0]
```

## What the gate checks

The gate computes a NumPy reference using the stable identity

$$
\max(x,0) + \log1p(\exp(-|x|)).
$$

The maximum absolute error between the submitted implementation and the reference must satisfy

$$
\max_i |y_i-\hat{y}_i| < 10^{-10}.
$$

The test includes values with $|x|$ close to $1000$, where the naive implementation $\log(1+\exp(x))$ can overflow.
