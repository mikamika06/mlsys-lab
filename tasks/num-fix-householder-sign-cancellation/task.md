## Context

Householder transformations are used to build numerically stable QR decompositions. Given a vector $x \in \mathbb{R}^m$, a reflector can transform it toward a multiple of the first basis vector:

$$
H x = x - 2v\frac{v^\top x}{v^\top v},
$$

where $v$ is the reflector vector.

A common construction uses

$$
v = x - \lVert x \rVert e_1.
$$

This formula can suffer from catastrophic cancellation when the first element of $x$ is positive and close to $\lVert x \rVert$. The subtraction removes significant digits and can make the reflector inaccurate.

A stable sign choice uses

$$
v = x - \alpha e_1,
$$

where

$$
\alpha = -\operatorname{sign}(x_0)\lVert x\rVert .
$$

This avoids subtracting two nearly equal values. The resulting reflector should still produce a valid transformation that reconstructs the input accurately.

## Task

Implement `householder_fixed(x)`:

```python
def householder_fixed(x: np.ndarray) -> np.ndarray:
    ...
```

The function receives a one-dimensional NumPy array of floating point values and returns the Householder reflector matrix $H$ with shape $(m, m)$.

Construct the reflector using the stable sign convention. The returned matrix must satisfy

$$
Hx \approx -\operatorname{sign}(x_0)\lVert x\rVert e_1
$$

and should reconstruct the transformation accurately for cancellation-prone inputs.

Use NumPy operations only.

## Example

```python
import numpy as np

x = np.array([10.0, 1e-8, -2e-8])
H = householder_fixed(x)

y = H @ x
# y is approximately [-sqrt(100.0), 0.0, 0.0]
```

## What the gate checks

The gate builds several vectors and computes the oracle reflector directly with the stable Householder sign rule. It compares the submitted reconstruction against the oracle reconstruction using

$$
\max_i |a_i-b_i|.
$$

The `max_abs_err` value must be less than $10^{-10}$. Implementations that use the cancellation-prone formula $v=x-\lVert x\rVert e_1$ fail on inputs where the subtraction loses precision.
