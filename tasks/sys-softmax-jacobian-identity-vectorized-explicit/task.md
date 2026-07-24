## Context

Softmax maps a vector of logits $z \in \mathbb{R}^m$ to probabilities

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}} .
$$

During a backward pass, the gradient with respect to the logits is often written as a Jacobian-vector product. The full softmax Jacobian for one row is

$$
J = \operatorname{diag}(p) - p p^\top .
$$

Materializing $J$ costs $O(m^2)$ memory for every row. The same result can be computed from the identity

$$
Jv = p \odot \left(v - (p^\top v)\mathbf{1}\right),
$$

where $\odot$ is elementwise multiplication. This form is used in memory-efficient attention backward implementations because it avoids constructing large intermediate matrices.

## Task

Implement `softmax_jacobian_vjp(p, dY)`:

```python
def softmax_jacobian_vjp(p: np.ndarray, dY: np.ndarray) -> np.ndarray:
    ...
```

The inputs are two-dimensional arrays with shape $(n, m)$:

- `p` contains softmax probabilities for each row.
- `dY` contains an upstream gradient for each probability output.

Return the Jacobian-vector product for each row, equivalent to multiplying every row by its explicit matrix $J$.

Use a vectorized NumPy implementation. Do not construct the full $m \times m$ Jacobian for any row.

The output must be a `float64` NumPy array with shape $(n, m)$.

## Example

```python
import numpy as np

p = np.array([[0.2, 0.3, 0.5]])
dY = np.array([[1.0, 2.0, 4.0]])

out = softmax_jacobian_vjp(p, dY)
# Equivalent to:
# (diag([0.2,0.3,0.5]) - [0.2,0.3,0.5]^T[0.2,0.3,0.5]) @ [1,2,4]
```

## What the gate checks

The gate computes a reference result by explicitly forming each row's Jacobian and multiplying it by the upstream gradient. The submitted implementation must match this NumPy oracle with relative error $\le 10^{-6}$.

A second guard checks allocations while running larger rows. Implementations that materialize the full $m \times m$ Jacobian create large temporary arrays and fail. The passing approach uses the identity

$$
dZ = p \odot (dY - \sum_i p_i dY_i)
$$

without allocating an $m \times m$ matrix.
