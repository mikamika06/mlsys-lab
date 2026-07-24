## Context

Production pruning and sparsity-control systems often optimize mask logits instead of binary masks. A differentiable mask is obtained with the sigmoid function:

$$
m_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}},
$$

where $z$ is the vector of mask logits and $m$ is the current soft mask.

The current sparsity is measured as the fraction of pruned weights:

$$
s(z) = 1 - \frac{1}{n}\sum_{i=1}^{n} m_i .
$$

A Lagrangian penalty can enforce a target sparsity value $s_t$:

$$
L(z) = \frac{1}{n}\sum_{i=1}^{n}(m_i - 0.5)^2
+ \lambda (s(z) - s_t)^2 .
$$

The first term is a task-shaped regularizer that encourages mask values away from uncertain middle values. The second term is a shape constraint penalty. The gradient must be computed with respect to the original logits $z$, not the mask values.

For this loss,

$$
\frac{\partial m_i}{\partial z_i} = m_i(1-m_i),
$$

and the chain rule gives the gradient of the complete Lagrangian with respect to every logit.

## Task

Implement `shape_constraint_loss(logits, target_sparsity, lam)`:

```python
def shape_constraint_loss(
    logits: np.ndarray,
    target_sparsity: float,
    lam: float
) -> tuple[float, np.ndarray]:
    ...
```

The function receives a one-dimensional NumPy array of mask logits, a target sparsity value, and the Lagrange multiplier $\lambda$. Return a tuple containing:

1. The scalar loss $L(z)$ as a Python float.
2. A NumPy array containing $\frac{\partial L}{\partial z}$ with the same shape as `logits` and dtype `float64`.

Use the differentiable formulation above. Do not threshold logits into binary masks.

## Example

```python
import numpy as np

logits = np.array([-2.0, 0.0, 2.0])
loss, grad = shape_constraint_loss(logits, 0.5, 3.0)

# loss is a scalar
# grad has shape (3,)
```

## What the gate checks

The gate computes the loss and gradient from an independent numerical oracle. The gradient oracle uses central finite differences:

$$
g_i \approx \frac{L(z_i+h)-L(z_i-h)}{2h}.
$$

The returned loss and gradient are compared against the oracle result using the maximum absolute error:

$$
\max_j |x_j - x_j^{oracle}|.
$$

The maximum error must be below $10^{-5}$.
