## Context

A common differentiable masking pattern represents a soft selection over $k$ choices using
logits $z \in \mathbb{R}^k$. The probabilities are computed with softmax:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} .
$$

The expected value of a mask feature vector $v \in \mathbb{R}^k$ is the soft mask value

$$
m = \sum_i p_i v_i .
$$

A downstream scalar objective compares this expected mask to a target value $t$:

$$
L = (m - t)^2 .
$$

The gradient with respect to logits must account for the softmax Jacobian. The Jacobian
entries are

$$
\frac{\partial p_i}{\partial z_j} = p_i(\mathbf{1}_{i=j} - p_j).
$$

A production implementation must combine this Jacobian with the derivative of the
downstream loss rather than treating each probability independently.

## Task

Implement `expected_mask_grad(logits, values, target)`:

```python
def expected_mask_grad(
    logits: np.ndarray,
    values: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    ...
```

The arguments have shape:

- `logits`: a 2-D NumPy array with shape $(n, k)$.
- `values`: a 1-D NumPy array with shape $(k,)$ containing mask feature values.
- `target`: a 1-D NumPy array with shape $(n,)$ containing desired soft mask values.

Return an array with shape $(n, k)$ containing $\frac{\partial L}{\partial z}$ for each row.

Use NumPy operations. The returned dtype must be `float64`.

## Example

```python
import numpy as np

logits = np.array([[2.0, 0.0, -1.0]])
values = np.array([1.0, 0.0, 0.5])
target = np.array([0.7])

grad = expected_mask_grad(logits, values, target)
```

The output contains the gradient of

$$
L = \left(\sum_i p_i v_i - t\right)^2
$$

with respect to each input logit.

## What the gate checks

The gate computes a numerical oracle using central finite differences of the same scalar
objective. The submitted gradient must match the finite-difference gradient with
`max_abs_err < 1e-5`.

A shortcut based on independent sigmoid-style derivatives fails because softmax
probabilities are coupled through the full Jacobian.
