## Context

In automatic differentiation, a tensor $a \in \mathbb{R}^{s_1 \times \cdots \times s_k}$ is often **broadcast** to a larger target shape $S_t = (t_1, \ldots, t_m)$ (where $m \geq k$) during a forward computation. NumPy and PyTorch both align shapes from the right: a dimension of size 1 in the input can be replicated to any size $t_i > 1$, and a missing leading dimension is treated as size 1.

When backpropagating through a broadcast, the incoming gradient $\bar{y}$ has the **target** shape $S_t$, but the parameter's gradient must have the **input** shape $S_a = (s_1, \ldots, s_k)$. The backward reduction is **sum-to-shape**: for each dimension where the input had size 1 (or was absent) and the target had size $t_i > 1$, sum along that axis.

Concretely, pad $S_a$ with leading 1s to match $m$:

$$p_i = \begin{cases} 1 & \text{if } i < m - k \\ s_{i-(m-k)} & \text{otherwise} \end{cases}$$

Then the output is:

$$\bar{a}_{j_1,\ldots,j_k} = \sum_{\{i \;:\; p_i = 1,\; t_i > 1\}} \bar{y}_{j_1,\ldots,j_k}$$

where the sum collapses each broadcasted axis. In NumPy this is `np.sum` with `keepdims=True` followed by `reshape`.

## Task

Implement:

```python
import numpy as np

def sum_to_shape(grad: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Reduce grad by summing along broadcasted dimensions to produce input_shape."""
    ...
```

`grad` is a NumPy array whose shape is the broadcast target shape. `input_shape` is a tuple giving the original (smaller) shape. Return a NumPy array of shape `input_shape` whose dtype is `float64`, obtained by summing `grad` along every axis that was broadcast.

You may assume `input_shape` is always a valid broadcast source for `grad.shape`.

## Example

```python
import numpy as np

grad = np.arange(24.0).reshape(2, 3, 4)   # shape (2, 3, 4)
result = sum_to_shape(grad, (3, 4))
# result[i, j] = grad[0, i, j] + grad[1, i, j]
# result.shape == (3, 4), dtype == float64
```

## What the gate checks

One gate. The **max absolute error** between the student output and a NumPy reference across 9 test cases (varying dimensionality, leading broadcasts, trailing broadcasts, non-contiguous broadcasts, scalar inputs, identity cases) must be less than $10^{-9}$. The reference is computed at grading time using the same sum-then-reshape algorithm against NumPy's own `ndarray.sum`.
